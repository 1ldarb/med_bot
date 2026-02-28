import os, asyncio, logging, re
from typing import Optional, List, TypedDict, Dict, Any, Literal
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F, Router
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage

from pydantic import BaseModel, Field, field_validator
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai

# --- 1. SETTINGS ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_PATH = "vectorstore/db_faiss"

genai.configure(api_key=API_KEY)

# --- 2. DATA MODELS ---
class PatientProfile(BaseModel):
    gender: str = Field(description="Gender: male/female/unknown")
    age: int = Field(description="Age")
    smoking_status: bool = Field(default=False, description="Patient smokes")
    diabetes: bool = Field(default=False, description="Patient has diabetes")
    family_history_cancer: bool = Field(default=False, description="Family history of cancer")
    history: Optional[str] = Field(default="no complaints", description="Medical history or current symptoms")

    @field_validator('age')
    @classmethod
    def check_age(cls, v):
        if v < 0 or v > 120: return 50
        return v

class AgentState(TypedDict):
    input: str
    language: str
    patient_data: Optional[dict]
    context: List[str]
    answer: str
    is_emergency: bool
    format_for_emr: bool
    media_path: Optional[str]
    intent: str

# --- 3. AI INITIALIZATION ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)
vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0.1)

# --- 4. GRAPH NODES ---

async def router_node(state: AgentState):
    """Determines the intent of the user's request based on text and media."""
    text = state.get('input', '')
    media = state.get('media_path')
    
    if media and media.endswith('.ogg'):
        return {"intent": "voice_to_soap"}
    elif media and (media.endswith('.jpg') or media.endswith('.png')):
        # Let's assume photos are hospital summaries for now.
        return {"intent": "sikum"}
    
    # Check simple text triggers
    lower_text = text.lower()
    if lower_text.startswith('/explain'):
        return {"intent": "explain"}
    elif lower_text.startswith('/hafnaya'):
        return {"intent": "hafnaya"}
    elif lower_text.startswith('/sikum'):
        return {"intent": "sikum"}

    class IntentObj(BaseModel):
        intent: str = Field(description="Categorize into exactly one: hafnaya, explain, sikum, prevention")
    
    prompt = f"""
    Classify the intent of the doctor's request into EXACTLY ONE of the following:
    - hafnaya: Asking to draft a referral, justification, or Form 17.
    - explain: Asking to explain or translate medical instructions for a patient.
    - sikum: Asking to summarize a hospital discharge (Sikum Ishpuz) or medical document.
    - prevention: Providing patient details for prevention, screening, or vaccination plan, or chronic medication check.
    
    Request: {text}
    """
    
    structured_llm = llm.with_structured_output(IntentObj)
    try:
        res = await structured_llm.ainvoke(prompt)
        intent = res.intent if res.intent in ["hafnaya", "explain", "sikum", "prevention"] else "prevention"
    except Exception as e:
        logging.error(f"Router error: {e}")
        intent = "prevention"
        
    return {"intent": intent}

async def sikum_node(state: AgentState):
    text = state.get('input', '')
    media = state.get('media_path')
    
    prompt = "You are an assistant for a family doctor in Israel. Summarize this hospital discharge (Sikum Ishpuz) or medical document. Output: 1. Новые диагнозы (New diagnoses). 2. Изменения в лечении (Medication changes). 3. Задачи для семейного врача (Tasks for family doctor)."
    
    if media:
        myfile = genai.upload_file(media)
        m = genai.GenerativeModel("gemini-2.0-flash")
        resp = await m.generate_content_async([myfile, prompt, text])
        myfile.delete()
        answer = resp.text
    else:
        resp = await llm.ainvoke(prompt + "\n\nText: " + text)
        answer = resp.content
    return {"answer": answer}

async def hafnaya_node(state: AgentState):
    prompt = f"Role: You are a family doctor in Israel. Write a formal medical referral/justification (Hafnaya/Tofes 17) in formal medical Hebrew for the HMO (Kupat Holim) based on this request: {state.get('input', '')}. Make it professional, concise, and persuasive."
    resp = await llm.ainvoke(prompt)
    return {"answer": resp.content}

async def explain_node(state: AgentState):
    prompt = f"Role: You are an empathetic family doctor. Translate and simplify these medical instructions for a patient without medical jargon, using empathy and a friendly tone. Write the explanation in this language: {state.get('language', 'ru')}.\nInstructions to explain: {state.get('input', '')}"
    resp = await llm.ainvoke(prompt)
    return {"answer": resp.content}

async def soap_node(state: AgentState):
    media = state.get('media_path')
    prompt = "Transcribe this audio dictation and format it as a professional medical SOAP note (Subjective, Objective, Assessment, Plan) in medical Hebrew, ready to be pasted into the patient's EMR (Chameleon/Clinic)."
    
    if media:
        myfile = genai.upload_file(media)
        m = genai.GenerativeModel("gemini-2.0-flash")
        resp = await m.generate_content_async([myfile, prompt])
        myfile.delete()
        answer = resp.text
    else:
        answer = "No audio provided for SOAP transcription."
    return {"answer": answer}

async def analyze_node(state: AgentState):
    """Data extraction and 'red flag' search (Prevention intent)"""
    text = state.get('input', '')
    prev_p_dict = state.get('patient_data') or {}
    
    emergency_keywords = ["numb", "face", "chest", "breathe", "consciousness", "paralysis", "stroke"]
    is_emergency = any(word in text.lower() for word in emergency_keywords)

    structured_llm = llm.with_structured_output(PatientProfile)
    try:
        prompt_text = f"Extract current patient profile from this text ONLY: {text}. Previous patient data (if any): {prev_p_dict}. Update the previous data with new info."
        profile_obj = await structured_llm.ainvoke(prompt_text)
        p_dict = profile_obj.model_dump()
    except Exception as e:
        logging.error(f"Extraction error: {e}")
        if isinstance(prev_p_dict, dict) and prev_p_dict:
            p_dict = {str(k): v for k, v in prev_p_dict.items()}
        else:
            p_dict = {"gender": "unknown", "age": 50, "smoking_status": False, "diabetes": False, "family_history_cancer": False, "history": text}
        p_dict["history"] = text

    if p_dict.get('age') == 50:
        match = re.search(r'(\d{1,3})', text)
        if match: p_dict['age'] = int(match.group(1))

    return {"patient_data": p_dict, "is_emergency": is_emergency, "context": []}

async def retrieve_node(state: AgentState):
    p_data = state.get('patient_data')
    p: dict = p_data if isinstance(p_data, dict) else {}
    search_query = f"prevention screening vaccination {p.get('gender')} {p.get('age')} years old smoking:{p.get('smoking_status')} diabetes:{p.get('diabetes')} family_cancer:{p.get('family_history_cancer')} {p.get('history')}"
    
    try:
        docs = vector_db.similarity_search(search_query, k=12)
        context_list = []
        for d in docs:
            page_num = d.metadata.get('page', 'Unknown')
            context_list.append(f"[Page {page_num}]: {d.page_content}")
    except Exception as e:
        logging.error(f"Retriever error: {e}")
        context_list = []
        
    return {"context": context_list}

async def generate_node(state: AgentState):
    p_data = state.get('patient_data')
    p: dict = p_data if isinstance(p_data, dict) else {}
    age = p.get('age')
    gender = p.get('gender')
    
    emergency_note = ""
    if state.get('is_emergency'):
        emergency_note = "⚠️ ATTENTION: The described symptoms may require emergency care. Immediately go to the emergency room (Miyun) or call an ambulance (101).\n\n"

    lang_inst = {
        "ru": f"in Russian for a patient {age} years old. Structure: 1. Screening, 2. Vaccination.",
        "he": f"בעברית עבור מטופל בן {age}. מבנה: 1. סקר, 2. חיסונים.",
        "en": f"in medical English for a {age} years old patient. Structure: 1. Screening, 2. Vaccinations."
    }
    
    emr_format = "\nFORMAT REQUIREMENT: Write a concise SOAP note or EMR summary suitable for copying into a medical record, avoiding fluff." if state.get('format_for_emr') else ""

    prompt = f"""
    Role: You are a general practitioner in Israel. You write a prevention plan according to Tabenkin.
    PATIENT: gender {gender}, age {age}, smoking: {p.get('smoking_status')}, diabetes: {p.get('diabetes')}, family history of cancer: {p.get('family_history_cancer')}, history {p.get('history')}.
    
    STRICT RULES:
    1. USE ONLY CONTEXT. If there is no data in the context for {age} years - do not invent it.
    2. GENDER FILTERING: If the patient is male - remove mammography, smears and osteoporosis for women.
    3. AGE FILTERING: If the patient is {age} years old, remove recommendations for other ages. (For example, if colonoscopy is from 50, and the patient is 25 - remove it).
    4. If the age is 75-80+, write that screenings are carried out according to the individual decision of the doctor.
    5. IGNORE HEART FAILURE if it is not in the current history: {p.get('history')}.
    6. CLACULATORS: If the patient's age and risk factors require SCORE or FRAX assessment according to the context, add a proactive reminder for the doctor.
    7. CITATIONS: ALWAYS cite the source [Page X] from the context for each recommendation.
    8. CHRONIC MEDS CHECKER: If the patient takes chronic medications mentioned in history (like ACE inhibitors, statins, ARBs, diuretics), proactively remind the doctor to check required annual blood tests (e.g. Potassium, Creatinine, Liver enzymes) before renewing prescriptions.
    {emr_format}
    
    CONTEXT:
    {' '.join(state.get('context', []))}
    
    ANSWER {lang_inst.get(state.get('language', 'ru'), 'ru')}
    """
    
    try:
        response = await llm.ainvoke(prompt)
        text = response.content
    except Exception as e:
        logging.error(f"Generation error: {e}")
        text = "Sorry, an error occurred during generation."
        
    return {"answer": emergency_note + text}

# Conditional routing function
def route_intent(state: AgentState) -> str:
    return state.get("intent", "prevention")

# Graph Assembly
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("sikum", sikum_node)
workflow.add_node("hafnaya", hafnaya_node)
workflow.add_node("explain", explain_node)
workflow.add_node("soap", soap_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", route_intent, {
    "sikum": "sikum",
    "hafnaya": "hafnaya",
    "explain": "explain",
    "voice_to_soap": "soap",
    "prevention": "analyze"
})
workflow.add_edge("sikum", END)
workflow.add_edge("hafnaya", END)
workflow.add_edge("explain", END)
workflow.add_edge("soap", END)
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
graph_app = workflow.compile(checkpointer=memory)

# --- 5. TELEGRAM LOGIC ---
router = Router()

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Русский 🇷🇺", callback_data="lang_ru")],
        [InlineKeyboardButton(text="English 🇺🇸", callback_data="lang_en")],
        [InlineKeyboardButton(text="עברית 🇮🇱", callback_data="lang_he")],
        [InlineKeyboardButton(text="Формат ЭМК (SOAP)", callback_data="emr_format_toggle")]
    ])
    welcome_text = (
        "👩‍⚕️ Добро пожаловать, доктор!\n\n"
        "Я ваш медицинский ИИ-ассистент. Я могу:\n"
        "1️⃣ **Проверять профилактику/онкоскрининги (по Табенкину)**: просто опишите пациента.\n"
        "2️⃣ **Создавать обоснования для Тофес 17**: напишите `/hafnaya Пациент...`\n"
        "3️⃣ **Сумаризировать выписки (Sikum)**: скиньте фото или напишите `/sikum Текст...`\n"
        "4️⃣ **Переводить сложное для пациентов**: напишите `/explain Инструкция...`\n"
        "5️⃣ **Генерировать SOAP-записи**: просто **запишите голосовое сообщение** 🎤.\n\n"
        "Выберите язык рекомендаций:"
    )
    await message.answer(welcome_text, reply_markup=kb, parse_mode="Markdown")

@router.callback_query(F.data == "emr_format_toggle")
async def toggle_emr(callback: types.CallbackQuery, state: FSMContext):
    u_data = await state.get_data()
    is_emr = not u_data.get("format_for_emr", False)
    await state.update_data(format_for_emr=is_emr)
    status = "включен" if is_emr else "отключен"
    await callback.answer(f"Формат для ЭМК {status}.")

@router.callback_query(F.data.startswith("lang_"))
async def set_lang(callback: types.CallbackQuery, state: FSMContext):
    lang = callback.data.split("_")[1]
    await state.update_data(language=lang)
    await callback.message.answer("Опишите пациента или используйте команды (/hafnaya, /explain, /sikum). Можно отправить голос или фото.")
    await callback.answer()

async def process_media_message(message: types.Message, state: FSMContext, bot: Bot, media_id: str, file_ext: str, input_text: str):
    u_data = await state.get_data()
    lang = u_data.get("language", "ru")
    format_for_emr = u_data.get("format_for_emr", False)
    
    wait_msg = await message.answer("🔍 Анализирую данные...")
    file_path = f"tmp_{media_id}{file_ext}"
    
    try:
        # Download media
        file_info = await bot.get_file(media_id)
        await bot.download_file(file_info.file_path, file_path)
        
        safe_text = re.sub(r'\b\d{9}\b', '[ID_REMOVED]', input_text)
        config = {"configurable": {"thread_id": str(message.chat.id)}}
        
        result = await graph_app.ainvoke({
            "input": safe_text, 
            "language": lang, 
            "format_for_emr": format_for_emr,
            "media_path": file_path
        }, config=config)
        
        answer = result.get("answer", "")
        safe_answer = answer.replace('<', '&lt;').replace('>', '&gt;')
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_answer).replace('* ', '• ')
        
        await wait_msg.edit_text(html[:4000], parse_mode="HTML")
    except Exception as e:
        logging.error(f"Error processing media: {e}")
        await wait_msg.edit_text("Произошла ошибка при обработке файла.")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.message(F.voice)
async def handle_voice(message: types.Message, state: FSMContext, bot: Bot):
    await process_media_message(message, state, bot, message.voice.file_id, ".ogg", "")

@router.message(F.photo)
async def handle_photo(message: types.Message, state: FSMContext, bot: Bot):
    # Take the highest resolution photo
    photo = message.photo[-1]
    caption = message.caption or ""
    await process_media_message(message, state, bot, photo.file_id, ".jpg", caption)

@router.message(F.document)
async def handle_document(message: types.Message, state: FSMContext, bot: Bot):
    caption = message.caption or ""
    await process_media_message(message, state, bot, message.document.file_id, ".pdf", caption)

@router.message()
async def handle_question(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    lang = u_data.get("language", "ru")
    format_for_emr = u_data.get("format_for_emr", False)
    
    wait_msg = await message.answer("🔍 Анализирую данные...")
    
    try:
        safe_text = re.sub(r'\b\d{9}\b', '[ID_REMOVED]', message.text)
        config = {"configurable": {"thread_id": str(message.chat.id)}}
        
        result = await graph_app.ainvoke({
            "input": safe_text, 
            "language": lang, 
            "format_for_emr": format_for_emr,
            "media_path": None
        }, config=config)
        
        answer = result.get("answer", "")
        safe_answer = answer.replace('<', '&lt;').replace('>', '&gt;')
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_answer).replace('* ', '• ')
        
        await wait_msg.edit_text(html[:4000], parse_mode="HTML")
    except Exception as e:
        logging.error(f"Error: {e}")
        await wait_msg.edit_text("Произошла ошибка. Пожалуйста, попробуйте перефразировать.")

async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
