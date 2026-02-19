import os, asyncio, logging, re
from typing import Optional, List, TypedDict
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

# --- 1. SETTINGS ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_PATH = "vectorstore/db_faiss"

# --- 2. DATA MODELS ---
class PatientProfile(BaseModel):
    gender: str = Field(description="Gender: male/female/unknown")
    age: int = Field(description="Age")
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

# --- 3. AI INITIALIZATION ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)
vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0.1)

# --- 4. GRAPH NODES ---

async def analyze_node(state: AgentState):
    """Data extraction and 'red flag' search"""
    text = state['input']
    
    # Emergency situation detector
    emergency_keywords = ["numb", "face", "chest", "breathe", "consciousness", "paralysis", "stroke"]
    is_emergency = any(word in text.lower() for word in emergency_keywords)

    structured_llm = llm.with_structured_output(PatientProfile)
    try:
        # Instruct the model to ignore old context
        profile_obj = await structured_llm.ainvoke(f"Extract current patient profile from this text ONLY: {text}")
        p_dict = profile_obj.model_dump()
    except Exception as e:
        logging.error(f"Extraction error: {e}")
        p_dict = {"gender": "unknown", "age": 50, "history": text}

    # If the AI missed the age, try Regex
    if p_dict.get('age') == 50:
        match = re.search(r'(\d{1,3})', text)
        if match: p_dict['age'] = int(match.group(1))

    return {"patient_data": p_dict, "is_emergency": is_emergency, "context": []}

async def retrieve_node(state: AgentState):
    """Database search with extended query"""
    p = state['patient_data']
    search_query = f"prevention screening vaccination {p['gender']} {p['age']} years old {p['history']}"
    
    docs = vector_db.similarity_search(search_query, k=12)
    return {"context": [d.page_content for d in docs]}

async def generate_node(state: AgentState):
    """Final answer with strict filtering"""
    p = state['patient_data']
    age = p['age']
    gender = p['gender']
    
    emergency_note = ""
    if state.get('is_emergency'):
        emergency_note = "⚠️ ATTENTION: The described symptoms may require emergency care. Immediately go to the emergency room (Miyun) or call an ambulance (101).\n\n"

    lang_inst = {
        "ru": f"in Russian for a patient {age} years old. Structure: 1. Screening, 2. Vaccination.",
        "he": f"בעברית עבור מטופל בן {age}. מבנה: 1. סקר, 2. חיסונים.",
        "en": f"in medical English for a {age} years old patient. Structure: 1. Screening, 2. Vaccinations."
    }

    prompt = f"""
    Role: You are a general practitioner in Israel. You write a prevention plan according to Tabenkin.
    PATIENT: gender {gender}, age {age}, history {p['history']}.
    
    STRICT RULES:
    1. USE ONLY CONTEXT. If there is no data in the context for {age} years - do not invent it.
    2. GENDER FILTERING: If the patient is male - remove mammography, smears and osteoporosis for women.
    3. AGE FILTERING: If the patient is {age} years old, remove recommendations for other ages. (For example, if colonoscopy is from 50, and the patient is 25 - remove it).
    4. If the age is 75-80+, write that screenings are carried out according to the individual decision of the doctor.
    5. IGNORE HEART FAILURE if it is not in the current history: {p['history']}.
    
    CONTEXT:
    {' '.join(state['context'])}
    
    ANSWER {lang_inst.get(state['language'], 'ru')}
    """
    
    response = await llm.ainvoke(prompt)
    return {"answer": emergency_note + response.content}

# Graph Assembly
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph_app = workflow.compile()

# --- 5. TELEGRAM LOGIC ---
router = Router()

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Русский 🇷🇺", callback_data="lang_ru")],
        [InlineKeyboardButton(text="English 🇺🇸", callback_data="lang_en")],
        [InlineKeyboardButton(text="עברית 🇮🇱", callback_data="lang_he")]
    ])
    await message.answer("Choose a language / Select language:", reply_markup=kb)

@router.callback_query(F.data.startswith("lang_"))
async def set_lang(callback: types.CallbackQuery, state: FSMContext):
    lang = callback.data.split("_")[1]
    await state.update_data(language=lang)
    await callback.message.answer("Describe the patient (gender, age, symptoms/anamnesis):")
    await callback.answer()

@router.message()
async def handle_question(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    lang = u_data.get("language", "ru")
    wait_msg = await message.answer("🔍 Analyzing data...")
    
    try:
        # Calling the graph. Each call is a new state.
        result = await graph_app.ainvoke({"input": message.text, "language": lang})
        answer = result["answer"]
        
        # Escaping special characters for Telegram HTML
        safe_answer = answer.replace('<', '&lt;').replace('>', '&gt;')
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_answer).replace('* ', '• ')
        
        await wait_msg.edit_text(html[:4000], parse_mode="HTML")
    except Exception as e:
        logging.error(f"Error: {e}")
        await wait_msg.edit_text("An error occurred. Please try rephrasing your request.")

async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
