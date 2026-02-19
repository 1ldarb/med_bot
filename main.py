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

# --- 1. НАСТРОЙКИ ---
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_PATH = "vectorstore/db_faiss"

# --- 2. МОДЕЛИ ДАННЫХ ---
class PatientProfile(BaseModel):
    gender: str = Field(description="Пол: male/female/unknown")
    age: int = Field(description="Возраст")
    history: Optional[str] = Field(default="жалоб нет", description="Анамнез или текущие симптомы")

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

# --- 3. ИНИЦИАЛИЗАЦИЯ ИИ ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)
vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0.1)

# --- 4. УЗЛЫ ГРАФА ---

async def analyze_node(state: AgentState):
    """Извлечение данных и поиск 'красных флагов'"""
    text = state['input']
    
    # Детектор экстренных ситуаций
    emergency_keywords = ["онемел", "лицо", "груди", "дышать", "сознание", "паралич", "инсульт"]
    is_emergency = any(word in text.lower() for word in emergency_keywords)

    structured_llm = llm.with_structured_output(PatientProfile)
    try:
        # Инструктируем модель игнорировать старый контекст
        profile_obj = await structured_llm.ainvoke(f"Extract current patient profile from this text ONLY: {text}")
        p_dict = profile_obj.model_dump()
    except Exception as e:
        logging.error(f"Extraction error: {e}")
        p_dict = {"gender": "unknown", "age": 50, "history": text}

    # Если ИИ пропустил возраст, пробуем Regex
    if p_dict.get('age') == 50:
        match = re.search(r'(\d{1,3})', text)
        if match: p_dict['age'] = int(match.group(1))

    return {"patient_data": p_dict, "is_emergency": is_emergency, "context": []}

async def retrieve_node(state: AgentState):
    """Поиск по базе с расширенным запросом"""
    p = state['patient_data']
    search_query = f"профилактика скрининг вакцинация {p['gender']} {p['age']} лет {p['history']}"
    
    docs = vector_db.similarity_search(search_query, k=12)
    return {"context": [d.page_content for d in docs]}

async def generate_node(state: AgentState):
    """Финальный ответ с жесткой фильтрацией"""
    p = state['patient_data']
    age = p['age']
    gender = p['gender']
    
    emergency_note = ""
    if state.get('is_emergency'):
        emergency_note = "⚠️ ВНИМАНИЕ: Описанные симптомы могут требовать экстренной помощи. Немедленно обратитесь в приемный покой (Миюн) или вызовите скорую помощь (101).\n\n"

    lang_inst = {
        "ru": f"на русском языке для пациента {age} лет. Структура: 1. Скрининг, 2. Вакцинация.",
        "he": f"בעברית עבור מטופל בן {age}. מבנה: 1. סקר, 2. חיסונים.",
        "en": f"in medical English for a {age} years old patient. Structure: 1. Screening, 2. Vaccinations."
    }

    prompt = f"""
    Роль: Ты врач-терапевт в Израиле. Пишешь план профилактики по Табенкину.
    ПАЦИЕНТ: пол {gender}, возраст {age}, история {p['history']}.
    
    СТРОГИЕ ПРАВИЛА:
    1. ИСПОЛЬЗУЙ ТОЛЬКО КОНТЕКСТ. Если в контексте нет данных для {age} лет — не выдумывай их.
    2. ФИЛЬТРАЦИЯ ПО ПОЛУ: Если пациент male — удали маммографию, мазки и остеопороз для женщин.
    3. ФИЛЬТРАЦИЯ ПО ВОЗРАСТУ: Если пациенту {age} лет, удали рекомендации для других возрастов. (Например, если колоноскопия с 50, а пациенту 25 — удали её).
    4. Если возраст 75-80+, пиши, что скрининги проводятся по индивидуальному решению врача.
    5. ИГНОРИРУЙ СЕРДЕЧНУЮ НЕДОСТАТОЧНОСТЬ, если её нет в текущей истории: {p['history']}.
    
    КОНТЕКСТ:
    {' '.join(state['context'])}
    
    ОТВЕТЬ {lang_inst.get(state['language'], 'ru')}
    """
    
    response = await llm.ainvoke(prompt)
    return {"answer": emergency_note + response.content}

# Сборка графа
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("analyze")
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph_app = workflow.compile()

# --- 5. ТЕЛЕГРАМ ЛОГИКА ---
router = Router()

@router.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Русский 🇷🇺", callback_data="lang_ru")],
        [InlineKeyboardButton(text="English 🇺🇸", callback_data="lang_en")],
        [InlineKeyboardButton(text="עברית 🇮🇱", callback_data="lang_he")]
    ])
    await message.answer("Выберите язык / Select language:", reply_markup=kb)

@router.callback_query(F.data.startswith("lang_"))
async def set_lang(callback: types.CallbackQuery, state: FSMContext):
    lang = callback.data.split("_")[1]
    await state.update_data(language=lang)
    await callback.message.answer("Опишите пациента (пол, возраст, симптомы/анамнез):")
    await callback.answer()

@router.message()
async def handle_question(message: types.Message, state: FSMContext):
    u_data = await state.get_data()
    lang = u_data.get("language", "ru")
    wait_msg = await message.answer("🔍 Анализирую данные...")
    
    try:
        # Вызываем граф. Каждый вызов — новое состояние.
        result = await graph_app.ainvoke({"input": message.text, "language": lang})
        answer = result["answer"]
        
        # Экранирование спецсимволов для Telegram HTML
        safe_answer = answer.replace('<', '&lt;').replace('>', '&gt;')
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', safe_answer).replace('* ', '• ')
        
        await wait_msg.edit_text(html[:4000], parse_mode="HTML")
    except Exception as e:
        logging.error(f"Error: {e}")
        await wait_msg.edit_text("Произошла ошибка. Пожалуйста, попробуйте переформулировать запрос.")

async def main():
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
