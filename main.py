import os, asyncio, logging
from aiogram import Bot, Dispatcher, types, F
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Настройки
DB_PATH = "faiss_index"
PDF_FILE = "Prevention.pdf"
TOKEN = os.getenv("TELEGRAM_TOKEN")
API_KEY = os.getenv("GOOGLE_API_KEY")

bot = Bot(token=TOKEN)
dp = Dispatcher()
qa_chain = None

async def setup_rag():
    global qa_chain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    
    if os.path.exists(DB_PATH):
        logger.info(">>> Загрузка базы с диска (экономим лимиты)...")
        vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info(">>> Индексация PDF (первый запуск)...")
        loader = PyPDFLoader(PDF_FILE)
        pages = loader.load_and_split()
        vector_store = FAISS.from_documents(pages, embeddings)
        vector_store.save_local(DB_PATH)
        logger.info(">>> База сохранена на диск!")

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

@dp.message(F.text == "/start")
async def start_handler(message: types.Message):
    await message.answer("Медицинский бот готов. Спрашивайте по протоколу.")

@dp.message(F.text)
async def handle_question(message: types.Message):
    if not qa_chain:
        await message.answer("Система еще загружается...")
        return
    
    logger.info(f"Вопрос: {message.text}")
    # Асинхронный вызов, чтобы бот не «тупил»
    response = await qa_chain.ainvoke({"query": message.text})
    await message.answer(response["result"])

async def main():
    await setup_rag()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

