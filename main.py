import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher()

# Мощные мультиязычные эмбеддинги
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Загрузка и индексация (RAG)
loader = PyPDFLoader("Prevention.pdf")
pages = loader.load_and_split()
vector_store = FAISS.from_documents(pages, embeddings)

# Создаем мультиязычный промпт
template = """
You are a professional medical assistant specialized in Israeli clinical protocols.
Use the following context to answer the user's question accurately.
Respond in the SAME LANGUAGE as the user's question (Russian, Hebrew, or English).
If the answer is not in the context, say you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": QA_PROMPT}
)

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    # Приветствие на трех языках
    text = (
        "🇷🇺 Здравствуйте! Я ИИ-ассистент по медпротоколам.\n"
        "🇺🇸 Hello! I am your Medical Protocol AI assistant.\n"
        "🇮🇱 שלום! אני עוזר בינה מלאכותית לפרוטוקולים רפואיים."
    )
    await message.answer(text)

@dp.message()
async def query_handler(message: types.Message):
    response = qa_chain.invoke(message.text)
    await message.answer(response["result"])

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
