import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def create_vector_db():
    print("⏳ Начинаю процесс индексации PDF...")
    
    # 1. Загрузка PDF (Prevention.pdf должен быть в той же папке)
    if not os.path.exists("Prevention.pdf"):
        print("❌ Ошибка: Файл Prevention.pdf не найден!")
        return

    loader = PyPDFLoader("Prevention.pdf")
    documents = loader.load()

    # 2. Нарезка текста на куски
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    # 3. Инициализация эмбеддингов Google
# 3. Инициализация эмбеддингов Google
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # 4. Создание векторной базы FAISS
    db = FAISS.from_documents(texts, embeddings)
    
    # 5. Сохранение локально
    os.makedirs("vectorstore/db_faiss", exist_ok=True)
    db.save_local("vectorstore/db_faiss")
    print("✅ База данных успешно создана и сохранена в vectorstore/db_faiss")

if __name__ == "__main__":
    create_vector_db()
