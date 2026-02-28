import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_vector_db():
    print("⏳ Starting PDF indexing process...")
    
    # 1. Load PDF (Prevention.pdf must be in the same folder)
    if not os.path.exists("Prevention.pdf"):
        print("❌ Error: Prevention.pdf file not found!")
        return

    loader = PyPDFLoader("Prevention.pdf")
    documents = loader.load()

    # 2. Slice text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    texts = text_splitter.split_documents(documents)

    # 3. Initialize Google embeddings
# 3. Initialize Google embeddings
    api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # 4. Create FAISS vector database
    db = FAISS.from_documents(texts, embeddings)
    
    # Extract metadata to flat dicts to avoid pydantic V1/V2 pickle issues
    for doc_id, doc in db.docstore._dict.items():
        doc.metadata = {k: str(v) for k, v in doc.metadata.items()}
    
    # 5. Save locally
    os.makedirs("vectorstore/db_faiss", exist_ok=True)
    db.save_local("vectorstore/db_faiss")
    print("✅ Database successfully created and saved in vectorstore/db_faiss")

if __name__ == "__main__":
    create_vector_db()
