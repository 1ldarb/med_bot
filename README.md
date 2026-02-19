# Medical Prevention AI Agent (Israel) 🏥🤖

Advanced Agentic RAG system built with **LangGraph** and **Aiogram 3**. The bot provides personalized medical prevention plans (screenings and vaccinations) based on Israeli medical standards (Tabenkin guidelines).

## 🌟 Key Features

### 1. Multi-Step Agentic Workflow (LangGraph)
Unlike simple chatbots, this system uses a directed cyclic graph (state machine) to process queries:
* **Analyze Node:** Extracts structured patient data (age, gender, history) using Gemini's structured output. It also includes an **Emergency Detector** for "red flags."
* **Retrieve Node:** Performs semantic search in a vector database (FAISS) using age-and-gender-specific queries.
* **Generate Node:** Synthesizes a final medical response with strict filtering rules to prevent hallucinations.

### 2. RAG (Retrieval-Augmented Generation)
* **Vector Store:** Uses FAISS for local storage of medical guidelines.
* **Embeddings:** Google Generative AI Embeddings.
* **Context Filtering:** Hard-coded logic ensures a male patient won't receive recommendations for a mammogram, and a 25-year-old won't be told to do a colonoscopy.

### 3. Multi-Language Support
Full support for **Russian, Hebrew, and English**, including language-specific medical terminology and UI.

## 🚀 Tech Stack
* **Framework:** Aiogram 3.x (Async)
* **Orchestration:** LangGraph (StateGraph)
* **LLM:** Gemini 2.0 Flash (via LangChain)
* **Vector DB:** FAISS
* **Data Validation:** Pydantic v2
* **Deployment:** Docker & Hetzner Cloud

## 🛠 Project Structure
* `main.py` — Entry point, LangGraph definitions, and Telegram handlers.
* `ingest.py` — Script for processing PDF guidelines and creating the vector store.
* `vectorstore/` — Pre-computed FAISS index.
* `Dockerfile` — Ready-to-use containerization.
