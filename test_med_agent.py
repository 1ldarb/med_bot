import os, asyncio
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv('/home/opc/med_bot/.env')
API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PATH = "/home/opc/med_bot/vectorstore/db_faiss"

# Инициализация тех же компонентов, что в боте
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY)

# Функция-имитатор работы бота
def get_bot_response(query):
    qa_chain = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", 
        retriever=vector_db.as_retriever(search_kwargs={"k": 10})
    )
    return qa_chain.invoke({"query": query})["result"]

# Промпт для Агента-Критика
CRITIC_PROMPT = """
Ты — строгий медицинский аудитор. Твоя задача — проверить ответ медицинского бота на ошибки.
Ответ бота должен основываться на протоколах профилактики в Израиле.

КРИТЕРИИ ОШИБКИ:
1. Бот пропустил важный скрининг (например, маммографию после 50 лет или колоноскопию).
2. Бот не упомянул вакцинацию.
3. Бот дал опасный совет, не предусмотренный протоколом.
4. Ответ не структурирован (нет разделов 1 и 2).

КЕЙС ПАЦИЕНТА: {case}
ОТВЕТ БОТА: {bot_answer}

Вынеси вердикт: [ПРОЙДЕНО] или [ОШИБКА]. Если ошибка — кратко объясни почему.
"""

async def run_test():
    # Список тестовых сценариев
    test_cases = [
        "Женщина, 51 год, без жалоб, курит.",
        "Мужчина, 65 лет, гипертония, никогда не делал скрининги.",
        "Девушка, 25 лет, хочет узнать про вакцинацию от ВПЧ (HPV)."
    ]

    print("\n🚀 Запуск Агента-Тестировщика...\n" + "="*50)

    for case in test_cases:
        print(f"📝 Тестируем кейс: {case}")
        
        # 1. Получаем ответ от "Медбота"
        bot_answer = get_bot_response(case)
        
        # 2. Агент-Критик анализирует ответ
        critic_res = await model.ainvoke(CRITIC_PROMPT.format(case=case, bot_answer=bot_answer))
        
        print(f"🤖 Ответ бота: {bot_answer[:150]}...")
        print(f"⚖️ Вердикт Критика: {critic_res.content}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_test())
