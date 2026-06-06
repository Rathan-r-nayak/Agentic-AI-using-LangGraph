import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Using the active auto-balancing free router to prevent 404 errors completely
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="openrouter/free",  # 👈 This replaces the deepseek-r1:free string
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Helpdesk Testing Script"
    }
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Provide concise analytical answers."),
    ("human", "{user_input}")
])

chain = prompt | llm

print("📡 Querying OpenRouter Free Endpoint...")
try:
    response = chain.invoke({"user_input": "Explain the optimal way to clear database locks in PostgreSQL"})
    print("\n==============================================")
    print(f"🤖 AI Response:\n{response.content}")
    print("==============================================")
except Exception as e:
    print(f"Connection failed: {e}")