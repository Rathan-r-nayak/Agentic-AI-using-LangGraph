import os
from dotenv import load_dotenv
# 👇 Use the official, built-in ChatOpenAI class instead
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load environment variables from your local .env file
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ ERROR: OPENROUTER_API_KEY not found in environment or .env file.")
    exit(1)
else:
    print(f"🔒 API Key detected safely (Prefix: {api_key[:7]}...)")


# 2. Initialize using ChatOpenAI pointed at OpenRouter's URL
# This bypasses wrapper bugs and passes headers natively
llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openrouter/free",  # Auto-routes to the fastest free model available
    temperature=0.3,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Smart Helpdesk Triage App"
    }
)

# 3. Create a quick test chain to ensure parsing is working fine
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful IT support operations validator. Respond concisely."),
    ("human", "{user_input}")
])

chain = prompt | llm

# 4. Invoke the model and verify output
print("\n📡 Pinging OpenRouter API...")
try:
    response = chain.invoke({"user_input": "Run an eco check on connection status and reply with 'API ONLINE'"})
    
    print("\n==============================================")
    print("🎉 SUCCESS! OpenRouter & LangChain are working fine!")
    print(f"🤖 AI Response:\n{response.content}")
    print("==============================================")

except Exception as e:
    print(f"\n❌ Connection Failed: {e}")