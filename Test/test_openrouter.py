import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()
# Set your API key
# os.environ["OPENROUTER_API_KEY"] = "your_openrouter_api_key_here"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize ChatOpenAI pointing to OpenRouter's base routing server
# Initialize ChatOpenAI pointing to OpenRouter's auto-free router
# fast_llm = ChatOpenAI(
#     openai_api_key=API_KEY,
#     base_url="https://openrouter.ai/api/v1",
#     model="openrouter/free",  # 🚨 This auto-allocates free models dynamically!
#     temperature=0.0,
#     default_headers={
#         "HTTP-Referer": "http://localhost:8501",
#         "X-Title": "Relay AI Production"
#     }
# )


fast_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="deepseek/deepseek-v4-flash:free",
    temperature=0.0,
    default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Relay Fast Router"}
)

# 2. THE HEAVY REASONER (GPT-4o Intelligence Tier for Worker Nodes)
heavy_reasoning_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-120b:free", # Or "meta-llama/llama-3.3-70b-instruct:free"
    temperature=0.2,
    default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Relay Worker Core"}
)

# 3. THE VISION LLM (For Screenshots)
vision_llm = ChatOpenAI(
    openai_api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="google/gemma-4-31b-it:free",
    temperature=0.0,
    default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Relay Vision Engine"}
)

# Test invocation array
messages = [
    SystemMessage(content="You are a helpful IT gatekeeper assistant."),
    HumanMessage(content="Hello! My name is Rathan.")
]

try:
    print("--- 🧠 SENDING REQUEST TO CLOUD OPENROUTER ---")
    response = heavy_reasoning_llm.invoke(messages)
    print("Success! AI output:")
    print(response.content)
except Exception as e:
    print(f"-> ⚠️ LangChain routing failed: {e}")