import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

print("📡 Connecting to local native Ollama service...")

# Initialize ChatOpenAI pointing to your local native Ollama instance
# This bypasses OpenRouter completely—no keys or internet required!
llm = ChatOpenAI(
    model="gemma2:2b",
    openai_api_key="ollama", # Placeholder string to pass LangChain initialization checks
    openai_api_base="http://localhost:11434/v1", # Native Ollama local port mapping
    temperature=0.3,
)

# Create a quick validation chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a local model operations validator. Respond concisely."),
    ("human", "{user_input}")
])

chain = prompt | llm

print("💎 Pinging local Gemma model...")
try:
    response = chain.invoke({
        "user_input": "Confirm your model name and reply with 'LOCAL INFRASTRUCTURE OPERATIONAL'"
    })
    
    print("\n==============================================")
    print("🎉 SUCCESS! Local Gemma & LangChain are working fine!")
    print(f"🤖 AI Response:\n{response.content.strip()}")
    print(response)
    print("==============================================")

except Exception as e:
    print(f"\n❌ Local Connection Failed: {e}")
    print("💡 Troubleshooting: Check if the background service is active by running 'systemctl status ollama'")