import asyncio
import logging
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()

# FIX 1: Silence the annoying 'additionalProperties' warning
logging.getLogger("langchain_google_vertexai.functions_utils").addFilter(
    lambda record: "'additionalProperties' is not supported in schema" not in record.getMessage()
)
logging.getLogger("langchain_google_genai.chat_models").addFilter(
    lambda record: "'additionalProperties' is not supported in schema" not in record.getMessage()
)

async def main():
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

    client_config = {
        "rag_server": {
            "command": "python", 
            "args": ["rag_on_mcp.py"], 
            "transport": "stdio",
        }
    }

    client = MultiServerMCPClient(client_config)
    tools = await client.get_tools()
    print(f"Successfully connected! Loaded tools: {[t.name for t in tools]}")

    # FIX 3: Give the agent a System Prompt so it knows it can chat normally
    system_prompt = (
        "You are a helpful and intelligent AI assistant. "
        "You have access to tools to ingest and retrieve documents. "
        "you should answer it using the tools and if you dont find the relavent information, then say dont know."
    )
    
    # Pass the prompt into the agent using state_modifier
    agent = create_agent(llm, tools=tools, system_prompt=system_prompt)

    chat_history = []
    
    print("\nAgent is ready! Ask your questions. (Type 'quit' or 'exit' to stop)\n")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Closing connection. Goodbye!")
            break
        
        chat_history.append({"role": "user", "content": user_input})
        print("Agent is thinking...")
        
        response = await agent.ainvoke({"messages": chat_history})

        for msg in response["messages"]:
            if msg.type == "tool":
                print(f"\n[HIDDEN DATA FROM MCP SERVER]\n{msg.content}\n")
        
        # FIX 2: Safely extract text whether it's a normal string or a SynthID signature list
        ai_content = response["messages"][-1].content
        if isinstance(ai_content, list):
            # Loop through the blocks and grab only the text parts, ignoring the signature
            ai_message = "".join([block.get("text", "") for block in ai_content if block.get("type") == "text"])
        else:
            ai_message = ai_content
            
        print(f"\n=== Agent ===\n{ai_message}\n")
        
        chat_history = response["messages"]

if __name__ == "__main__":
    asyncio.run(main())