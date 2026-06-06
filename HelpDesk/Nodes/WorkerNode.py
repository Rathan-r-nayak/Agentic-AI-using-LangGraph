from langchain.messages import ToolMessage

from Config.LLMConfig import primary_llm, openrouter_llm, gemma_llm, reasoning_llm
from langchain_core.prompts import ChatPromptTemplate
from Utils.Helpers import fetch_user_ltm
from Utils.Logger import get_logger
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore

logger = get_logger("WORKER")

async def worker_node(state, config: RunnableConfig, store: BaseStore):
    # ==========================================
    # 1. EXTRACT TASK SAFELY (Handles Strings, Dicts, or Objects)
    # ==========================================
    task = state.get("task", "")
    
    if isinstance(task, str):
        task_title = "Resolution Steps"
        objective = task
        req_str = "Provide clear technical instructions."
    elif isinstance(task, dict):
        task_title = task.get("title", "Task Execution")
        objective = task.get("objective", str(task))
        requirements = task.get("technical_requirements", [])
        req_str = ", ".join(requirements) if isinstance(requirements, list) else str(requirements)
    else:
        # Assuming it's your Pydantic object from the Orchestrator
        task_title = getattr(task, "title", "Task Execution")
        objective = getattr(task, "objective", str(task))
        requirements = getattr(task, "technical_requirements", [])
        req_str = ", ".join(requirements) if isinstance(requirements, list) else str(requirements)

    logger.info(f"Executing: {task_title}")

    mcp_tools = config.get("configurable", {}).get("mcp_tools", [])
    llm_with_tools = reasoning_llm.bind_tools(mcp_tools)
    

    # ==========================================
    # 2. EXTRACT RAW STATE DATA
    # ==========================================
    docs = state.get("documents", [])
    
    raw_doc_text = "\n\n".join([
        f"Source: {d.metadata if hasattr(d, 'metadata') else d.get('metadata', 'Unknown')}\n"
        f"Content: {d.page_content if hasattr(d, 'page_content') else d.get('content', str(d))}" 
        for d in docs
    ])

    raw_question = state.get("question", "")
    # raw_ltm = state.get("long_term_facts", "")
    raw_ltm = fetch_user_ltm(config, store)
    raw_stm = state.get("chat_history", state.get("messages", ""))

    # ==========================================
    # 3. PROMPT & EXECUTE WITH FALLBACK
    # ==========================================

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly specialized IT Support Worker.
        Your job is to execute your assigned task perfectly and present the data in a highly readable, professional format.
        
        USER CONTEXT:
        - Known Facts: {ltm_facts}
        - Recent Conversation: {stm_history}
        
        TASK OBJECTIVE: {objective}
        REQUIREMENTS: {requirements}
        
        CRITICAL RULES:
        1. DYNAMIC FORMATTING (CRITICAL): 
           - IF your task is to retrieve, show, or summarize a ticket: YOU MUST present the data using a clean Markdown Table or a structured Key-Value dashboard view. You MUST prominently display the Ticket ID, Current Status, Category, Location, Description, and Resolution (if any). Do not bury this data in paragraphs.
           - IF your task is diagnostic (RCA, Preventive Advice): Use sharp bullet points and bold headers.
        2. Adapt your instructions to the User's known environment.
        3. Strictly use standard IT infrastructure terminology. Absolutely no biological, medical, or emergency-room analogies.
        4. No conversational filler. Do NOT write an intro (e.g., "Here are the details...") or a conclusion. Output ONLY the requested data payload.
        """),
        ("human", "Issue: {question}\n\nDatabase Data & Manuals:\n{doc_text}")
    ])



    initial_messages = prompt.format_messages(
        ltm_facts=raw_ltm,
        stm_history=raw_stm,
        objective=objective,
        requirements=req_str,
        question=raw_question,
        doc_text=raw_doc_text
    )

    try:
        response = await llm_with_tools.ainvoke(initial_messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"Worker '{task_title}' requested {len(response.tool_calls)} tool calls.")

            tool_map = {t.name: t for t in mcp_tools}

            conversation_history = initial_messages + [response]

            for tc in response.tool_calls:
                tool_name = tc["name"]

                if tool_name in tool_map:
                    
                    tool_result = await tool_map[tool_name].ainvoke(tc["args"])
                    logger.info(f"🛠️ MCP Server Output for '{tool_name}': {tool_result}")

                    conversation_history.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                    )
                else:
                    logger.error(f"❌ Tool execution failed: '{tool_name}' not found in tool_map.")
                    conversation_history.append(
                        ToolMessage(content=f"Error: Tool {tool_name} not found.", tool_call_id=tc["id"])
                    )


            final_response = await llm_with_tools.ainvoke(conversation_history)
            content = final_response.content
        else:
            content = response.content

    except Exception as e:
        logger.error(f"Worker execution failed ({task_title}): {e}")
        content = "*(Content unavailable. Please verify system logs manually based on standard IT protocols.)*"
    
    return {"worker_results": [f"### {task_title}\n{content}"]}