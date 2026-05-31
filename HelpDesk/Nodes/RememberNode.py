import uuid
from typing import Dict, Any
from Config.LLMConfig import fast_llm
from Schema.MemoryDecision import MemoryDecision 
from State.HelpDeskState import HelpDeskState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger
from Utils.Helpers import fetch_user_ltm  # Reusing your utility!
from langgraph.store.base import BaseStore

logger = get_logger("REMEMBER")

def remember_node(state: HelpDeskState, config: RunnableConfig, store: BaseStore) -> Dict[str, Any]:
    logger.info("--- 🧠 RUNNING MEMORY EXTRACTION ---")

    # 1. Get the actual list of chat messages for memory extraction
    chat_history = state.get("messages", [])
    
    # 2. Get the final AI response just for your console logging
    final_generation = state.get("generation", "NO MESSAGE PREVIEW")

    # --- BULLETPROOF HUMAN MESSAGE FILTER ---
    human_messages = []
    for msg in chat_history: # 👈 Loop over the actual list, not the text string!
        if hasattr(msg, "type") and msg.type == "human":
            human_messages.append(msg.content)
        elif isinstance(msg, dict) and msg.get("role") == "user":
            human_messages.append(msg.get("content", ""))

    if not human_messages:
        logger.warning("No human messages found to analyze.")
        logger.info(f"🤖 AI RES : {final_generation}")
        print("=" * 100)
        return {}

    last_user_message = human_messages[-1].strip()
    
    if not last_user_message:
        logger.info(f"🤖 AI RES : {final_generation}")
        print("=" * 100)
        return {}

    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("user", user_id, "details")

    logger.info(f"Analyzing message from '{user_id}': '{last_user_message}'")

    # 1. Use the shared utility to fetch existing LTM cleanly
    existing_facts = fetch_user_ltm(config, store)

    structured_llm = fast_llm.with_structured_output(MemoryDecision)

    MEMORY_SYSTEM_PROMPT = """You are the Long-Term Memory manager for an AI assistant. 
    Your job is to extract ANY persistent facts the user shares about themselves.
    
    Examples of facts you must ALWAYS extract:
    - Their name (e.g., "I am John" -> "User's name is John")
    - Their job title, department, or role
    - Their computer OS, hardware, or software setup
    - Their working preferences
    
    EXISTING FACTS WE ALREADY KNOW:
    {existing_facts}
    
    CRITICAL DEDUPLICATION RULES:
    1. Compare the MEANING of the user's message to the EXISTING FACTS.
    2. If the user repeats a fact we already know, YOU MUST IGNORE IT.
    3. If it is brand-new information, extract it clearly.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SYSTEM_PROMPT),
        ("human", "New Message: {last_message}")
    ])

    try:
        chain = prompt | structured_llm
        decision: MemoryDecision = chain.invoke({
            "existing_facts": existing_facts,
            "last_message": last_user_message
        })

        if decision.should_write and decision.memories:
            logger.info(f"AI decided to store {len(decision.memories)} NEW memories.")
            for memory in decision.memories:
                logger.info(f"SAVING FACT TO DB: {memory}")
                store.put(
                    namespace=namespace, 
                    key=str(uuid.uuid4()), 
                    value={"data": memory}
                )
        else:
            logger.info("No new memory-worthy information extracted.")
            
    except Exception as e:
        logger.error(f"Memory extraction LLM failed: {e}")

    logger.info("--- MEMORY EXTRACTION COMPLETED ---")

    logger.info(f"🤖 AI RES : {messages}")
    print("=" * 100)
    # Return empty dict since we wrote to the BaseStore, not the State Channel
    return {}