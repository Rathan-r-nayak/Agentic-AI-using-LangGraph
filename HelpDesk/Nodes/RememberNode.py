from Config.LLMConfig import fast_llm
from Schema.MemoryDecision import MemoryDecision 
from Utils.DatabaseManager import get_user_facts, save_user_fact
from State.HelpDeskState import HelpDeskState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage # 🚨 CRITICAL IMPORT
from Utils.Logger import get_logger
from langgraph.store.base import BaseStore
import uuid


logger = get_logger("REMEMBER")

def remember_node(state: HelpDeskState, config: RunnableConfig, store: BaseStore):
    logger.info("Extracting long term memory")

    user_id = config.get("configurable", {}).get("user_id", "default_user")
    
    namespace = ("user", user_id, "details")
    last_user_message = state["messages"][-1].content

    logger.info(f"Analyzing last message from user '{user_id}': '{last_user_message}'")

    user_memory = store.search(namespace)

    if user_memory:
        user_memory_text = "\n".join(data.value["data"] for data in user_memory)
        logger.info(f"Existing memories found:\n{user_memory_text}")
    else:
        user_memory_text = "No previous memory."
        logger.info("No existing memories found.")

    logger.debug(f"Detected Input: {last_user_message}")


    structured_llm = fast_llm.with_structured_output(MemoryDecision)

    MEMORY_PROMPT = """
    You are an AI assistant's memory manager. Analyze the user's message and determine if 
    there are any new, persistent facts that should be remembered for future conversations.
    Do not store temporary conversational filler.
    
    Existing Knowledge:
    {user_details_content}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_PROMPT),
        ("human", "New Message: {last_message}")
    ])

    chain = prompt | structured_llm
    decision = chain.invoke({
        "user_details_content": user_memory_text,
        "last_message": last_user_message
    })

    if decision.should_write and decision.memories:
        logger.info(f"AI decided to store {len(decision.memories)} NEW memories.")
        for memory in decision.memories:
            logger.info(f"SAVING FACT TO DB: {memory}")
            # Insert the new fact into the PostgresStore
            store.put(
                namespace=namespace, 
                key=str(uuid.uuid4()), 
                value={"data": memory}
            )
    else:
        logger.info("No new memory-worthy information extracted.")

    logger.info("--- MEMORY EXTRACTION COMPLETED ---")
    return state