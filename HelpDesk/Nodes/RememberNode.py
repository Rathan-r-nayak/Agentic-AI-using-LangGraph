from Config.LLMConfig import fast_llm
from Schema.MemoryDecision import MemoryDecision 
from Utils.DatabaseManager import get_user_facts, save_user_fact
from State.HelpDeskState import HelpDeskState
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage # 🚨 CRITICAL IMPORT
from Utils.Logger import get_logger

logger = get_logger("REMEMBER")

def remember_node(state: HelpDeskState, config: RunnableConfig):
    logger.info("Extracting long term memory")
    
    # 1. Get messages and the raw question
    messages = state.get("messages", [])
    question = state.get("question", "") # 🚨 BACKUP: Use the 'question' key if messages are weird
    
    last_user_message = ""

    # Try to find the message in the message history first
    for m in reversed(messages):
        if isinstance(m, HumanMessage) or getattr(m, 'type', '') == 'human':
            last_user_message = m.content
            break
    
    # If messages list was empty or didn't have a human msg, use the 'question' state variable
    if not last_user_message and question:
        last_user_message = question

    if not last_user_message: 
        logger.warning("No human message found in state or question.")
        return {}

    # 2. Extract User ID and Current Facts
    user_id = config["configurable"].get("thread_id", "default_user")
    user_memory_text = get_user_facts(user_id)

    structured_llm = fast_llm.with_structured_output(MemoryDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract permanent facts (Name, Role, OS, Environment) from the user message. 
         Current Knowledge: {user_details_content}"""),
        ("human", "{last_message}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        decision = chain.invoke({
            "user_details_content": user_memory_text,
            "last_message": last_user_message
        })
        
        logger.debug(f"Detected Input: {last_user_message}")
        
        if decision.memories:
            for memory in decision.memories:
                logger.info(f"DATABASE CALL: save_user_fact('{user_id}', '{memory}')")
                save_user_fact(user_id, memory)
                logger.info("SUCCESS")
        else:
            logger.info("LLM extracted ZERO memories.")
            
    except Exception as e:
        logger.error(f"Memory extraction failed: {e}")

    return {}