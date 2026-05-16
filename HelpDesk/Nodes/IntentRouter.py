from Config.LLMConfig import fast_llm
from Schema.IntentClassification import IntentClassification
from State.HelpDeskState import HelpDeskState
from Utils.Logger import get_logger

logger = get_logger("INTENT_ROUTER")

def intent_router_node(state: HelpDeskState):
    """
    Entry point for Helpdesk AI. 
    Determines if the flow goes to the Greeting response or the Technical pipeline.
    """
    logger.info("Classifying user intent")
    
    # 1. Extract context from the State
    question = state.get("question", "")
    long_term_facts = state.get("long_term_facts", "No known facts about user.")
    logger.info(f"Long-term facts: {long_term_facts}")
    
    # 2. Format Recent Chat History (Last 4-5 messages)
    msgs = state.get("messages", [])
    logger.info(f"Messages: {msgs}")
    
    # We slice [-5:-1] to get recent history, excluding the *current* question 
    # (assuming the current question is appended last, depending on your graph setup).
    if len(msgs) > 0:
        recent_history = "\n".join([f"{m.type.capitalize()}: {m.content}" for m in msgs[-5:]])
    else:
        recent_history = "No previous conversation."

    # 3. Setup structured output
    structured_llm = fast_llm.with_structured_output(IntentClassification)
    
    # 4. Inject Memory into the System Prompt
    system_prompt = f"""You are the entry router for an IT Helpdesk AI. 
    Your job is to classify the user's newest message. Use the provided context 
    to understand short or vague follow-up messages.
    
    --- USER PROFILE (Long-Term Memory) ---
    {long_term_facts}
    
    --- RECENT CONVERSATION ---
    {recent_history}
    
    --- CLASSIFICATION RULES ---
    - 'greeting': If the user is initiating casual conversation (Hi, Hello), saying Thanks, or closing the chat.
    - 'technical_query': If the user describes a system issue, error code, OR if their message is a continuation of ongoing troubleshooting (e.g., "it didn't work", "yes I tried that", "what's the next step?").
    """
    
    # 5. Invoke the LLM
    try:
        result = structured_llm.invoke([
            ("system", system_prompt),
            ("human", f"User's newest message: {question}")
        ])
        
        logger.info(f"Intent Detected: [{result.intent.upper()}]")
        return {"intent": result.intent}
        
    except Exception as e:
        logger.error(f"Router Error: {e}")
        # Fallback to technical query to ensure they get help if the router fails
        return {"intent": "technical_query"}