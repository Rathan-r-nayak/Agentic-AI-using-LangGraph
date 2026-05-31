from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

from Config.LLMConfig import reasoning_llm
from Schema.GateKeeperDecision import GatekeeperDecision
from State.HelpDeskState import HelpDeskState
from Utils.Helpers import fetch_user_ltm, format_chat_history
from Utils.Logger import get_logger

logger = get_logger("GATEKEEPER")

def gatekeeper_node(state: HelpDeskState, config: RunnableConfig, store: BaseStore) -> Dict[str, Any]:
    """
    Routes user intent between the technical Map-Reduce pipeline or casual banter.
    Uses a structured Pydantic schema to enforce strict boolean routing flags.
    """
    question = state.get("question", "")
    print("=" * 100)
    logger.info(f"🗣️ USER REQ : {question}")
    
    logger.info("--- 🛡️ RUNNING INTENT ROUTER & GATEKEEPER CHECK ---")
    
    if not question:
        logger.warning("Empty question received in state.")
        return {"requires_rag": False, "generation": "How can I help you today?"}

    logger.info(f"User Query: '{question}'")

    # Fetch context
    long_term_facts = fetch_user_ltm(config, store)
    messages = state.get("messages", [])
    stm_history = format_chat_history(messages)

    # Log context summaries rather than dumping huge strings to the terminal
    logger.info(f"Loaded Context -> LTM Facts: {len(long_term_facts.splitlines())} items | STM History: {len(messages)} recent messages")

    # Bind the LLM to the Pydantic Schema
    structured_llm = reasoning_llm.with_structured_output(GatekeeperDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the specialized Gatekeeper and Greeting Agent for an enterprise IT Helpdesk.
        
        Evaluate the user's input and fill out the schema precisely.
        
        CRITICAL ROUTING RULES:
        1. ACTION REQUIRED -> Set 'is_technical_it_query' to True if the user:
           - Asks a technical IT question (VPN, crashes, code, hardware).
           - Asks to check the status or details of a support ticket (e.g., "Ticket ID 3").
           - Asks to create, update, or escalate a ticket.
           - Needs you to look up historical resolutions or database records.
           *(Leave 'message_content' blank if True)*
           
        2. CASUAL/GREETING -> Set 'is_technical_it_query' to False if the user:
           - Is just saying Hello, Thank You, or Goodbye.
           - Is asking a purely identity-based question ("Who am I?").
           - Is asking something completely outside of IT support (jokes, weather).
           *(Write a warm, conversational response in 'message_content' if False)*
        
        IDENTITY OVERRIDE: If the user asks for their name or details about themselves, you MUST look at the Long-Term Facts and Recent Chat History to answer them directly. Do NOT use generic corporate greetings.
        
        Long-Term Facts about this user:
        {long_term_facts}
        
        Recent Chat History: 
        {stm_history}
        """),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm  
    
    try:
        # The response is a strongly-typed GatekeeperDecision object
        decision: GatekeeperDecision = chain.invoke({
            "long_term_facts": long_term_facts, 
            "question": question,
            "stm_history": stm_history
        })
        
        logger.info(f"Decision Result: Action Required = {decision.is_technical_it_query}")

        # Clean Routing Logic
        if decision.is_technical_it_query:
            return {
                "requires_rag": True, # This flag triggers your conditional edge to the Orchestrator
            }
        else:
            logger.info("Routing to conversational response.")
            new_message = AIMessage(content=decision.message_content)
            return {
                "requires_rag": False,
                "messages": [new_message],
                "generation": decision.message_content
            }
            
    except Exception as e:
        # Failsafe: If the LLM throws a 429/500 error or fails schema validation, 
        # default to RAG/Tools so the user's technical issue isn't ignored.
        logger.error(f"Gatekeeper LLM extraction failed: {e}")
        logger.warning("Failsafe triggered: Defaulting to Action pipeline.")
        return {
            "requires_rag": True
        }