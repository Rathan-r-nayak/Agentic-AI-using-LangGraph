from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage
from Config.LLMConfig import fast_llm
from HelpDesk.Schema.GateKeeperDecision import GatekeeperDecision
from State.HelpDeskState import HelpDeskState
from Utils.DatabaseManager import get_user_facts
from Utils.Helpers import format_chat_history
import logging

logger = logging.getLogger(__name__)

# Import your new schema (adjust path as needed)

def gatekeeper_node(state: HelpDeskState, config: RunnableConfig):
    """
    Routes user intent between the technical RAG pipeline or casual banter.
    Uses a structured Pydantic schema to enforce strict boolean routing flags.
    """
    
    logger.info("--- 🛡️ RUNNING INTENT ROUTER & GATEKEEPER CHECK ---")
    question = state.get("question", "")
    logger.info(f"Question: {question}")
    
    # Fetch LONG-TERM memory
    user_id = config["configurable"].get("thread_id", "default_user")
    long_term_facts = get_user_facts(user_id)
    long_term_facts = long_term_facts if long_term_facts else 'No known facts about user.'
    
    # fetch past chats
    messages = state.get("messages", [])
    stm_history = format_chat_history(messages)

    logger.info(f"Facts: {long_term_facts}")
    
    # Bind the LLM to the Pydantic Schema
    structured_llm = fast_llm.with_structured_output(GatekeeperDecision)
    
    # Cleaned-up System Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the specialized Gatekeeper and Greeting Agent for an enterprise IT Helpdesk.
        
        Evaluate the user's input and fill out the schema precisely.
        - If it's a technical IT issue (VPN, crashes, databases, code), set 'is_technical_it_query' to True and leave 'message_content' blank.
        - If it's a greeting, casual check-in, identity question ("Who am I?"), or completely OUTSIDE the context of IT support (jokes, weather), set 'is_technical_it_query' to False and write a warm, conversational response in 'message_content'.
        
        Long-Term Facts about this user:
        {long_term_facts}
        
        Recent Chat History: 
        {stm_history}
        """),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm  
    
    # The response is now a strongly-typed GatekeeperDecision object
    decision: GatekeeperDecision = chain.invoke({
        "long_term_facts": long_term_facts, 
        "question": question,
        "stm_history": stm_history
    })
    
    logger.info(f"Gatekeeper Decision: RAG Required = {decision.is_technical_it_query}")

    # Clean Routing Logic
    if decision.is_technical_it_query:
        return {
            "requires_rag": True,
        }
    else:
        # It's a greeting/chatter. Save the AI's response to the chat history.
        new_message = AIMessage(content=decision.message_content)
        return {
            "requires_rag": False,
            "messages": [new_message] 
        }