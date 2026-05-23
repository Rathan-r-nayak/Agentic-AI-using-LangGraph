from Config.LLMConfig import primary_llm
from Schema.ResolutionPlan import ResolutionPlan
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Helpers import fetch_user_ltm, format_chat_history
from Utils.Logger import get_logger
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore


logger = get_logger("ORCHESTRATOR")

def orchestrator_node(state: HelpDeskState, config: RunnableConfig, store: BaseStore):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR ---")
    logger.info("Drafting resolution plan with memory context")
    
    docs = state.get("documents", [])
    question = state.get("question", "")
    # ltm_facts = state.get("long_term_facts", "No known facts.")
    ltm_facts = fetch_user_ltm(config, store)
    stm_history = format_chat_history(state.get("messages", []))
    
    # Safely format docs in case metadata is missing
    doc_text = "\n\n".join([f"Source: {d.get('metadata', 'Unknown')}\nContent: {d.get('content', '')}" for d in docs])

    structured_llm = primary_llm.with_structured_output(ResolutionPlan)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the L3 Lead IT Service Manager.
        Create a 'Resolution Plan' with parallel tasks for your workers.
        
        CRITICAL CONTEXT:
        - User Facts (Long-Term): {ltm_facts}
        - Recent Chat (Short-Term): {stm_history}
        
        Instructions:
        1. Tailor the tasks to the User's OS, Environment, and Role found in the 'Facts'.
        2. If the 'Recent Chat' shows they already tried a step, instruct the workers NOT to suggest it again.
        3. Mandatory Tasks: 'root_cause', 'resolution_steps', 'preventive_advice'.
        4. Strictly use IT and engineering terminology. Avoid all medical analogies.
        """),
        ("human", "Current Issue: {question}\n\nRetrieved Manuals:\n{doc_text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        plan = chain.invoke({
            "ltm_facts": ltm_facts,
            "stm_history": stm_history,
            "question": question, 
            "doc_text": doc_text
        })
        
        return {"tasks": [task.model_dump() for task in plan.tasks]}
        
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        # FALLBACK: Provide default tasks to prevent the UI from crashing
        return {"tasks": [
            {"title": "Root Cause Analysis", "objective": "Analyze the root cause", "technical_requirements": []}, 
            {"title": "Resolution Steps", "objective": "Provide step-by-step resolution", "technical_requirements": []}, 
            {"title": "Preventive Advice", "objective": "Offer preventive advice", "technical_requirements": []}
        ]}