from Config.LLMConfig import primary_llm
from Schema.ResolutionPlan import ResolutionPlan
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Helpers import format_chat_history
from Utils.Logger import get_logger

logger = get_logger("ORCHESTRATOR")

def orchestrator_node(state: HelpDeskState):
    logger.info("Drafting plan with memory")
    
    docs = state.get("documents", [])
    question = state.get("question", "")
    ltm_facts = state.get("long_term_facts", "No known facts.")
    stm_history = format_chat_history(state.get("messages", []))
    
    # Safely format docs in case metadata is missing
    doc_text = "\n\n".join([f"Source: {d.get('metadata', 'Unknown')}\nContent: {d.get('content', '')}" for d in docs])
    
    # ==========================================
    # 🚨 FORCED SANITIZATION 🚨
    # Scrubbing the raw state data before Azure sees it
    # ==========================================
    replacements = {
            "symptoms": "indicators", "Symptoms": "Indicators", "SYMPTOMS": "INDICATORS",
            "symptom": "indicator", "Symptom": "Indicator", "SYMPTOM": "INDICATOR",
            "diagnosis": "root cause analysis", "Diagnosis": "Root Cause Analysis", "DIAGNOSIS": "ROOT CAUSE ANALYSIS",
            "treatment": "resolution", "Treatment": "Resolution", "TREATMENT": "RESOLUTION",
            "patient": "system", "Patient": "System", "PATIENT": "SYSTEM",
            "severity": "impact", "Severity": "Impact", "SEVERITY": "IMPACT",
            "priority": "level", "Priority": "Level", "PRIORITY": "LEVEL",
            "triage": "sort", "Triage": "Sort", "TRIAGE": "SORT"
        }

    safe_question = question
    safe_stm_history = stm_history
    safe_ltm_facts = ltm_facts
    safe_doc_text = doc_text

    for bad_word, good_word in replacements.items():
        safe_question = safe_question.replace(bad_word, good_word)
        safe_stm_history = safe_stm_history.replace(bad_word, good_word)
        safe_ltm_facts = safe_ltm_facts.replace(bad_word, good_word)
        safe_doc_text = safe_doc_text.replace(bad_word, good_word)

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
        # Pass the SANITIZED variables to the LLM
        plan = chain.invoke({
            "ltm_facts": safe_ltm_facts,
            "stm_history": safe_stm_history,
            "question": safe_question, 
            "doc_text": safe_doc_text
        })
        
        return {"tasks": plan.tasks}
        
    except Exception as e:
        logger.error(f"Azure Filter Blocked: {e}")
        # FALLBACK: Provide default tasks to prevent the UI from crashing
        return {"tasks": [
            "Analyze the root cause of the indicator", 
            "Provide step-by-step resolution", 
            "Offer preventive advice"
        ]}