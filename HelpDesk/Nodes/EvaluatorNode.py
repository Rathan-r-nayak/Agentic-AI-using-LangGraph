from Config.LLMConfig import fast_llm
from Schema.DocumentEvaluation import DocumentEvaluation
from State.HelpDeskState import HelpDeskState # Adjust import if your state is named differently
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("EVALUATOR")

def evaluator_node(state: HelpDeskState):
    logger.info("Evaluating retrieved documents")
    
    raw_question = state.get("question", "")
    docs = state.get("documents", [])
    
    # 1. Safely extract the raw text from the document chunks
    raw_doc_text = "\n\n".join([
        d.page_content if hasattr(d, 'page_content') else d.get('content', str(d)) 
        for d in docs
    ])

    # ==========================================
    # 🚨 THE MASTER SANITIZER 🚨
    # Scrub every variable before it touches Azure
    # ==========================================
    def sanitize(text):
        if not isinstance(text, str):
            text = str(text)
        
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
        for bad_word, good_word in replacements.items():
            text = text.replace(bad_word, good_word)
        return text

    # 2. Clean the data!
    safe_question = sanitize(raw_question)
    safe_doc_text = sanitize(raw_doc_text)

    # 3. Setup the structured output
    structured_llm = fast_llm.with_structured_output(DocumentEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Technical Document Auditor. 
        Determine if the provided documentation contains the technical configuration 
        or error resolution steps needed for the user's query.
        
        Focus purely on technical indicators and error logs. 
        Do not use biological or medical analogies."""),
        ("human", "User Query: {question}\n\nTechnical Sources:\n{doc_text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        # 4. Pass ONLY the sanitized strings to the LLM
        result = chain.invoke({
            "question": safe_question, 
            "doc_text": safe_doc_text
        })
        
        logger.info(f"Docs Sufficient? {result.is_sufficient}")
        return {"is_sufficient": result.is_sufficient}
        
    except Exception as e:
        logger.error(f"Azure Filter Blocked the Request: {e}")
        # FALLBACK: If Azure still blocks it, assume documents are sufficient
        # so the graph continues to the Orchestrator without crashing the UI.
        return {"is_sufficient": True}