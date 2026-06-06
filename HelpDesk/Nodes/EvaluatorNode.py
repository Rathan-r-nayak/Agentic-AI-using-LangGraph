from typing import Dict, Any
from Config.LLMConfig import fast_llm
from Schema.DocumentEvaluation import DocumentEvaluation
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("EVALUATOR")

def evaluator_node(state: HelpDeskState) -> Dict[str, Any]:
    logger.info("--- ⚖️ RUNNING DOCUMENT EVALUATOR ---")
    
    raw_question = state.get("question", "")
    docs = state.get("documents", [])
    
    # Safely extract text (or default to empty text if no docs)
    if docs:
        raw_doc_text = "\n\n".join([
            d.page_content if hasattr(d, 'page_content') else d.get('content', str(d)) 
            for d in docs
        ])
    else:
        raw_doc_text = "No internal documents retrieved."

    # Bind to your Pydantic schema
    structured_llm = fast_llm.with_structured_output(DocumentEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Technical Document Auditor for an IT Helpdesk. 
        Determine if the system has enough information to proceed to the next step.
        
        CRITICAL ROUTING RULES:
        1. ACTION & TICKET QUERIES: If the user is asking to look up a specific ticket (e.g., "details of ticket id 3"), create a ticket, or check an employee's history, you MUST set 'is_sufficient' to True. These queries require our internal database tools, NOT web search.
        2. TROUBLESHOOTING QUERIES: If the user is asking a general IT question (e.g., "VPN isn't working"), evaluate the 'Technical Sources' below. 
           - If the sources contain the answer, set 'is_sufficient' to True.
           - If the sources are empty or do NOT answer the question, set 'is_sufficient' to False (this will trigger a fallback web search)."""),
        ("human", "User Query: {question}\n\nTechnical Sources:\n{doc_text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "question": raw_question, 
            "doc_text": raw_doc_text
        })
        
        logger.info(f"Evaluation Complete. Docs Sufficient? {result.is_sufficient}")
        return {"is_sufficient": result.is_sufficient}
        
    except Exception as e:
        logger.error(f"LLM Error during evaluation: {e}")
        # FALLBACK: Assume sufficient so graph continues to orchestrator/tools instead of hallucinating web searches
        return {"is_sufficient": True}