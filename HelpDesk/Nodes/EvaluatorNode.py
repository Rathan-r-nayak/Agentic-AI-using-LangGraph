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
    
    # CRITICAL SHORT-CIRCUIT: Don't call the LLM if there are no docs!
    if not docs:
        logger.info("No documents provided to evaluate. Marking as insufficient.")
        return {"is_sufficient": False}
    
    # Safely extract text
    raw_doc_text = "\n\n".join([
        d.page_content if hasattr(d, 'page_content') else d.get('content', str(d)) 
        for d in docs
    ])

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
        result = chain.invoke({
            "question": raw_question, 
            "doc_text": raw_doc_text
        })
        
        logger.info(f"Evaluation Complete. Docs Sufficient? {result.is_sufficient}")
        return {"is_sufficient": result.is_sufficient}
        
    except Exception as e:
        logger.error(f"LLM Error during evaluation: {e}")
        # FALLBACK: Assume sufficient so graph continues without crashing
        return {"is_sufficient": True}