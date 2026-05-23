from typing import Dict, Any
from Config.LLMConfig import primary_llm
from Schema.CritiqueReview import CritiqueReview
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("CRITIQUE")

def critique_node(state: HelpDeskState) -> Dict[str, Any]:
    logger.info("--- 🛡️ RUNNING CRITIQUE & SAFETY CHECK ---")
    
    generation = state.get("generation", "")
    if not generation:
        logger.warning("No generation text found to critique.")
        return {}
    
    structured_llm = primary_llm.with_structured_output(CritiqueReview)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Compliance & Safety Officer for Helpdesk AI.
        Review the proposed IT resolution text.
        
        RULES:
        1. No Passwords: If the text tells the user a raw password, remove it and tell them to 'use their secure credentials'.
        2. Tone: Ensure the tone is highly professional and empathetic. No condescending language.
        3. If you find a violation, set is_safe to False and rewrite the text in 'scrubbed_text'.
        4. If it is safe, set is_safe to True and return the exact same text in 'scrubbed_text'.
        """),
        ("human", "Proposed Resolution:\n{generation}")
    ])
    
    try:
        chain = prompt | structured_llm
        review: CritiqueReview = chain.invoke({"generation": generation})
        
        if review.is_safe:
            logger.info("Status: APPROVED (Safe & Professional)")
        else:
            logger.warning("Status: VIOLATION DETECTED. Scrubbing text.")
            logger.warning(f"Reasoning: {review.reasoning}")
            
        return {
            "generation": review.scrubbed_text, 
            "is_private": review.is_safe 
        }
        
    except Exception as e:
        logger.error(f"Critique LLM API failed: {e}")
        # Fallback: Assume safe and return original generation if check fails
        return {"generation": generation, "is_private": True}