from typing import Dict, Any
from Config.LLMConfig import fast_llm
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("MERGE")

def merge_node(state: HelpDeskState) -> Dict[str, Any]:
    logger.info("--- 🧩 RUNNING MERGE NODE ---")
    
    worker_results = state.get("worker_results", [])
    
    if not worker_results:
        logger.warning("No worker results found to merge.")
        return {"generation": "I am currently unable to generate a complete resolution plan. Please escalate this ticket."}
        
    combined_draft = "\n\n".join(worker_results)
    logger.info(f"Merging {len(worker_results)} worker sections.")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Technical Editor for Helpdesk AI.
        Your junior workers have generated separate sections of an IT incident report.
        
        TASK:
        1. Read the combined draft below.
        2. Fix any awkward transitions between the sections.
        3. Ensure the formatting (Markdown, bolding, code blocks) is consistent.
        4. Do NOT remove any technical facts, error codes, or steps. Just polish the flow.
        """),
        ("human", "Draft Report:\n{draft}")
    ])
    
    try:
        chain = prompt | fast_llm
        response = chain.invoke({"draft": combined_draft})
        logger.info("Sections successfully merged and polished.")
        return {"generation": response.content}
        
    except Exception as e:
        logger.error(f"Merge LLM failed: {e}")
        # Fallback: Just return the raw concatenated draft so the user still gets help
        return {"generation": combined_draft}