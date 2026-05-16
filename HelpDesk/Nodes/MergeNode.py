from Config.LLMConfig import fast_llm
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("MERGE")

def merge_node(state: HelpDeskState):
    logger.info("Merging worker results")
    
    # Grab the list of generated sections from the parallel workers
    worker_results = state.get("worker_results", [])
    
    if not worker_results:
        logger.error("No worker results found.")
        return {"generation": "I am currently unable to generate a complete resolution plan. Please escalate this ticket."}
        
    # Join the raw sections together
    combined_draft = "\n\n".join(worker_results)
    
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
    
    # Fast LLM is perfect for simple editing and formatting
    chain = prompt | fast_llm
    response = chain.invoke({"draft": combined_draft})
    
    logger.info("Sections successfully merged and polished.")
    
    # Save the polished version to the 'generation' state
    return {"generation": response.content}