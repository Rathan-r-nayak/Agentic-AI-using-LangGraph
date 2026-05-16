from State.HelpDeskState import HelpDeskState
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from Utils.Logger import get_logger

logger = get_logger("WEB_SEARCH")

def web_search_node(state: HelpDeskState):
    logger.info("Executing web search (DuckDuckGo)")
    
    # Check the Human-in-the-loop flag
    approved = state.get("web_search_approved", True) 
    
    if not approved:
        logger.info("Web Search denied by user. Proceeding to Orchestrator.")
        return {} # State remains unchanged

    # Use the optimized query if available, otherwise the raw question
    search_query = state.get("search_query") or state.get("question")
    logger.info(f"Searching web for: {search_query}")
    
    try:
        # Initialize the wrapper to pull the top 3 results
        ddg = DuckDuckGoSearchAPIWrapper(max_results=3)
        
        # This returns a list of dicts: [{'title': '...', 'snippet': '...', 'link': '...'}]
        results = ddg.results(search_query, max_results=3)
        
        web_docs = []
        for r in results:
            web_docs.append({
                "content": r.get("snippet", ""),
                "metadata": {
                    "category": "Web Search", 
                    "source": r.get("link", "web"),
                    "title": r.get("title", "Unknown")
                }
            })
            
    except Exception as e:
        logger.error(f"DuckDuckGo Search Failed: {e}")
        web_docs = []

    # Append the new web docs to the existing internal docs
    existing_docs = state.get("documents", [])
    existing_docs.extend(web_docs)
    
    logger.info(f"Added {len(web_docs)} web documents to context.")
    return {"documents": existing_docs}