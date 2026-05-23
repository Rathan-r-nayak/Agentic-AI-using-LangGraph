from typing import Dict, Any
from State.HelpDeskState import HelpDeskState
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from Utils.Logger import get_logger

logger = get_logger("WEB_SEARCH")

def web_search_node(state: HelpDeskState) -> Dict[str, Any]:
    logger.info("--- 🌐 RUNNING WEB SEARCH ---")
    
    # Check the Human-in-the-loop flag
    approved = state.get("web_search_approved", True) 
    if not approved:
        logger.info("Web Search denied by user constraints. Bypassing node.")
        return {} # State remains unchanged

    search_query = state.get("search_query") or state.get("question", "")
    if not search_query.strip():
        logger.warning("Empty search query. Skipping DuckDuckGo search.")
        return {}
        
    logger.info(f"Pinging DuckDuckGo for: '{search_query}'")
    
    try:
        ddg = DuckDuckGoSearchAPIWrapper(max_results=3)
        results = ddg.results(search_query, max_results=3)
        
        web_docs = []
        found_titles = []
        
        for r in results:
            title = r.get("title", "Unknown Site")
            found_titles.append(title)
            
            web_docs.append({
                "content": r.get("snippet", ""),
                "metadata": {
                    "category": "Web Search", 
                    "source": r.get("link", "web"),
                    "title": title
                }
            })
            
        logger.info(f"Retrieved {len(web_docs)} web snippets: {', '.join(found_titles)}")
            
    except Exception as e:
        logger.error(f"DuckDuckGo API Request Failed: {e}")
        web_docs = []

    # Safely append to existing context
    existing_docs = state.get("documents", [])
    # Using list concatenation to avoid mutating the original state list directly
    updated_docs = existing_docs + web_docs 
    
    return {"documents": updated_docs}