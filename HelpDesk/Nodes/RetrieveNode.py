from typing import Dict, Any
from State.HelpDeskState import HelpDeskState
from Utils.VectorStore import get_vector_store
from Utils.Logger import get_logger

logger = get_logger("RETRIEVE")

def retrieve_node(state: HelpDeskState) -> Dict[str, Any]:
    """
    Retrieves relevant IT manuals from ChromaDB using semantic search 
    and dynamic metadata filtering.
    """
    logger.info("--- 📥 RUNNING DOCUMENT RETRIEVAL ---")
    
    query = state.get("search_query") or state.get("question", "")
    if not query.strip():
        logger.warning("Empty search query received. Skipping retrieval.")
        return {"documents": []}
        
    category = state.get("category", "")
    
    # Safely build the ChromaDB filter
    search_filter = None
    if category and category.lower() != "none":
        search_filter = {"category": category}

    logger.info(f"Target Query: '{query}' | Active Filter: {search_filter}")

    try:
        vector_store = get_vector_store()
        
        # Perform Similarity Search
        raw_docs = vector_store.similarity_search(
            query=query, 
            k=4, 
            filter=search_filter
        )
        
        formatted_docs = []
        for doc in raw_docs:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        if formatted_docs:
            # Log a clean summary of what was found
            sources = [doc.get("metadata", {}).get("source", "Unknown") for doc in formatted_docs]
            # Use a set to remove duplicate source names in the log
            unique_sources = ", ".join(list(set(sources)))
            logger.info(f"Found {len(formatted_docs)} chunks from: {unique_sources}")
        else:
            logger.warning("No matches found in local Vector DB.")

    except Exception as e:
        logger.error(f"Retrieval Error: {e}")
        formatted_docs = []
        
    return {"documents": formatted_docs}