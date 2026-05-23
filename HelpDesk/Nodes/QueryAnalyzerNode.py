from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from Config.LLMConfig import fast_llm
from Schema.QueryAnalysis import generate_dynamic_query_schema
from State.HelpDeskState import HelpDeskState
from Utils.Logger import get_logger

logger = get_logger("QUERY_ANALYZER")

def query_analyzer_node(state: HelpDeskState) -> Dict[str, Any]:
    """
    Optimizes the raw user question for Vector DB retrieval.
    Dynamically fetches valid categories from the live catalog and extracts application targets.
    """
    logger.info("--- 🔍 RUNNING QUERY ANALYZER ---")
    
    # 1. Early Exit Guard: Prevent API calls if the query is mysteriously empty
    user_query = state.get("question", "").strip()
    if not user_query:
        logger.warning("Empty question received. Bypassing LLM analysis.")
        return {
            "category": "None",
            "application_name": "None",
            "search_query": ""
        }

    logger.info(f"Original Query: '{user_query}'")

    try:
        # 2. Generate schema and bind LLM
        LiveQuerySchema = generate_dynamic_query_schema()
        structured_llm = fast_llm.with_structured_output(LiveQuerySchema)

        system_prompt = (
            "You are an expert IT triage routing assistant. Analyze the user's issue, "
            "and select the single most accurate infrastructure assignment scope option. "
            "Strip away all conversational filler to create a highly optimized vector search query."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt | structured_llm
        
        # 3. Invoke the model
        analysis = chain.invoke({"question": user_query})
        
        # Initialize defaults
        category = "None"
        application_name = "None"
        
        # 4. Safely unpack the string. Added .lower() check and .strip() for safety.
        assigned_scope = getattr(analysis, 'assigned_scope', "None")
        
        if assigned_scope and assigned_scope.lower() != "none":
            parts = assigned_scope.split(" - ", 1)
            if len(parts) == 2:
                category = parts[0].strip()
                application_name = parts[1].strip()
        
        # 5. Extract query with safe fallback
        optimized_search_query = getattr(analysis, 'optimized_search_query', "")
        if not optimized_search_query:
            optimized_search_query = user_query
            
        logger.info(f"Classified Target -> Category: '{category}' | App Scope: '{application_name}'")
        logger.info(f"Optimized Search Query -> '{optimized_search_query}'")
        
        # 6. Return strictly typed dictionary
        return {
            "category": category,
            "application_name": application_name,
            "search_query": optimized_search_query
        }
        
    except Exception as e:
        logger.error(f"Error during query analysis/extraction: {e}")
        logger.warning("Failsafe triggered: Using raw query for vector search.")
        
        # Fallback state mutation to keep the graph moving
        return {
            "category": "None",
            "application_name": "None",
            "search_query": user_query
        }