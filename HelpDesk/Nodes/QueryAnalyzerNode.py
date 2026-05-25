from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from Config.LLMConfig import fast_llm
# Import your new schema generator factory function
from Schema.QueryAnalysis import generate_separated_query_schema 
from State.HelpDeskState import HelpDeskState
from Utils.Logger import get_logger

logger = get_logger("QUERY_ANALYZER")

def query_analyzer_node(state: HelpDeskState) -> Dict[str, Any]:
    """
    Optimizes the raw user question for Vector DB retrieval.
    Dynamically fetches valid domains and sub-items from the live catalog 
    and maps them to distinct search attributes.
    """
    logger.info("--- 🔍 RUNNING QUERY ANALYZER (SEPARATED SCHEMAS) ---")
    
    # 1. Early Exit Guard
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
        # 2. Use your new dual-field schema generator and bind it to the LLM
        LiveQuerySchema = generate_separated_query_schema()
        structured_llm = fast_llm.with_structured_output(LiveQuerySchema)

        system_prompt = (
            "You are an expert IT triage routing assistant. Analyze the user's issue, "
            "and map it to the single most accurate category and item parameters provided in the schema constraints. "
            "Strip away all conversational conversational filler to create a highly optimized vector search query."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt | structured_llm
        
        # 3. Invoke the model
        analysis = chain.invoke({"question": user_query})
        
        # 4. Extract fields cleanly without string manipulations or splitting!
        category = getattr(analysis, 'catalog_category', None) or "None"
        application_name = getattr(analysis, 'catalog_item', None) or "None"
        
        # 5. Extract query with safe fallback
        # (Assuming your new schema model uses 'query' or 'optimized_search_query')
        optimized_search_query = getattr(analysis, 'query', "")
        if not optimized_search_query:
            optimized_search_query = getattr(analysis, 'optimized_search_query', user_query)
            
        logger.info(f"Classified Target -> Category: '{category}' | Item Scope: '{application_name}'")
        logger.info(f"Optimized Search Query -> '{optimized_search_query}'")
        
        # 6. Return strictly typed dictionary to mutate your graph state keys
        return {
            "category": category,
            "application_name": application_name,
            "search_query": optimized_search_query
        }
        
    except Exception as e:
        logger.error(f"Error during separated query analysis: {e}")
        logger.warning("Failsafe triggered: Using raw query for vector search.")
        
        return {
            "category": "None",
            "application_name": "None",
            "search_query": user_query
        }