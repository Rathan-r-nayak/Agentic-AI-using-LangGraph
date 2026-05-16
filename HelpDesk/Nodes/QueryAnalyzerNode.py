from Config.LLMConfig import fast_llm
from Schema.QueryAnalysis import QueryAnalysis
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Logger import get_logger

logger = get_logger("QUERY_ANALYZER")

def query_analyzer_node(state: HelpDeskState):
    """
    Optimizes the raw user question for Vector DB retrieval.
    Extracts application targets and assigns technical categories only under 100% certainty.
    """
    logger.info("Analyzing & rewriting query")
    
    # Bind our updated optional-field Pydantic schema to the fast LLM
    structured_llm = fast_llm.with_structured_output(QueryAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an IT Helpdesk Query Optimizer tasked with preparing data for Vector Database lookups.
        
        CRITICAL CORE RULES:
        1. Rewrite the query to focus purely on technical keywords, error codes, and system symptoms.
        2. Identify the target application if mentioned.
        3. STRICT CATEGORIZATION RULE: Assign a 'category' (e.g., Database, Network, Software) ONLY if you are 100% certain. If there is any ambiguity, lack of details, or general phrasing, you MUST set 'category' to null.
        
        Do not assess urgency or user mood. Maintain strict focus on the technical infrastructure."""),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        analysis: QueryAnalysis = chain.invoke({"question": state["question"]})
        
        # Fall back to the original question if rewriting fails or returns empty
        optimisd_search_query = analysis.optimized_search_query if analysis.optimized_search_query else state["question"]
        
        logger.info(f"Assigned Category: {analysis.category}")
        logger.info(f"Optimized Query: {optimisd_search_query}")
        
        return {
            "category": analysis.category,  # This will seamlessly pass str or None to the state
            "search_query": optimisd_search_query
        }
    except Exception as e:
        logger.error(f"Error in Query Analyzer extraction loop: {e}")
        # Fallback state mutation to keep the graph moving
        return {
            "category": None,
            "search_query": state["question"]
        }