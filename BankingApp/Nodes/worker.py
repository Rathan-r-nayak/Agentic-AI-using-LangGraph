from langchain.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from State.banking_state import WorkerState
from Config.llm_config import primary_llm


# Assuming you fetch your tools dynamically from the MCP server here
# tools = mcp_client.get_tools()
banking_tools = [] # Replace with your actual tools list

WORKER_SYSTEM_PROMPT = """You are an autonomous Banking Task Worker.
Your objective is to execute the specific task assigned to you.

Rules:
1. Use the provided tools to fetch required information or execute actions.
2. ALWAYS query the internal knowledge base first.
3. Only use the web search tool (duckduckgo) if the internal KB returns insufficient data.
4. Output your final answer clearly, using Markdown tables for tabular data.
"""

# Compile the ReAct subgraph
react_worker_graph = create_react_agent(
    model=primary_llm,
    tools=banking_tools,
    state_modifier=WORKER_SYSTEM_PROMPT
)

from Utils.Logger import get_logger

logger = get_logger("WORKER_NODE")

def worker_node_function(state: WorkerState):
    task = state["task"]
    logger.info(f"⚙️ WORKER STARTED: Executing Task -> {task.task_id}: {task.description}")
    
    # create_react_agent expects a dict with a "messages" key
    worker_input = {
        "messages": [HumanMessage(content=f"Execute this task: {task.description}")]
    }
    
    # Invoke the compiled ReAct subgraph
    try:
        result = react_worker_graph.invoke(worker_input)
        
        # Extract the final AI response from the ReAct agent's internal state
        final_answer = result["messages"][-1].content
        logger.info(f"✅ WORKER FINISHED: {task.task_id}")
        
    except Exception as e:
        logger.error(f"Worker failed on task {task.task_id}: {e}")
        final_answer = f"Error executing task: {str(e)}"
    
    # Return a dictionary that targets the Annotated reducer in the parent BankingState
    # The string is formatted to give context to the Synthesizer node later
    return {
        "worker_responses": [f"--- Result for {task.task_id} ---\n{final_answer}"]
    }