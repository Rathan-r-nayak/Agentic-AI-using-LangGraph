import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver

# 1. Import State & Utilities
from HelpDesk.Routers.Routers import distribute_tasks, route_after_evaluation, route_after_greeting
from State.HelpDeskState import HelpDeskState
from Utils.Helpers import format_chat_history

# 2. Import All Nodes
from HelpDesk.Nodes.GateKeeperNode import gatekeeper_node
from Nodes.QueryAnalyzerNode import query_analyzer_node
from Nodes.RetrieveNode import retrieve_node
from Nodes.EvaluatorNode import evaluator_node
from Nodes.WebSearchNode import web_search_node
from Nodes.OrchestratorNode import orchestrator_node
from Nodes.WorkerNode import worker_node
from Nodes.MergeNode import merge_node
from Nodes.CritiqueNode import critique_node
from Nodes.RememberNode import remember_node
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool
from Utils.Logger import get_logger
import os

DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_triage_db")



logger = get_logger("MAIN")

# ==========================================
# BUILD THE GRAPH
# ==========================================

logger.info("Compiling helpdesk chat workflow")

workflow = StateGraph(HelpDeskState)


# ------------- Nodes -------------
workflow.add_node("gatekeeper_node", gatekeeper_node)
workflow.add_node("query_analyzer_node", query_analyzer_node)
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("evaluator_node", evaluator_node)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("worker_node", worker_node)
workflow.add_node("merge_node", merge_node)
workflow.add_node("critique_node", critique_node)
workflow.add_node("remember_node", remember_node)



# ------------- Edges -------------
# 1. give query to the gatekeeper node
workflow.add_edge(START, "gatekeeper_node")

# 2. Greeting -> RAG Pipeline or Remember/End
workflow.add_conditional_edges(
    "gatekeeper_node",
    route_after_greeting,
    {
        "query_analyzer_node": "query_analyzer_node",
        "remember_node": "remember_node"
    }
)

# 3. Retrieval & Evaluation
workflow.add_edge("query_analyzer_node", "retrieve_node")
workflow.add_edge("retrieve_node", "evaluator_node")

workflow.add_conditional_edges(
    "evaluator_node",
    route_after_evaluation,
    {
        "orchestrator_node": "orchestrator_node",
        "web_search_node": "web_search_node"
    }
)

# Web Search leads back to Orchestrator
workflow.add_edge("web_search_node", "orchestrator_node")

# 4. Parallel Workers (Map-Reduce)
# The "Map" step
workflow.add_conditional_edges(
    "orchestrator_node",
    distribute_tasks,
    ["worker_node"]
)

# The "Reduce" step: All workers automatically flow into merge_node 
# once they have all finished their parallel execution.
workflow.add_edge("worker_node", "merge_node")

# 5. Cleanup & Safety
workflow.add_edge("merge_node", "critique_node")
workflow.add_edge("critique_node", "remember_node")
workflow.add_edge("remember_node", END)

# ==========================================
# COMPILE WITH CHECKPOINTER
# ==========================================



pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={"autocommit": True} # Required for LangGraph savers
)

# Setup Checkpointer (Short-term thread snapshots)
checkpointer = PostgresSaver(pool)
checkpointer.setup() 

# Setup Store (Long-term cross-thread global memory)
store = PostgresStore(pool)
store.setup()





# Using MemorySaver for current session history
# memory = MemorySaver()

app = workflow.compile(
    checkpointer=checkpointer, 
    store=store,
    interrupt_before=["web_search_node"]
)

logger.info("Graph compiled successfully")