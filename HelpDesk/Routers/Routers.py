from Utils.Helpers import format_chat_history
from HelpDesk.State import HelpDeskState
from langgraph.constants import Send



# after the gatekeeper node, this router will be executed
def route_after_greeting(state: HelpDeskState):
    generation = state.get("generation", "")
    requires_rag : bool = state.get("requires_rag", True)

    if(requires_rag):
        return "query_analyzer_node"

    return "remember_node"


def route_after_evaluation(state: HelpDeskState):
    if state.get("is_sufficient"):
        return "orchestrator_node"
    return "web_search_node"

def distribute_tasks(state: HelpDeskState):
    """
    MAP: This sends tasks to parallel worker_nodes
    """
    ltm = state.get("long_term_facts", "")
    # Ensure stm is a string for the worker prompts
    stm = format_chat_history(state.get("messages", []))
    
    return [
        Send("worker_node", {
            "task": t, 
            "documents": state.get("documents", []),
            "question": state["question"],
            "long_term_facts": ltm,
            "chat_history": stm
        }) for t in state.get("tasks", [])
    ]
