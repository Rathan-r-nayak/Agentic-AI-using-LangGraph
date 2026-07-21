from langgraph.graph import END
from BankingApp.State import banking_state
from Utils.Logger import get_logger

logger = get_logger("Primary Classifier")

def route_triage(state: banking_state):
    """
    Reads the state output from the triage_router.
    Routes to the orchestrator if a workflow is needed, otherwise ends the graph.
    """

    if(state.get("requires_worflow")):
        logger.info("➡️ ROUTING: Sending to Orchestrator for task planning.")
        return "orchestrator"

    logger.info("🛑 ROUTING: Direct response generated. Ending graph.")
    return END