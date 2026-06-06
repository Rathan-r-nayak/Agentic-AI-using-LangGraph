from Config.LLMConfig import primary_llm, fast_llm, openrouter_llm, gemma_llm
from Schema.ResolutionPlan import ResolutionPlan
from State.HelpDeskState import HelpDeskState
from langchain_core.prompts import ChatPromptTemplate
from Utils.Helpers import fetch_user_ltm, format_chat_history
from Utils.Logger import get_logger
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore


logger = get_logger("ORCHESTRATOR")

def orchestrator_node(state: HelpDeskState, config: RunnableConfig, store: BaseStore):
    logger.info("--- 🧠 RUNNING ORCHESTRATOR ---")
    logger.info("Drafting resolution plan with memory context")
    
    docs = state.get("documents", [])
    question = state.get("question", "")
    # ltm_facts = state.get("long_term_facts", "No known facts.")
    ltm_facts = fetch_user_ltm(config, store)
    stm_history = format_chat_history(state.get("messages", []))

    mcp_tools = config.get("configurable", {}).get("mcp_tools", [])

    if mcp_tools:
        tool_descriptions = "\n".join([
            f"- Tool Name: '{tool.name}' | Description: {tool.description}" 
            for tool in mcp_tools
        ])
    else:
        tool_descriptions = "- No external tools currently available."
    
    # Safely format docs in case metadata is missing
    doc_text = "\n\n".join([f"Source: {d.get('metadata', 'Unknown')}\nContent: {d.get('content', '')}" for d in docs])

    structured_llm = gemma_llm.with_structured_output(ResolutionPlan)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the L3 Lead IT Service Manager operating as a high-velocity Triage Orchestrator.
        Your job is to parse the user's issue and compile a targeted 'Resolution Plan' containing discrete execution tasks for your worker nodes.
        
        CRITICAL CONTEXT:
        - User Environment & Profile (Long-Term): {ltm_facts}
        - Chat History (Short-Term): {stm_history}
        
        AVAILABLE WORKER CAPABILITIES (TOOLS):
        {tool_descriptions}
        
        ORCHESTRATION MODES & DYNAMIC TASK RULES:
        
        MODE A: DATA LOOKUP / STATUS QUERY (e.g., "Give me details for ticket X", "What is the status of issue Y")
        - CRITICAL: Generate EXACTLY ONE task: 'Retrieve Ticket Details'.
        - Instruct the worker to execute the relevant retrieval tool from the capabilities list.
        - DO NOT generate tasks for root cause analysis, preventive advice, or escalation. The user is only looking for real-time status parameters.
        
        MODE B: LIVE INCIDENT RESOLUTION (e.g., "My database connection is failing", "The screen has vertical lines")
        - Generate focused, parallel technical tasks based on the following rules:
          1. 'root_cause': Analyze logs or structural context to isolate the breakdown mechanism.
          2. 'resolution_steps': Formulate deterministic remediation workflows.
          3. 'preventive_advice': Engineering recommendations to avoid regression.
          4. 'escalation' (Optional): Trigger only if initial isolation steps are projected to breach SLA parameters or require tier-3 privileges.
        
        TASK ENGINEERING INSTRUCTIONS:
        1. Tailor all plan parameters to the User's explicit OS, shell environment, and development dependencies extracted from 'User Environment & Profile'.
        2. Inspect 'Chat History'. If the user has already executed an administrative action or troubleshooting step, explicitly instruct workers to skip that step and proceed to next-hop diagnostics.
        3. Maintain a strict engineering and IT infrastructure taxonomy. Avoid all medical or healthcare analogies completely.
        """),
        ("human", "Current Issue: {question}\n\nRetrieved Manuals:\n{doc_text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        plan = chain.invoke({
            "ltm_facts": ltm_facts,
            "stm_history": stm_history,
            "question": question, 
            "doc_text": doc_text,
            "tool_descriptions": tool_descriptions
        })
        
        return {"tasks": [task.model_dump() for task in plan.tasks]}
        
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        # FALLBACK: Provide default tasks to prevent the UI from crashing
        return {"tasks": [
            {"title": "Root Cause Analysis", "objective": "Analyze the root cause", "technical_requirements": []}, 
            {"title": "Resolution Steps", "objective": "Provide step-by-step resolution", "technical_requirements": []}, 
            {"title": "Preventive Advice", "objective": "Offer preventive advice", "technical_requirements": []}
        ]}