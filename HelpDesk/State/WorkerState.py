from typing import TypedDict, List, Dict
from Schema.ResolutionPlan import SupportTask

class WorkerState(TypedDict):
    task: SupportTask
    documents: List[Dict] 
    question: str         
    long_term_facts: str   # <-- NEW: The user's permanent facts
    chat_history: str      # <-- NEW: What they just said moments ago