from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class HelpDeskState(TypedDict, total=False):
    # Chat Variables
    question: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    requires_rag: bool
    
    # Routing & RAG
    intent: str
    category: str
    application_name: str
    search_query: str
    is_sufficient: bool
    documents: list[dict]
    
    # Orchestration
    tasks: list[str]
    long_term_facts: str
    worker_results: Annotated[list[str], operator.add]