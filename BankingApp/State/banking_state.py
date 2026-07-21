import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_classic.schema import BaseMessage
from langgraph.graph import add_messages


class BankingState(TypedDict, total=False):
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    requires_worflow: bool
    documents: list[dict]
    is_sufficient: bool
    tasks: list[str]
    worker_results: Annotated[list[str], operator.add]
    generation: str

class WorkerState(TypedDict):
    task: Any # Or use your Task Pydantic model