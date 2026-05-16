from pydantic import BaseModel, Field
from typing import List

class MemoryDecision(BaseModel):
    """
    Structured output for the memory extraction node. 
    Determines if new technical facts should be persisted to long-term storage.
    """
    
    should_write: bool = Field(
        description="Whether the last message contains new, permanent, or important technical facts worth saving."
    )
    
    memories: List[str] = Field(
        default_factory=list,
        description="A list of atomic technical facts extracted from the conversation. "
                    "Example: 'User uses Oracle DB v19c', 'User is located in New York office'."
    )
    
    reasoning: str = Field(
        description="Brief technical justification for why these specific memories were or were not extracted."
    )