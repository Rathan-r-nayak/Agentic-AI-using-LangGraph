from pydantic import BaseModel, Field
from typing import List, Literal

class SupportTask(BaseModel):
    """An individual task delegated to a parallel worker node."""
    id: int
    title: str
    objective: str = Field(
        description="One sentence describing exactly what this worker must extract or generate based on the documents."
    )
    technical_requirements: List[str] = Field(
        min_length=1, 
        max_length=4, 
        description="Specific technical details to include (e.g., 'Must include the ORA-12154 error code', 'Must mention TNSNAMES.ORA')."
    )
    task_type: Literal["root_cause", "resolution_steps", "preventive_advice", "user_communication"] = Field(
        description="The specific domain of this worker. Use 'resolution_steps' exactly once."
    )

class ResolutionPlan(BaseModel):
    """The master plan created by the Orchestrator for the workers."""
    incident_summary: str = Field(
        description="A clear, technical one-sentence summary of the user's issue."
    )
    # severity: Literal["Low", "Medium", "High", "Critical"] = Field(
    #     description="The assessed severity of the issue based on the user's report."
    # )
    tasks: List[SupportTask] = Field(
        description="The list of parallel tasks the workers will execute to build the final response."
    )