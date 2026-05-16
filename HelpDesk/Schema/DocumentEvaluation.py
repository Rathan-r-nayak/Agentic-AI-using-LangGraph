from pydantic import BaseModel, Field

class DocumentEvaluation(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the technical indicators and error logs are resolved by the document."
    )
    # If you have a reasoning field, check it too!
    reasoning: str = Field(
        description="Technical justification for the decision based on system behaviors."
    )