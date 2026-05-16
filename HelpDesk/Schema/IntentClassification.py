from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: str = Field(
        description="Classify as 'greeting' or 'technical_query'."
    )