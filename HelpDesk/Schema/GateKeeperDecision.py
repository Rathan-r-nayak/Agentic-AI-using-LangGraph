from pydantic import BaseModel, Field

class GatekeeperDecision(BaseModel):
    is_technical_it_query: bool = Field(
        description="True if the user is asking an IT question, checking a ticket, or needs action taken. False for casual greetings."
    )
    message_content: str = Field(
        description="The conversational response if this is a casual greeting. Empty if technical."
    )