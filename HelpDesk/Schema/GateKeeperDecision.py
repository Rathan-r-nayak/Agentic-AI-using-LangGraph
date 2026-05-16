from pydantic import BaseModel, Field

class GatekeeperDecision(BaseModel):
    is_technical_it_query: bool = Field(
        description="True ONLY IF the input is a real, explicit IT support request, technical question, or system failure report. False for greetings, casual chat, or non-IT topics."
    )
    message_content: str = Field(
        description="If is_technical_it_query is False, provide a warm conversational response leveraging Long-Term Facts. If True, leave this blank."
    )