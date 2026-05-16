from pydantic import BaseModel, Field

class CritiqueReview(BaseModel):
    """Evaluates the final generation for safety, tone, and PII."""
    
    is_safe: bool = Field(
        description="True if the text is free of sensitive info (passwords, raw internal IPs) and the tone is professional."
    )
    scrubbed_text: str = Field(
        description="The final output text. If is_safe is False, rewrite the text to remove the violation. If True, return the text exactly as provided."
    )
    reasoning: str = Field(
        description="Brief explanation of the safety decision."
    )