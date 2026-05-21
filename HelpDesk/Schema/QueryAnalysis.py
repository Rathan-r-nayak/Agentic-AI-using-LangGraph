from pydantic import BaseModel, Field

class QueryAnalysis(BaseModel):
    category: str = Field(
        description="The technical domain of the issue (e.g., Database, Network, Software, Hardware)."
    )
    application_name: str = Field(
        description="The specific application experiencing the issue, or 'None' if unknown."
    )
    optimized_search_query: str = Field(
        description="A rewritten search string focusing purely on error codes and technical indicators."
    )


from pydantic import BaseModel, Field
from typing import Optional

class QueryAnalysis(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="The technical domain (e.g., Database, Network). Set to null (None) UNLESS you are 100% confident based strictly on explicit facts in the query."
    )
    application_name: Optional[str] = Field(
        default=None,
        description="The specific application experiencing the issue, or null (None) if unknown."
    )
    optimized_search_query: str = Field(
        description="A rewritten search string focusing purely on error codes and technical indicators."
    )