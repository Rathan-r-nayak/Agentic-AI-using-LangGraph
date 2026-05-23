from pydantic import BaseModel, Field
from typing import Optional
from Utils.Logger import get_logger
from Utils.ApiResopnse import fetch_system_catalog_data
from typing import List, Literal

logger = get_logger("QUERY_ANALYSIS")



def generate_dynamic_query_schema() -> type[BaseModel]:
    combined_options = []
    catalog_data = fetch_system_catalog_data()


    if catalog_data and isinstance(catalog_data, list):
        try:
            unique_mappings = {
                f"{item['catalog_category']} - {item['catalog_item']}" 
                for item in catalog_data 
                if item.get("catalog_category") and item.get("catalog_item")
            }

            if unique_mappings:
                combined_options = sorted(list(unique_mappings))
                logger.info(f"Successfully generated schema with {len(combined_options)} unique catalog mappings.")
        except Exception as e:
            logger.error(f"Failed parsing structural fields. Using defaults. Error: {e}")
            
    if "None" not in combined_options:
        combined_options.append("None")




    class DynamicQueryAnalysis(BaseModel):
        # --- FIX 2: Change to strict string, default to "None" ---
        assigned_scope: str = Field(
            default="None",
            description="The classified scope of the issue. You MUST pick exactly one string from the enum list. If you are not 100% confident, you MUST select 'None'.",
            json_schema_extra={
                "enum": combined_options
            }
        )
        optimized_search_query: str = Field(
            description="A rewritten search string focusing purely on error codes and technical indicators."
        )

    return DynamicQueryAnalysis





# class QueryAnalysis(BaseModel):
#     category: Optional[str] = Field(
#         default=None,
#         description="The technical domain (e.g., Database, Network). Set to null (None) UNLESS you are 100% confident based strictly on explicit facts in the query."
#     )
#     application_name: Optional[str] = Field(
#         default=None,
#         description="The specific application experiencing the issue, or null (None) if unknown."
#     )
#     optimized_search_query: str = Field(
#         description="A rewritten search string focusing purely on error codes and technical indicators."
#     )