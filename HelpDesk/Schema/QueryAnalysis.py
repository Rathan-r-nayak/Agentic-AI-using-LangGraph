from dataclasses import Field
from typing import Literal, Optional

from pydantic import BaseModel, create_model

from Utils.ApiResopnse import fetch_system_catalog_data


def generate_separated_query_schema() -> type[BaseModel]:
    """
    Dynamically generates a production-grade Pydantic model with 
    strict Literal choices fetched directly from the database catalog.
    """
    catalog_data = fetch_system_catalog_data()
    
    categories = set()
    items = set()
    
    for row in catalog_data:
        if row.get("catalog_category"):
            categories.add(row["catalog_category"])
        if row.get("catalog_item"):
            items.add(row["catalog_item"])
            
    # Fallback lists to keep the system robust if the network fails
    category_list = sorted(list(categories)) or ["IT Services", "Facility Services"]
    item_list = sorted(list(items)) or ["Hardware Support", "Software Licensing", "HVAC and Temperature Control"]

    # 1. Create the static Literal types using an unpacked tuple argument (* operator)
    # This syntax is fully compliant with modern type checkers like Pylance/Mypy
    CategoryType = Literal[tuple(category_list)]  # type: ignore
    ItemType = Literal[tuple(item_list)]

    # 2. Build the fields dictionary for the dynamic constructor
    # Syntax: field_name = (type, Field_metadata)
    fields = {
        "catalog_category": (
            Optional[CategoryType], 
            Field(
                default=None, 
                description=(
                    "Restricts search to a specific core corporate domain layer. "
                    "CRITICAL: Select a value ONLY if the user's issue explicitly matches one of the choices. "
                    "If you are unsure, ambiguous, or the request spans multiple areas, DO NOT guess—leave this as null."
                )
            )
        ),
        "catalog_item": (
            Optional[ItemType], 
            Field(
                default=None, 
                description=(
                    "Restricts search to a specific operational asset class or technical service group. "
                    "CRITICAL: Select a value ONLY if the user's issue explicitly implies this operational scope. "
                    "If you are not 100% certain, or if the request is generic, DO NOT guess—leave this as null."
                )
            )
        ),
        "query": (
            str, 
            Field(description="The semantic problem description or key phrases optimized to search against old resolutions.")
        )
    }

    # 3. Construct the Pydantic Model dynamically at runtime
    DynamicHistoricalSearchArgs = create_model(
        "HistoricalSearchArgs",
        **fields
    )

    return DynamicHistoricalSearchArgs