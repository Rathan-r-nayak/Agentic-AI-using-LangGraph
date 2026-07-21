import os
import httpx
import streamlit as st
from Utils.Logger import get_logger


FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/api")

logger = get_logger("APIRESPONSE")

def fetch_system_catalog_data():
    """Fetches the configuration lookup catalog directly from the FastAPI backend."""
    try:
        response = httpx.get(f"{FASTAPI_URL}/tickets/catalog-options") # Adjust path to your actual endpoint
        if response.status_code == 200:
            data = response.json() 
            logger.info(f"Fetched {len(data)} values from the /tickets/catalog-options")
            return data
    except Exception as e:
        logger.error(f"⚠️ Failed to connect to backend catalog service: {e}")
    return []