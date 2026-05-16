import httpx
import os
import ssl
import warnings
from urllib3.exceptions import InsecureRequestWarning
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Retrieve the key from environment variables
# If the key is missing, it will return None
API_KEY = os.getenv("TCS_API_KEY")


if not API_KEY:
    raise ValueError("TCS_API_KEY not found! Ensure your .env file is set up correctly.")


# ==========================================
# 1. LAB ENVIRONMENT SSL BYPASS
# ==========================================
# This ensures you don't get SSL errors in the TCS lab environment
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
warnings.simplefilter('ignore', InsecureRequestWarning)

# Initialize the persistent client
client = httpx.Client(verify=False, timeout=120.0)

# ==========================================
# 2. CONFIGURATION & KEYS
# ==========================================
BASE_URL = "https://genailab.tcs.in"





# ==========================================
# 3. LLM INITIALIZATIONS
# ==========================================

primary_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o",  # Back to a working deployment
    api_key=API_KEY,
    http_client=client,
    temperature=0.2
)

# (Keep your fast_llm and reasoning_llm as they were)

# FAST LLM: For Routing, Classification, and Summarization
fast_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o-mini", 
    api_key=API_KEY,
    http_client=client,
    temperature=0
)

# REASONING LLM: Specifically for your Orchestrator node
reasoning_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure_ai/genailab-maas-DeepSeek-R1", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)


embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=BASE_URL,
    azure_deployment="azure/genailab-maas-text-embedding-3-large",
    api_key=API_KEY,
    openai_api_version="2023-05-15",
    http_client=client
)

reasoning_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure_ai/genailab-maas-DeepSeek-R1", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)