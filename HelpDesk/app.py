import asyncio
import os
import logging
import warnings

from McpClient.mcp_client import ChatbotMCPClient

MCP_SSE_URL = "http://127.0.0.1:8000/mcp/sse"
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_triage_db")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/api")

# ==========================================
# 1. SUPPRESS WARNINGS & LOGS
# ==========================================
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for noisy_logger in ["transformers", "huggingface_hub", "urllib3", "httpx", "absl"]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

# ==========================================
# 2. IMPORTS
# ==========================================
import streamlit as st
from langchain.messages import HumanMessage, AIMessage
from Utils.ApiResopnse import fetch_system_catalog_data

import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import backend components
from Utils.DocumentProcessor import process_and_index_files
from main import get_compiled_app  # Ensure this returns the UNCOMPILED workflow
from Utils.VectorStore import add_documents_to_store, get_indexed_files
from Utils.Helpers import analyze_image_context
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

st.set_page_config(page_title="Smart Issue Triage - IT Support 🎧", layout="wide", page_icon="🤖")

@st.cache_data(ttl=300)
def fetch_system_catalog():
    """Fetches the configuration lookup catalog directly from the FastAPI backend."""
    return fetch_system_catalog_data()

# ==========================================
# 3. ASYNC DATABASE HELPERS
# ==========================================
# These safely interact with Postgres & LangGraph without crashing Streamlit

async def fetch_history_async(config_dict):
    """Safely fetch chat history from the async checkpointer."""
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        app = get_compiled_app().compile(checkpointer=checkpointer)
        state = await app.aget_state(config_dict)
        formatted_messages = []
        if state and state.values and "messages" in state.values:
            for msg in state.values["messages"]:
                if isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    # Filter out empty tool-calling messages from the UI
                    if msg.content:
                        formatted_messages.append({"role": "assistant", "content": msg.content})
        return formatted_messages

async def check_graph_state_async(config_dict):
    """Safely check if the graph is paused at an interrupt."""
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        app = get_compiled_app().compile(checkpointer=checkpointer)
        return await app.aget_state(config_dict)

async def resume_graph_execution(approved: bool, config_dict):
    """Safely update the state and resume the graph."""
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        app = get_compiled_app().compile(checkpointer=checkpointer)
        
        # 1. Update the state (e.g., approve web search)
        await app.aupdate_state(config_dict, {"web_search_approved": approved})
        
        # 2. Re-establish MCP connection before resuming so workers have tools
        mcp_client = ChatbotMCPClient(sse_url=MCP_SSE_URL)
        await mcp_client.connect()
        try:
            config_dict["configurable"]["mcp_tools"] = mcp_client.get_tools()
            # 3. Resume the graph by passing None
            return await app.ainvoke(None, config=config_dict)
        finally:
            await mcp_client.disconnect()


# ==========================================
# 4. SESSION STATE & CONFIG
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "default_user"}}

if "messages" not in st.session_state:
    # Safely load historical messages using our async helper
    st.session_state.messages = asyncio.run(fetch_history_async(config))

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🔐 Access Control")
    role = st.radio("Select Persona:", ["👤 User / Employee", "🛡️ IT Admin"], index=0)
    
    st.divider()
    st.info(f"Thread ID: {st.session_state.thread_id}")
    if st.button("Reset Session"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ==========================================
# 6. ADMIN INTERFACE
# ==========================================
if role == "🛡️ IT Admin":
    st.header("Knowledge Base Manager")
    st.write("Upload technical manuals or policy docs to the Azure-powered Vector Store.")
    
    catalog_data = fetch_system_catalog()
    category_mapping = {}
    if catalog_data and isinstance(catalog_data, list):
        for item in catalog_data:
            cat = item.get("catalog_category")
            itm = item.get("catalog_item")
            if cat and itm:
                if cat not in category_mapping:
                    category_mapping[cat] = set()
                category_mapping[cat].add(itm)

    if not category_mapping:
        category_mapping = {"No valid categories found": ["No valid items found"]}

    available_categories = sorted(list(category_mapping.keys()))

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader("Select PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    with col2:
        selected_category = st.selectbox("Assign Core Category", options=available_categories)
        matching_items = sorted(list(category_mapping.get(selected_category, [])))
        selected_item = st.selectbox("Assign Technical Scope Item", options=matching_items)

    if st.button("🚀 Index to Vector DB"):
        if uploaded_files:
            if selected_category == "No valid categories found":
                st.error("Cannot index documents to an invalid catalog destination.")
                st.stop()

            temp_files_info = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_files_info.append((tmp.name, uploaded_file.name)) 

            with st.spinner("Processing and chunking files..."):
                success = process_and_index_files(temp_files_info, selected_category, selected_item)
            
            for path, _ in temp_files_info:
                if os.path.exists(path):
                    os.remove(path)
            
            if success:
                st.success(f"Successfully indexed {len(uploaded_files)} files under {selected_category} -> {selected_item}!")
            else:
                st.error("Something went wrong during indexing.")
        else:
            st.warning("Please upload at least one file before indexing.")
                
    st.divider()
    st.subheader("📚 Current Knowledge Base")
    indexed_files = get_indexed_files()
    
    if indexed_files:
        with st.expander(f"View {len(indexed_files)} Indexed Documents"):
            for idx, filename in enumerate(indexed_files, 1):
                st.markdown(f"**{idx}.** {filename}")
    else:
        st.info("No documents are currently indexed in the Vector DB.")

# ==========================================
# 7. USER INTERFACE
# ==========================================
else:
    st.header("Smart Issue Triage - IT Helpdesk")
    st.caption("AI-Powered Resolution Engine")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- HUMAN-IN-THE-LOOP (Web Search Approval) ---
    # Use the async helper to check if the graph is paused
    current_state = asyncio.run(check_graph_state_async(config))
    
    if current_state and current_state.next and "web_search_node" in current_state.next:
        st.warning("⚠️ Local context is insufficient. Should I search the internet?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Web Search"):
                result = asyncio.run(resume_graph_execution(approved=True, config_dict=config))
                if result and "messages" in result:
                    response = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        with c2:
            if st.button("❌ Deny (Use Context Only)"):
                result = asyncio.run(resume_graph_execution(approved=False, config_dict=config))
                if result and "messages" in result:
                    response = result["messages"][-1].content
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # --- CHAT INPUT & VISION ---
    elif user_input := st.chat_input("Ask a question or upload a screenshot...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
        
        prompt_text = user_input.text
        final_prompt = prompt_text
        
        if user_input.files:
            t_file = tempfile.NamedTemporaryFile(delete=False, suffix=user_input.files[0].name)
            t_path = t_file.name
            try:
                t_file.write(user_input.files[0].getbuffer())
                t_file.close() 
                analysis = analyze_image_context(t_path)
                final_prompt += f"\n\n[Vision Context]: {analysis}"
            finally:
                if os.path.exists(t_path):
                    os.remove(t_path) 

        st.chat_message("user").markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        with st.chat_message("assistant"):
            
            async def process_agent_request(prompt):
                mcp_client = ChatbotMCPClient(sse_url=MCP_SSE_URL)
                await mcp_client.connect()

                try:
                    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
                        await checkpointer.setup()
                        workflow = get_compiled_app() 
                        app = workflow.compile(checkpointer=checkpointer)

                        config["configurable"]["mcp_tools"] = mcp_client.get_tools()
                
                        graph_input = {
                            "question": prompt,
                            "messages": [HumanMessage(content=prompt)] 
                        }
                    
                        return await app.ainvoke(graph_input, config=config)
                finally:
                    await mcp_client.disconnect()
            
            # Execute
            result = asyncio.run(process_agent_request(final_prompt))
            
            # Route UI based on state
            next_state = asyncio.run(check_graph_state_async(config))
            
            if next_state and next_state.next:
                st.rerun()
            else:
                # Extract the output cleanly
                if result and isinstance(result, dict):
                    if "messages" in result and len(result["messages"]) > 0:
                        response = result["messages"][-1].content
                    elif "worker_results" in result:
                        response = "\n\n".join(result["worker_results"])
                    else:
                        response = str(result)
                else:
                    response = "Agent completed tasks, but no output was parsed."
                    
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})