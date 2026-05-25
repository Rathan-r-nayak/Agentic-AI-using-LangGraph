import os
import logging
import warnings

# 1. Mute HuggingFace Transformers BEFORE they are ever imported
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # (Optional but highly recommended for Streamlit apps)

# 2. Suppress specific deprecation/alias warnings
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
warnings.filterwarnings("ignore", message=".*Deserializing unregistered type.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 3. Silence noisy third-party libraries by setting them to ERROR
for noisy_logger in ["transformers", "huggingface_hub", "urllib3", "httpx", "absl"]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)

# Set the root logger to WARNING to hide generic INFO logs from other packages
logging.getLogger().setLevel(logging.WARNING)

# ==========================================
# 1. NOW IT IS SAFE TO IMPORT EVERYTHING ELSE
# ==========================================
import streamlit as st
from langchain.messages import HumanMessage
from Utils.ApiResopnse import fetch_system_catalog_data

import tempfile
import uuid
import httpx
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import backend components
from Utils.HistoryLoaders import load_chat_history
from Utils.DocumentProcessor import process_and_index_files
from main import get_compiled_app
from Utils.VectorStore import add_documents_to_store, get_indexed_files
from Utils.Helpers import analyze_image_context



FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/api")
st.set_page_config(page_title="Smart Issue Triage - IT Support 🎧", layout="wide", page_icon="🤖")


@st.cache_resource
def load_agent():
    """Caches the LangGraph compilation and DB pool so it doesn't crash Postgres."""
    return get_compiled_app()



@st.cache_data(ttl=300)  # Caches the catalog for 5 minutes
def fetch_system_catalog():
    """Fetches the configuration lookup catalog directly from the FastAPI backend."""
    return fetch_system_catalog_data()



langgraph_app = load_agent()


# ==========================================
# 1. SESSION STATE & CONFIG
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Sync with LangGraph Checkpointer
config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": "default_user"}}

# Load historical messages exclusively on first boot or thread reset
if "messages" not in st.session_state:
    # Use your new utility function to pull from the Postgres Checkpointer
    st.session_state.messages = load_chat_history(langgraph_app, st.session_state.thread_id)

# ==========================================
# 2. SIDEBAR - ROLE SELECTION
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
# 3. ADMIN INTERFACE (UPLOAD LOGIC)
# ==========================================
# ==========================================
# ADMIN PANEL: KNOWLEDGE BASE MANAGER
# ==========================================
if role == "🛡️ IT Admin":
    st.header("Knowledge Base Manager")
    st.write("Upload technical manuals or policy docs to the Azure-powered Vector Store.")
    
    # 1. Fetch data dynamically from FastAPI (list of dicts)
    catalog_data = fetch_system_catalog()

    # 2. Build a structured mapping dictionary instead of flat composite strings
    # Format: {"IT Services": {"Hardware Support", "Software Licensing"}, ...}
    category_mapping = {}
    if catalog_data and isinstance(catalog_data, list):
        for item in catalog_data:
            cat = item.get("catalog_category")
            itm = item.get("catalog_item")
            if cat and itm:
                if cat not in category_mapping:
                    category_mapping[cat] = set()
                category_mapping[cat].add(itm)

    # 3. Handle fallbacks gracefully if the backend response is empty
    if not category_mapping:
        category_mapping = {"No valid categories found": ["No valid items found"]}

    # Extract unique, sorted categories for the first dropdown
    available_categories = sorted(list(category_mapping.keys()))

    # --- UPLOAD & ASSIGN UI ---
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader("Select PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    
    with col2:
        # Step A: Admin selects the main Core Category Domain
        selected_category = st.selectbox(
            "Assign Core Category", 
            options=available_categories
        )
        
        # Step B: Dynamically resolve and sort items tied ONLY to that selected category
        matching_items = sorted(list(category_mapping.get(selected_category, [])))
        
        # Step C: Admin selects the sub-scope item from the filtered subset
        selected_item = st.selectbox(
            "Assign Technical Scope Item", 
            options=matching_items
        )

    # --- INDEXING LOGIC ---
    if st.button("🚀 Index to Vector DB"):
        if uploaded_files:
            # Look up values directly without string splitting (.split(" - ")) acrobatics
            if selected_category == "No valid categories found":
                st.error("Cannot index documents to an invalid catalog destination.")
                st.stop()

            temp_files_info = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_files_info.append((tmp.name, uploaded_file.name)) 

            with st.spinner("Processing and chunking files..."):
                # 💡 ADJUSTMENT NOTE: If your backend endpoint 'process_and_index_files' 
                # has been upgraded to take both fields separately, update it right here:
                success = process_and_index_files(temp_files_info, selected_category, selected_item)
            
            # Clean up temp files from disk
            for path, _ in temp_files_info:
                if os.path.exists(path):
                    os.remove(path)
            
            if success:
                st.success(f"Successfully indexed {len(uploaded_files)} files under {selected_category} -> {selected_item}!")
            else:
                st.error("Something went wrong during indexing.")
        else:
            st.warning("Please upload at least one file before indexing.")
                
    # --- VIEW CURRENT KNOWLEDGE BASE ---
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
# 4. USER INTERFACE (CHAT & VISION)
# ==========================================
else:
    st.header("Smart Issue Triage - IT Helpdesk")
    st.caption("AI-Powered Resolution Engine")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- HUMAN-IN-THE-LOOP (Web Search Approval) ---
    current_state = langgraph_app.get_state(config)
    if current_state.next and "web_search_node" in current_state.next:
        st.warning("⚠️ Local context is insufficient. Should I search the internet?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Web Search"):
                langgraph_app.update_state(config, {"web_search_approved": True})
                result = langgraph_app.invoke(None, config)
                st.session_state.messages.append({"role": "assistant", "content": result["generation"]})
                st.rerun()
        with c2:
            if st.button("❌ Deny (Use Context Only)"):
                langgraph_app.update_state(config, {"web_search_approved": False})
                result = langgraph_app.invoke(None, config)
                st.session_state.messages.append({"role": "assistant", "content": result["generation"]})
                st.rerun()

    # --- CHAT INPUT & VISION ---
    elif user_input := st.chat_input("Ask a question or upload a screenshot...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
        
        prompt_text = user_input.text
        final_prompt = prompt_text
        
        if user_input.files:
            # WINDOWS SAFE TEMP FILE HANDLING
            t_file = tempfile.NamedTemporaryFile(delete=False, suffix=user_input.files[0].name)
            t_path = t_file.name
            try:
                t_file.write(user_input.files[0].getbuffer())
                t_file.close() # Close handle so Vision tool can read it
                
                analysis = analyze_image_context(t_path)
                final_prompt += f"\n\n[Vision Context]: {analysis}"
            finally:
                if os.path.exists(t_path):
                    os.remove(t_path) # Remove only after handle is closed

        st.chat_message("user").markdown(prompt_text)
        st.session_state.messages.append({"role": "user", "content": prompt_text})

        with st.chat_message("assistant"):
            
            # --- THE CRITICAL FIX ---
            # You must explicitly pass the HumanMessage object so it enters the graph's message channel!
            graph_input = {
                "question": final_prompt,
                "messages": [HumanMessage(content=final_prompt)] 
            }
            
            # Invoke using the new dictionary
            result = langgraph_app.invoke(graph_input, config=config)
            # ------------------------
            
            # Check if graph is paused at an interrupt
            if langgraph_app.get_state(config).next:
                st.rerun()
            else:
                # Safely extract generation or default to a fallback
                response = result.get("generation", "Agent completed tasks.")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})