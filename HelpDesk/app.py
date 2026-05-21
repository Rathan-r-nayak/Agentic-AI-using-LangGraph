import streamlit as st
import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import backend components
from HelpDesk.Utils.HistoryLoaders import load_chat_history
from Utils.DocumentProcessor import process_and_index_files
from main import app as langgraph_app
from Utils.VectorStore import add_documents_to_store, get_indexed_files
from Utils.Helpers import analyze_image_context

# Page Configuration
# from Utils.HistoryLoader import load_chat_history

# Page Configuration
st.set_page_config(page_title="Smart AI Triage - IT Support 🎧", layout="wide", page_icon="🤖")

# ==========================================
# 1. SESSION STATE & CONFIG
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "relay_demo_user_001" 

# Sync with LangGraph Checkpointer
config = {"configurable": {"thread_id": st.session_state.thread_id}}

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
if role == "🛡️ IT Admin":
    st.header("Knowledge Base Manager")
    st.write("Upload technical manuals or policy docs to the Azure-powered Vector Store.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader("Select PDFs or Text files", type=["pdf", "txt"], accept_multiple_files=True)
    with col2:
        category = st.selectbox("Assign Technical Category", ["Database", "Network", "Software", "Hardware"])

    if st.button("🚀 Index to Vector DB"):
        if uploaded_files:
            temp_paths = []
            for uploaded_file in uploaded_files:
                # Use delete=False and close the file explicitly for Windows compatibility
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_paths.append(tmp.name)
            
            with st.spinner("Processing and chunking files..."):
                success = process_and_index_files(temp_paths, category)
            
            # Clean up temp files after indexing
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            if success:
                st.success(f"Successfully indexed {len(uploaded_files)} files!")
            else:
                st.error("Something went wrong during indexing.")
                
    st.divider()
    st.subheader("📚 Current Knowledge Base")
    indexed_files = get_indexed_files()
    if indexed_files:
        with st.expander(f"View {len(indexed_files)} Indexed Documents"):
            for file_name in indexed_files:
                st.markdown(f"- 📄 `{file_name}`")

# ==========================================
# 4. USER INTERFACE (CHAT & VISION)
# ==========================================
else:
    st.header("Relay IT Helpdesk")
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
            result = langgraph_app.invoke({"question": final_prompt}, config=config)
            
            # Check if graph is paused at an interrupt
            if langgraph_app.get_state(config).next:
                st.rerun()
            else:
                response = result.get("generation", "Agent completed tasks.")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})