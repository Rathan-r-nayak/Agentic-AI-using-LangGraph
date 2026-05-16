import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import your specific Vector Store and Embedding model here
# from Utils.VectorStore import vector_store, embeddings

st.set_page_config(page_title="Relay AI - Admin Portal", layout="centered")

st.title("🛡️ Admin Knowledge Base Manager")
st.markdown("Use this portal to index technical manuals and incident logs into the Vector DB.")

# 1. File Uploader
uploaded_files = st.file_uploader(
    "Upload Technical Documentation (PDF/TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

if st.button("🚀 Index to Vector Store"):
    if not uploaded_files:
        st.error("Please upload at least one file.")
    else:
        all_docs = []
        with st.spinner("Processing and Chunking Documents..."):
            for uploaded_file in uploaded_files:
                # Save to a temporary file for the LangChain loaders to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    file_path = tmp.name

                # Load based on extension
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                # Split documents into 1000-character chunks with 100-character overlap
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = loader.load_and_split(text_splitter)
                
                # Add metadata (for the Category filtering we built earlier!)
                category = st.selectbox(f"Category for {uploaded_file.name}", ["Database", "Network", "Software", "Hardware"], key=uploaded_file.name)
                for chunk in chunks:
                    chunk.metadata["category"] = category
                    chunk.metadata["source"] = uploaded_file.name
                
                all_docs.extend(chunks)
                os.remove(file_path)

        # 2. Push to Vector Store
        try:
            # REPLACE the line below with your actual vector store logic
            # vector_store.add_documents(all_docs)
            
            st.success(f"Successfully indexed {len(all_docs)} chunks from {len(uploaded_files)} files!")
            st.balloons()
        except Exception as e:
            st.error(f"Failed to index: {e}")

st.info("Note: This portal is for internal IT admins only. Ensure all documents are cleared of PII before indexing.")