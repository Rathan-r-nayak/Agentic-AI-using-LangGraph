import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from State.HelpDeskState import HelpDeskState
from Utils.VectorStore import add_documents_to_store
from Utils.Logger import get_logger

logger = get_logger("INDEXER")

def indexer_node(state: HelpDeskState):
    """
    Processes uploaded files, chunks them, and adds specific metadata 
    to ensure the RetrieveNode can filter them later.
    """
    logger.info("Indexing documents to vector store")
    
    upload_paths = state.get("upload_paths", [])
    category = state.get("category", "General")
    
    # FIX 1: Provide a fallback app_name. 
    # Even if the UI doesn't send one, we MUST tag it as "None".
    # Otherwise, the RetrieveNode filters will fail to find these chunks.
    app_name = state.get("application_name") or "None"
    
    if not upload_paths:
        logger.info("No documents provided for indexing.")
        # FIX 2: Only clear upload_paths. Do not return an empty "documents" list here.
        return {"upload_paths": []}

    all_chunks = []
    
    for path in upload_paths:
        if not os.path.exists(path):
            continue
            
        logger.info(f"Processing: {os.path.basename(path)}")
        
        # Load based on file type
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        
        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = loader.load_and_split(splitter)
        
        raw_name = os.path.basename(path)
        clean_name = raw_name.split("_", 1)[-1] if "_" in raw_name else raw_name
        
        # Tagging Metadata
        for chunk in chunks:
            chunk.metadata["category"] = category
            chunk.metadata["application_name"] = app_name  # Restored!
            chunk.metadata["source"] = clean_name          # Uses the cleaned name
            
        all_chunks.extend(chunks)

    # Save to ChromaDB via our VectorStore utility
    if all_chunks:
        success = add_documents_to_store(all_chunks)
        if success:
            logger.info(f"Successfully indexed {len(all_chunks)} chunks.")
            
    # FIX 4: Safer cleanup. Windows sometimes locks files, this prevents a crash.
    for path in upload_paths:
        try: 
            if os.path.exists(path):
                os.remove(path)
        except Exception as e: 
            logger.warning(f"Could not delete temp file {path}. It may be locked by the OS.")
    
    return {"upload_paths": []}