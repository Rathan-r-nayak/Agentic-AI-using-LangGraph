import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Utils.VectorStore import add_documents_to_store
from Utils.Logger import get_logger

logger = get_logger("DOCUMENT_PROCESSOR")

def process_and_index_files(file_paths: list, category: str):
    """
    A simple, standalone function to chunk and index files.
    Completely disconnected from LangGraph.
    """
    if not file_paths:
        return False
        
    all_chunks = []
    
    for path in file_paths:
        if not os.path.exists(path):
            continue
            
        # 1. Load the file
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        
        # 2. Chunk it
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = loader.load_and_split(splitter)
        
        # Clean up the weird temp filename (e.g., tmp123_manual.pdf -> manual.pdf)
        raw_name = os.path.basename(path)
        clean_name = raw_name.split("_", 1)[-1] if "_" in raw_name else raw_name
        
        # 3. Add required metadata for your Retriever
        for chunk in chunks:
            chunk.metadata["category"] = category
            chunk.metadata["application_name"] = "None" # Default required by your retriever
            chunk.metadata["source"] = clean_name
            
        all_chunks.extend(chunks)

    # 4. Save to Vector Store
    if all_chunks:
        success = add_documents_to_store(all_chunks)
        
        # 5. Cleanup the temporary files
        for path in file_paths:
            try: 
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Could not delete {path}: {e}")
                
        return success
        
    return False