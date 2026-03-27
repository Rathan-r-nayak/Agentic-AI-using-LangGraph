from fastmcp import FastMCP
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma # Use this instead
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid
from dotenv import load_dotenv


load_dotenv()

mcp = FastMCP("RAG on MCP")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

db_name = "./chroma_vector_store"


@mcp.tool
def ingest_document(file_path:str):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        metadatas = [{"source": file_path, "page": doc.metadata.get("page", 0)} for doc in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]


        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=db_name
        )
        return f"Success: Ingested {len(chunks)} chunks from {file_path} into the vector database."
    except Exception as e:
        return f"Error ingesting file: {str(e)}"



@mcp.tool
def fetch_documents(query:str):
    try:
        vector_db = Chroma(
            embedding_function = embeddings,
            persist_directory = db_name
        )
        results = vector_db.similarity_search(query=query, k=5)

        if not results:
            return "No relevant documents found in the database."
        
        formatted_results = "Here are the top matching documents from the database:\n\n"
        for i, doc in enumerate(results):
            # Extract metadata if available (PyPDFLoader usually adds 'source' and 'page')
            source = doc.metadata.get("source", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
                
            formatted_results += f"--- Match {i+1} (Source: {source}, Page: {page}) ---\n"
            formatted_results += f"{doc.page_content}\n\n"
                
        return formatted_results
        
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"



fetch_documents("what is agentic ai")


if __name__ == "__main__":
    mcp.run()