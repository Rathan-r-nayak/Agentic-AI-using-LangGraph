from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

persist_db_name = "./chroma_vector_store"
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")