# %%
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*Accessing `__path__` from .*")
warnings.simplefilter("ignore") # Ignore all warnings during import

import logging
# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START,END
from typing import TypedDict, Literal, List, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
import re
import requests
import uuid
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres"

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
)

checkpointer = PostgresSaver(pool)
checkpointer.setup() 

# %%
load_dotenv()

# %%

def get_db_name():
    # db_name = "./chroma_vector_store_yt"
    db_name = "./chroma_vector_store_yt_hf"
    return db_name

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# %%
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant")
    # return ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

# %%
class ChatState(TypedDict):
    query : str
    url : str
    title: str
    need_retrieval: bool
    is_upload : bool
    documents: List[Document]
    video_id: str
    web_search_needed: bool
    search_query: str
    answer: str
    messages: Annotated[list[BaseMessage], add_messages]

# %%
def fetch_youtube_title(url: str) -> str:
    """Fetches the title of a YouTube video using the official OEmbed API."""
    
    oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(oembed_url, headers=headers, timeout=5)
        
        response.raise_for_status() 
        
        data = response.json()
        return data.get("title", "Unknown Title")
        
    except Exception as e:
        logger.error(f"Error fetching title: {e}")
        return "Unknown Title"

# %%
def fetch_youtube_transcript(url: str) -> str:
    """Extracts the video ID from a URL and fetches the full text transcript."""
    
    match = get_video_id_match(url)
    
    if not match:
        raise ValueError("Could not find a valid YouTube Video ID in the provided URL.")
        
    video_id = match.group(1)
    
    try:

        api = YouTubeTranscriptApi()
        raw_transcript = api.fetch(video_id)
        
        # full_text = " ".join([segment['text'] for segment in raw_transcript])
        full_text = " ".join([segment.text for segment in raw_transcript])
        full_text = full_text.replace('\n', ' ')

        return full_text

    except Exception as e:
        logger.error(f"Error fetching transcript: {e}")
        return None

def get_video_id_match(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?.*v=|embed\/|shorts\/|v\/)|youtu\.be\/)([\w-]{11})"
    match = re.search(pattern, url)
    return match

# %%
def upload_document(state: ChatState):
    logger.info("--- 1. UPLOADING THE DOCUMENTS ---")
    url = state["url"]
    match = get_video_id_match(url)
    if not match:
        raise ValueError("Could not find a valid YouTube Video ID in the provided URL.")
    video_id = match.group(1)
    logger.info(f"   -> getting the transcript for video with id : {video_id}")

    transcript = fetch_youtube_transcript(url)
    title = fetch_youtube_title(url)

    logger.info(f"   -> fetching of the data completed video_id : {video_id}, title: {title}")
    if transcript:
        doc = Document(page_content=transcript, metadata={"source": url})
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents([doc])

        ids = []
        for index, chunk in enumerate(chunks):
            unique_string = f"{url}_chunk_{index}"
            
            chunk_id = hashlib.md5(unique_string.encode()).hexdigest()
            ids.append(chunk_id)
            
            # (Optional) Save the chunk number in metadata so the AI knows order
            chunk.metadata["chunk_number"] = index
            chunk.metadata["video_id"] = video_id
            chunk.metadata["title"] = title

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            ids=ids,
            persist_directory=get_db_name()
        )
        logger.info("--- DOCUMENT UPLOADING COMPLETED ---")
        return {"title": title, "video_id": video_id} # Update the state
    else:
        logger.warning(f"Warning: No transcript found for {url}")
        logger.info("--- DOCUMENT UPLOADING FAILED ---")
        return {"title": title, "video_id": video_id}
    
# %%



def decide_retrieval(state: ChatState):
    logger.info("--- DECIDING RETRIEVAL ---")
    class RetrieveDecision(BaseModel):
        need_retrieval: bool  = Field(description="true if external documents are needed to answer reliably, else false.")

    decide_retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You decide whether retrieval is needed.\n"
                "Return JSON that matches this schema:\n"
                "{{'should_retrieve': boolean}}\n\n"
                "Guidelines:\n"
                "- should_retrieve=true if answering requires specific facts, citations, or info likely not in the model.\n"
                "- should_retrieve=false for general explanations, definitions, or reasoning that doesn't need sources.\n"
                "- If unsure, choose true."
            ),
            ("human", "Question: {question}"),
        ]
    )

    llm = get_llm().with_structured_output(RetrieveDecision)
    decision: RetrieveDecision = llm.invoke(decide_retrieval_prompt.format_messages(question=state["query"]))

    logger.info(f"   -> Retrieval needed: {decision.need_retrieval}")
    return {"need_retrieval": decision.need_retrieval}



# %%
def retrive_document(state: ChatState):
    logger.info("--- 2. RETRIVING THE DOCUMENTS FROM VECTOR DB ---")
    vector_db = Chroma(
        embedding_function = get_embeddings(),
        persist_directory = get_db_name()
    )
    retirver = vector_db.as_retriever()
    logger.info(f"   -> retriving the documents with query : {state['query']}")
    documents = retirver.invoke(state['query'])
    logger.info("--- RETRIVAL COMPLETED ---")
    return {"documents": documents}

# %%
def grade_documents(state: ChatState):
    """Uses an LLM to grade if the retrieved documents are actually relevant."""
    logger.info("--- 3. GRADING DOCUMENTS ---")
    question = state["query"]
    documents = state["documents"]

    class Grade(BaseModel):
        binary_score: Literal["yes","no"] = Field(description="Relevance score 'yes' or 'no'")

    llm = get_llm()
    structured_llm = llm.with_structured_output(Grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Return 'yes' if relevant, or 'no' if irrelevant.
        
        Retrieved document: \n\n {document} \n\n
        User question: {question}""",
        input_variables=["document", "question"],
    )
    
    grader_chain = prompt | structured_llm

    doc_text = []
    for doc in documents:
        doc_text.append(doc.page_content if doc else "")
    # doc_text = documents[0].page_content if documents else ""
    score = grader_chain.invoke({"question": question, "document": "\n".join(doc_text)})

    logger.info(f"   -> grade score: {score.binary_score}")
    logger.info("--- GRADING COMPLETED ---")
    if score.binary_score == "yes":
        # logger.info("   -> Grade: RELEVANT. Proceeding to generation.")
        return {"web_search_needed": False}
    else:
        # logger.info("   -> Grade: IRRELEVANT. Fallback required.")
        return {"web_search_needed": True}

# %%
def rewrite_query(state: ChatState):
    """Rewrites the user's original question into an optimized web search query."""
    logger.info("--- 4. REWRITING QUERY FOR WEB SEARCH ---")
    query = state["query"]


    llm = get_llm()

    class RewrittenQuery(BaseModel):
        query: str = Field(description="The optimized search engine query")

    structred_llm = llm.with_structured_output(RewrittenQuery)

    prompt = PromptTemplate(
        template="""You are an expert at optimizing search queries. 
        Look at the initial question and formulate an optimized query that a search engine like Google would understand better.
        Break down complex questions into core keywords.
        
        Initial question: {query}""",
        input_variables=["query"],
    )

    rewrite_chain = prompt | structred_llm
    result = rewrite_chain.invoke({"query": query})

    logger.info(f"   -> Original: {query}")
    logger.info(f"   -> Rewritten: {result.query}")
    logger.info("--- QUERY REWRITING COMPLETED ---")
    # Update the state with the new search query
    return {"search_query": result.query}

# %%
def web_search(state: ChatState):
    """Fallback tool using the optimized search query."""
    logger.info("--- 5. PERFORMING WEB SEARCH ---")
    search_tool = DuckDuckGoSearchRun()
    # search_tool.invoke("apollo tyres share price")

    optimized_query = state['search_query']

    logger.info(f"   -> Searching the web for: '{optimized_query}'")
    
    search_result = search_tool.invoke(optimized_query)
    new_doc = Document(page_content=search_result)
    logger.info(f"   -> Web response length: {len(search_result)}")
    logger.info("--- WEB SEARCH COMPLETED ---")
    return {"documents": [new_doc]}

# %%
from langchain_core.prompts import MessagesPlaceholder

def generate(state: ChatState):
    """Generates the final answer."""
    logger.info("--- 6. GENERATING FINAL ANSWER ---")
    documents = state['documents']
    query = state['query']
    docs_text = "\n\n".join([doc.page_content for doc in documents])
    chat_history = state.get("messages", [])

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based strictly on the provided context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": docs_text,
        "question": query,
        "chat_history": chat_history
    })

    # prompt = PromptTemplate.from_template(
    #     "Answer the question based strictly on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    # )

    # response = chain.invoke({"context": docs_text, "question": query})

    logger.info(f"   -> response generated (length: {len(response.content)})")
    logger.info("--- GENERATION COMPLETED ---")
    return {"answer": response.content, "messages": [response]}


# %%


def generate_direct(state: ChatState):
    logger.info("--- GENERATING DIRECT ANSWER ---")
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the conversation history to stay in context.Answer using only your general knowledge.If it requires specific company info, say: 'I don't know based on my general knowledge.'"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])


    # direct_generation_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "Answer using only your general knowledge.\n"
    #             "If it requires specific company info, say:\n"
    #             "'I don't know based on my general knowledge.'"
    #         ),
    #         ("human", "{question}"),
    #     ]
    # )

    out = get_llm().invoke(prompt.format_messages(question=state["query"], chat_history=messages))
    logger.info("--- DIRECT GENERATION COMPLETED ---")
    return {"answer": out.content}


# %%
def route_after_grade(state: ChatState)-> Literal["rewrite_query", "generate"]:
    if state["web_search_needed"]:
        return "rewrite_query"
    else:
        return "generate"

# %%
def route(state: ChatState) -> Literal["upload", "decide_retrieval"]:
    if state.get("is_upload"):
        return "upload"
    
    # Otherwise, skip the upload and go straight to retrieving answers
    return "decide_retrieval"


# %%


def route_for_retrive(state: ChatState) -> Literal["retrive_document", "generate_direct"]:
    if state.get("need_retrieval"):
        return "retrive_document"
    
    return "generate_direct"
# %%

def get_uploaded_videos_from_chroma() -> list[str]:
    """Retrieves a list of unique video titles directly from ChromaDB."""
    try:
        # Connect to your existing database
        vector_db = Chroma(
            embedding_function=get_embeddings(),
            persist_directory=get_db_name()
        )

        db_data = vector_db.get(include = ["metadatas"])
        all_metadatas = db_data.get("metadatas", [])

        unique_titles = set()
        for metadata in all_metadatas:
            if metadata and "title" in metadata:
                unique_titles.add(metadata["title"])
                logger.info(f"Found video in DB: {metadata['title']}")
                
        # Return as a standard list
        return list(unique_titles)
        
    except Exception as e:
        logger.error(f"Error reading from ChromaDB: {e}")
        return []


# %%

graph = StateGraph(ChatState)

graph.add_node("upload", upload_document)
graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("retrive_document", retrive_document)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate)
graph.add_node("generate_direct", generate_direct)

graph.add_conditional_edges(START, route)
graph.add_edge("upload", END)


graph.add_conditional_edges("decide_retrieval", route_for_retrive)
graph.add_edge("generate_direct", END)
graph.add_edge("retrive_document", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grade, {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    })

graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "generate")

graph.add_edge("generate", END)

workflow = graph.compile(checkpointer=checkpointer)




def load_chat_history(thread_id : str):
    """Fetches past conversations from the LangGraph Postgres checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}

    try:
        history = list(workflow.get_state_history(config))
        history.reverse()
        
        chat_messages = []
        for snapshot in history:
            # LangGraph saves a snapshot after EVERY node. 
            # We only want the final state of each complete run (when 'next' is empty).
            if not snapshot.next:
                val = snapshot.values
                
                # We don't want to show video uploads as chat messages
                if val.get("is_upload"):
                    continue
                    
                q = val.get("query")
                a = val.get("answer")
                
                # If both exist, it was a completed Q&A turn
                if q and a:
                    chat_messages.append({"role": "user", "content": q})
                    chat_messages.append({"role": "assistant", "content": a})
        
        return chat_messages
        
    except Exception as e:
        logger.error(f"Failed to load chat history: {e}")
        return []