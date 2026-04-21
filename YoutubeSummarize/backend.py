# %%
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
from typing import TypedDict, Literal, List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import re
import requests
import uuid
import hashlib

# %%
load_dotenv()

# %%

def get_db_name():
    # db_name = "./chroma_vector_store_yt"
    db_name = "./chroma_vector_store_yt_hf"

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
    is_upload : bool
    documents: List[Document]
    video_id: str
    web_search_needed: bool
    search_query: str
    answer: str

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
        print(f"Error fetching title: {e}")
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
        print(f"Error fetching transcript: {e}")
        return None

def get_video_id_match(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?.*v=|embed\/|shorts\/|v\/)|youtu\.be\/)([\w-]{11})"
    match = re.search(pattern, url)
    return match

# %%
def upload_document(state: ChatState):
    print("--- 1. UPLOADIG THE DOCUMENTS ---")
    url = state["url"]
    match = get_video_id_match(url)
    if not match:
        raise ValueError("Could not find a valid YouTube Video ID in the provided URL.")
    video_id = match.group(1)
    print(f"   -> getting the transcript for video with id : {video_id}")

    transcript = fetch_youtube_transcript(url)
    title = fetch_youtube_title(url)

    print(f"   -> fetching of the data completed video_id : {video_id}, title: {title}")
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
        print("----------------------------------------------------------")
        return {"title": title, "video_id": video_id} # Update the state
    else:
        print(f"Warning: No transcript found for {url}")
        print("----------------------------------------------------------")
        return {"title": title, "video_id": video_id}

# %%
def retrive_document(state: ChatState):
    print("--- 2. RETRIVING THE DOCUMENTS FROM VECTOR DB ---")
    vector_db = Chroma(
        embedding_function = get_embeddings(),
        persist_directory = get_db_name()
    )
    retirver = vector_db.as_retriever()
    print(f"   -> retriving the documents with query : {state['query']}")
    documents = retirver.invoke(state['query'])
    print("----------------------------------------------------------")
    return {"documents": documents}

# %%
def grade_documents(state: ChatState):
    """Uses an LLM to grade if the retrieved documents are actually relevant."""
    print("--- 3. GRADING DOCUMENTS ---")
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

    doc_text = documents[0].page_content if documents else ""
    score = grader_chain.invoke({"question": question, "document": doc_text})

    print(f"   -> grade score: {score.binary_score}")
    print("----------------------------------------------------------")
    if score.binary_score == "yes":
        # print("   -> Grade: RELEVANT. Proceeding to generation.")
        return {"web_search_needed": False}
    else:
        # print("   -> Grade: IRRELEVANT. Fallback required.")
        return {"web_search_needed": True}

# %%
def rewrite_query(state: ChatState):
    """Rewrites the user's original question into an optimized web search query."""
    print("--- 4. REWRITING QUERY FOR WEB SEARCH ---")
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

    print(f"   -> Original: {query}")
    print(f"   -> Rewritten: {result.query}")
    print("----------------------------------------------------------")
    # Update the state with the new search query
    return {"search_query": result.query}

# %%
def web_search(state: ChatState):
    """Fallback tool using the optimized search query."""
    print("--- 5. PERFORMING WEB SEARCH ---")
    search_tool = DuckDuckGoSearchRun()
    # search_tool.invoke("apollo tyres share price")

    optimized_query = state['search_query']

    print(f"   -> Searching the web for: '{optimized_query}'")
    
    search_result = search_tool.invoke(optimized_query)
    new_doc = Document(page_content=search_result)
    print(f"   -> Web response: '{search_result}'")
    print("-----------------new_doc-----------------------------------------")
    return {"documents": [new_doc]}

# %%
def generate(state: ChatState):
    """Generates the final answer."""
    print("--- 6. GENERATING FINAL ANSWER ---")
    documents = state['documents']
    query = state['query']
    docs_text = "\n\n".join([doc.page_content for doc in documents])

    llm = get_llm()

    prompt = PromptTemplate.from_template(
        "Answer the question based strictly on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )

    chain = prompt | llm
    response = chain.invoke({"context": docs_text, "question": query})

    print(f"   -> response {response.content}")
    print("----------------------------------------------------------")
    return {"answer": response.content}

# %%
def route_after_grade(state: ChatState)-> Literal["rewrite_query", "generate"]:
    if state["web_search_needed"]:
        return "rewrite_query"
    else:
        return "generate"

# %%
def route(state: ChatState) -> Literal["upload", "retrive_document"]:
    if state.get("is_upload"):
        return "upload"
    
    # Otherwise, skip the upload and go straight to retrieving answers
    return "retrive_document"

# %%
graph = StateGraph(ChatState)

graph.add_node("upload", upload_document)
graph.add_node("retrive_document", retrive_document)
graph.add_node("grade_documents", grade_documents)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate)

graph.add_conditional_edges(START, route)
graph.add_edge("upload", END)
graph.add_edge("retrive_document", "grade_documents")
graph.add_conditional_edges("grade_documents", route_after_grade, {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    })

graph.add_edge("rewrite_query", "web_search")
graph.add_edge("web_search", "generate")

graph.add_edge("generate", END)

workflow = graph.compile()