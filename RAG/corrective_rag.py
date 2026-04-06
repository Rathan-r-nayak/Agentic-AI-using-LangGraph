from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import Literal, TypedDict,List
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, START,END



load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
db_name = "./chroma_vector_store_crag"

vector_db = Chroma(
    embedding_function = embeddings,
    persist_directory = db_name
)
retirver = vector_db.as_retriever()

class CRAGState(TypedDict):
    query: str
    search_query: str
    documents: List[Document]
    answer: str
    web_search_needed: bool


def retrive(state: CRAGState):
    """Fetches documents from our local vector database."""
    print("--- 1. RETRIEVING LOCAL DOCUMENTS ---")
    query = state['query']
    documents = retirver.invoke(query)
    print("----------------------------------------------------------")
    return {"documents": documents, "search_query": query}



def grade_documents(state: CRAGState):
    """Uses an LLM to grade if the retrieved documents are actually relevant."""
    print("--- 2. GRADING DOCUMENTS ---")
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
    

def rewrite_query(state: CRAGState):
    """Rewrites the user's original question into an optimized web search query."""
    print("--- 3. REWRITING QUERY FOR WEB SEARCH ---")
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

def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant")
    # return ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')


def web_search(state: CRAGState):
    """Fallback tool using the optimized search query."""
    print("--- 4. PERFORMING WEB SEARCH ---")
    search_tool = DuckDuckGoSearchRun()
    # search_tool.invoke("apollo tyres share price")

    optimized_query = state['search_query']

    print(f"   -> Searching the web for: '{optimized_query}'")
    
    search_result = search_tool.invoke(optimized_query)
    new_doc = Document(page_content=search_result)
    print(f"   -> Web response: '{search_result}'")
    print("----------------------------------------------------------")
    return {"documents": [new_doc]}

def generate(state: CRAGState):
    """Generates the final answer."""
    print("--- 5. GENERATING FINAL ANSWER ---")
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


def route_after_grade(state: CRAGState)-> Literal["rewrite_query", "generate"]:
    if state["web_search_needed"]:
        return "rewrite_query"
    else:
        return "generate"



workflow = StateGraph(CRAGState)

workflow.add_node("retrive", retrive)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrive")
workflow.add_edge("retrive", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_after_grade, {
        "rewrite_query": "rewrite_query",
        "generate": "generate"
    })

workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("web_search", "generate")

workflow.add_edge("generate", END)

app = workflow.compile()




print("\n==================================")
print("Testing Bad Query (Triggers Rewrite & Search)")
print("==================================\n")

bad_question = "What is Docker"
result = app.invoke({"query": bad_question})
print(f"\nFinal Answer: {result['answer']}")