from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import os


persist_db_name = "./chroma_vector_store"


load_dotenv()
# model = ChatGroq(model="llama-3.1-8b-instant")
model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
# model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
API_KEY = os.getenv("TMDB_API_KEY")

search_tool = DuckDuckGoSearchRun(region="is_en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def weather_details(city:str)->str:
    """This tool get the city name and returns the current temperature of a place in celcius"""
    try:
        city_details = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()
        latitude = city_details['results'][0]['latitude']
        longitude = city_details['results'][0]['longitude']
        weather_data = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true").json()
        return str(weather_data['current_weather']['temperature'])
    except Exception as e:
        return repr(e)

@tool
def movie_search(movie_name:str):
    """This tool iw going to get the movie name and returns the details about that movie"""
    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": API_KEY,
        "query": movie_name,
        "language": "en-US"
    }
    response = requests.get(url=url, params=params).json()

    description = ""
    release_date = ""
    for result in response["results"]:
        description = result.get("overview","")
        release_date = result.get("release_date","")
        original_language = result.get("original_language","")
        original_title = result.get("original_title","")
        break

    return {"description" : description, "original_title" : original_title, "release_date" : release_date, "language" : original_language}


@tool
def measure_aqi(city : str):
    """"This tools is going to give the Air quality index of the given city"""
    city_details = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}").json()

    longitude = city_details['results'][0]['longitude']
    latitude = city_details['results'][0]['latitude']
    
    aqi_details = requests.get(f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={latitude}&longitude={longitude}&current=us_aqi").json()
    aqi = aqi_details['current']['us_aqi']
    print(aqi)
    return aqi


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
@tool
def fetch_documents(query:str):
    """"This tools is going to fetch the data from the knowledgebase of the user"""
    try:
        vector_db = Chroma(
                embedding_function = embeddings,
                persist_directory = persist_db_name
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






# system_prompt = SystemMessage(
#         content=(
#             "You are a helpful and conversational AI assistant. "
#             "When you use a tool, you will receive a result. Use that result to answer "
#             "the user's question in a natural way. "
#             "Do not mention the internal function names. Just provide the information. "
#             "If you get a number for AQI, explain what it means (e.g., Good, Moderate)."
#         )
#     )

system_prompt = SystemMessage(
    content=(
        "You are a helpful and conversational AI assistant. "
        "When you use a tool, you will receive a result. Use that result to synthesize and write a clean, natural answer for the user. "
        "CRITICAL INSTRUCTION: NEVER copy-paste, repeat, or regurgitate the raw text or search snippets returned by the tools. "
        "Read the tool output silently, and only output your final, conversational answer in your own words. "
        "Do not mention the internal function names. If you get a number for AQI, explain what it means (e.g., Good, Moderate)."
    )
)


tools = [calculator,weather_details,movie_search,measure_aqi,search_tool, fetch_documents]
model_with_tools = model.bind_tools(tools)

# Think of ToolNode as the Execution Center.
tool_node = ToolNode(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def chat_node(state : ChatState):
    query = state["messages"]
    msg = [system_prompt] + query
    response = model_with_tools.invoke(msg)
    return {"messages" : [response]}

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
# This is a Decision Logic function.
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

check_pointer = InMemorySaver()
workflow = graph.compile(checkpointer=check_pointer)

