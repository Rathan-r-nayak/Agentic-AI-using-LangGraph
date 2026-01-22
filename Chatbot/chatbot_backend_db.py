from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()
model = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    message : Annotated[list[str], add_messages]

def chat_node(state : ChatState):
    query = state["message"]
    response = model.invoke(query)
    return {"message" : response}

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
check_pointer = SqliteSaver(conn=conn)

# check_pointer = InMemorySaver()
workflow = graph.compile(checkpointer=check_pointer)

