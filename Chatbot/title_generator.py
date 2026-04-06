from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


prompt = SystemMessage(
        content=(
            "You are a helpful AI who genearate clear title of the chat based on the prompt given. Dont exceed more than 50 characters"
        )
    )

class TitleState(TypedDict):
    question : str
    title : str


def get_title(state : TitleState):
    question = state['question']

    query = [prompt] + [HumanMessage(question)]
    llm = ChatGroq(model="llama-3.1-8b-instant")
    # llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
    response = llm.invoke(query)
    return {"title" : response}


graph = StateGraph(TitleState)

graph.add_node("title_node", get_title)

graph.add_edge(START, "title_node")
graph.add_edge("title_node", END)

title_workflow = graph.compile()

# init = {"question" : "Tell about Agentic AI"}
# print(workflow.invoke(init))