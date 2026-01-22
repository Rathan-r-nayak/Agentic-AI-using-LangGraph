from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# model = OllamaLLM(model="llama3")
model = ChatGroq(model="llama-3.1-8b-instant")

prompt = PromptTemplate(
    template = "Generate 2 interesting thing about {topic}",
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic":"switzerland"})

print(result)

chain.get_graph().print_ascii()