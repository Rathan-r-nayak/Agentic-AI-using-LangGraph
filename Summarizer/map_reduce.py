from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# 1. Load and chunk the massive book
loader = PyPDFLoader("data/agenticAi.pdf")
pages = loader.load_and_split()

# 2. Initialize your LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")

# 3. Create the Map-Reduce Chain
# LangChain handles all the complex routing, looping, and parallel calls internally!
chain = load_summarize_chain(llm, chain_type="map_reduce")

# 4. Run it
final_summary = chain.invoke(pages)
print(final_summary["output_text"])