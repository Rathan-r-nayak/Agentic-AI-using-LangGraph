import streamlit as st
# from chatbot_backend import workflow
from chatbot_backend_tools import workflow
from title_generator import title_workflow
from langchain.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import uuid
import tempfile

persist_db_name = "./chroma_vector_store"


# st.title("Asterisk ✳️")
st.set_page_config(
    page_title="Asterisk",
    page_icon="✳️",
    layout="wide"
)

st.sidebar.title("Asterisk ✳️")
st.sidebar.header('My Conversations')


# -----------------------------------UTILITY-------------------------

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread_to_thread_list(thread_id):
    if thread_id not in st.session_state["thread_id_list"]:
        st.session_state["thread_id_list"].append(thread_id)

def open_new_chat():
    new_thread_id = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["current_thread"] = new_thread_id
    add_thread_to_thread_list(new_thread_id)

# ----------------------------session variables-------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id_list" not in st.session_state:
    st.session_state["thread_id_list"] = []

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = generate_thread_id()

if "thread_title" not in st.session_state:
    st.session_state["thread_title"] = {}


add_thread_to_thread_list(st.session_state["current_thread"])
# -------------------------------UI-------------------------------------

if st.sidebar.button('New Chat'):    
    if(len(st.session_state["message_history"]) == 0):
        st.session_state["thread_id_list"].pop()

    open_new_chat()

for thread_id in st.session_state["thread_id_list"][::-1]:
    # if st.sidebar.button(str(thread_id)):
    display_data = str(st.session_state["thread_title"].get(thread_id, thread_id))
    
    if st.sidebar.button(display_data[:30].strip() + "...", help=display_data, use_container_width=True):
        if(len(st.session_state["message_history"]) == 0):
            st.session_state["thread_id_list"].pop()
        
        print(str(thread_id))
        values = workflow.get_state(config={"configurable":{"thread_id": thread_id}}).values
        print(values)
        st.session_state["current_thread"] = thread_id
        if("messages" in values):
            messages = values["messages"]
            tmp_messages = []
            for msg in messages:
                role = ""
                if isinstance(msg, HumanMessage):
                    role = "human"
                else:
                    role = "assistant"
                tmp_messages.append({"role":role, "content" : msg.content})
            st.session_state["message_history"] = tmp_messages


def ingest_document_to_chroma(upload_file:str, thread_id:str):
    """Saves the uploaded file, chunks it, and adds it to the FAISS database."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    for doc in chunks:
        doc.metadata["thread_id"] = str(thread_id)
        doc.metadata["source"] = upload_file.name
    # metadatas = [{"source":upload_file.name, "thread_id": thread_id, "page": doc.metadata.get("page", 0)} for doc in chunks]

    Chroma.from_documents(
        documents= chunks,
        embedding= embeddings,
        persist_directory= persist_db_name
        # metadatas= metadatas
    )


uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        ingest_document_to_chroma(uploaded_file, st.session_state["current_thread"])
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = set()
        st.session_state["processed_files"].add(uploaded_file.name)
        st.sidebar.success("ducument added to your knowledge base")


# print(st.session_state["current_thread"])
# list of dictonary
# if("message_history" not in st.session_state):
#     st.session_state["message_history"] = []

for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Enter you prompt here")

if user_input:
    with(st.chat_message("user")):
        if(len(st.session_state["message_history"]) == 0):
            st.session_state["thread_title"][st.session_state["current_thread"]] = title_workflow.invoke({"question" : user_input})["title"].content

        st.session_state["message_history"].append({"role":"user", "content":user_input})
        st.text(user_input)

    config = {"configurable" : {"thread_id" : st.session_state["current_thread"]}}
    # response = workflow.invoke({"messages" : HumanMessage(user_input)}, config=config) 
    # llm_reply = response["messages"][-1].content

    with(st.chat_message("assistant")):
            # st.session_state[""]

        response = st.write_stream(
            chunk_msg.content for chunk_msg, meta_data in workflow.stream({"messages" : HumanMessage(user_input)}, config=config, stream_mode="messages")
        )

        st.session_state["message_history"].append({"role":"assistant", "content":response})
        # st.markdown(llm_reply)
