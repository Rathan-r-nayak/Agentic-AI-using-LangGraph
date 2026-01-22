import streamlit as st
# from chatbot_backend import workflow
from chatbot_backend_tools import workflow
from langchain.messages import HumanMessage
import uuid

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


add_thread_to_thread_list(st.session_state["current_thread"])
# -------------------------------UI-------------------------------------

if st.sidebar.button('New Chat'):
    open_new_chat()

for thread_id in st.session_state["thread_id_list"][::-1]:
    if st.sidebar.button(str(thread_id)):
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
        st.session_state["message_history"].append({"role":"user", "content":user_input})
        st.text(user_input)

    config = {"configurable" : {"thread_id" : st.session_state["current_thread"]}}
    # response = workflow.invoke({"messages" : HumanMessage(user_input)}, config=config) 
    # llm_reply = response["messages"][-1].content

    with(st.chat_message("assistant")):
        response = st.write_stream(
            chunk_msg.content for chunk_msg, meta_data in workflow.stream({"messages" : HumanMessage(user_input)}, config=config, stream_mode="messages")
        )

        st.session_state["message_history"].append({"role":"assistant", "content":response})
        # st.markdown(llm_reply)