import streamlit as st
import uuid
import asyncio
from langchain_core.messages import AIMessage, HumanMessage

# Import your graph compiler from main.py
from main import compile_banking_agent

# 1. Page Configuration
st.set_page_config(page_title="Banking Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Secure Banking Agent")

# 2. Session State Initialization
# Manage the thread_id so the checkpointer knows which conversation to load
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Manage UI chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Compile the graph once per session
if "graph_app" not in st.session_state:
    # TODO: If you have your Postgres checkpointer ready, pass it here.
    # Example: 
    # from langgraph.checkpoint.postgres import PostgresSaver
    # checkpointer = PostgresSaver(...)
    # st.session_state.graph_app = compile_banking_agent(checkpointer=checkpointer)
    
    st.session_state.graph_app = compile_banking_agent(checkpointer=None)

# 3. Sidebar Controls
with st.sidebar:
    st.header("Session Info")
    st.caption(f"Thread ID: `{st.session_state.thread_id}`")
    
    # Button to start a completely new thread
    if st.button("Clear Chat / New Session"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# 4. Render Chat History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 5. Handle User Input
if prompt := st.chat_input("Ask about your accounts, transactions, or banking policies..."):
    
    # Display user input in UI and save to session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Configuration for the checkpointer mapping
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # State payload targeting the 'question' key in your BankingState
    input_state = {
        "question": prompt,
        "worker_responses": []
    }

    with st.chat_message("assistant"):
        with st.spinner("Processing request..."):
            try:
                # ✅ Changed back to asyncio.run and .ainvoke()
                result = asyncio.run(
                    st.session_state.graph_app.ainvoke(input_state, config=config)
                )
                
                # Extract the final synthesized answer
                final_answer = result.get("generation", "I'm sorry, I couldn't generate a response.")
                
                # Display and save the answer
                st.markdown(final_answer)
                st.session_state.messages.append(AIMessage(content=final_answer))
                
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")