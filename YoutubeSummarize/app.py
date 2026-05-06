import os
# Suppress heavy logging from libraries
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import warnings
# Broad suppression for these specific import-time path warnings
warnings.filterwarnings("ignore", message=".*Accessing `__path__` from .*")
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

import streamlit as st
from backend import get_uploaded_videos_from_chroma, workflow, load_chat_history  # Imports your compiled graph from backend.py
import logging
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- PAGE SETUP ---
st.set_page_config(page_title="Custom NotebookLM", page_icon="📚", layout="wide")
st.title("📚 Custom NotebookLM")

# --- SESSION STATE ---
# This keeps track of our chat history and uploaded videos so they don't disappear
if "messages" not in st.session_state:
    with st.spinner("Loading previous chats..."):
        st.session_state.messages = load_chat_history("my_first_conversation")
    
if "uploaded_videos" not in st.session_state:
    st.session_state.uploaded_videos = get_uploaded_videos_from_chroma()

# --- SIDEBAR: SOURCE UPLOADS ---
with st.sidebar:
    st.header("🗂️ Add Sources")
    st.write("Paste a YouTube link below to add it to your knowledge base.")
    
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/...")
    
    if st.button("Process Video", type="primary"):
        if url_input:
            logger.info(f"Processing video URL: {url_input}")
            with st.spinner("Extracting transcript & saving to ChromaDB..."):
                try:
                    config = {"configurable": {"thread_id": "my_first_conversation", "user_id": "rathan_001"}}

                    # Trigger the upload node in your LangGraph!
                    result = workflow.invoke({
                        "url": url_input, 
                        "is_upload": True, 
                        "query": ""
                    }, config=config)
                    
                    # Grab the title from your state return and save it to the UI
                    video_title = result.get("title", "Unknown Video")
                    
                    # Prevent duplicates in the UI list
                    if video_title not in st.session_state.uploaded_videos:
                        st.session_state.uploaded_videos.append(video_title)
                    
                    st.success(f"Successfully added: {video_title}")
                    logger.info(f"Successfully processed video: {video_title}")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    logger.error(f"Error processing video: {e}", exc_info=True)
        else:
            st.warning("Please enter a URL first.")

    st.divider()
    
    st.subheader("Your Knowledge Base:")
    if not st.session_state.uploaded_videos:
        st.info("No videos uploaded yet.")
    else:
        for video in st.session_state.uploaded_videos:
            st.write(f"✅ {video}")


# --- MAIN AREA: CHAT INTERFACE ---
st.write("Ask questions about the videos in your knowledge base!")

# Draw all past messages to the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The Chat Input box at the bottom of the screen
if prompt := st.chat_input("What is this video about?"):

    # Immediately show the user's question in the UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    logger.info(f"User asked: {prompt}")

    # Show a loading spinner while LangGraph does its retrieval and grading
    with st.chat_message("assistant"):
        with st.spinner("Searching transcripts and evaluating context..."):
            try:
                config = {"configurable": {"thread_id": "my_first_conversation"}}
                # Trigger the CRAG pipeline!
                # Notice we pass is_upload: False, so it skips the upload node
                user_msg = HumanMessage(content=prompt)
                result = workflow.invoke({
                    "url": "", 
                    "is_upload": False, 
                    "query": prompt,
                    "messages": [user_msg]
                }, config=config)

                answer = result["answer"]

                # Display the answer``
                st.markdown(answer)

                # If your graph decided to use DuckDuckGo, let the user know!
                if result.get("web_search_needed"):
                    st.caption("🔍 *This answer was supplemented with a live web search because the video didn't contain the answer.*")
                    logger.info("Answer supplemented with web search.")

                # Save the assistant's answer to the chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                logger.info("Assistant response generated successfully.")

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                logger.error(f"Error during generation: {e}", exc_info=True)