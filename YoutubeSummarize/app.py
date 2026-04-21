import streamlit as st
from backend import workflow  # Imports your compiled graph from backend.py

# --- PAGE SETUP ---
st.set_page_config(page_title="Custom NotebookLM", page_icon="📚", layout="wide")
st.title("📚 Custom NotebookLM")

# --- SESSION STATE ---
# This keeps track of our chat history and uploaded videos so they don't disappear
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "uploaded_videos" not in st.session_state:
    st.session_state.uploaded_videos = []

# --- SIDEBAR: SOURCE UPLOADS ---
with st.sidebar:
    st.header("🗂️ Add Sources")
    st.write("Paste a YouTube link below to add it to your knowledge base.")
    
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/...")
    
    if st.button("Process Video", type="primary"):
        if url_input:
            with st.spinner("Extracting transcript & saving to ChromaDB..."):
                try:
                    # Trigger the upload node in your LangGraph!
                    result = workflow.invoke({
                        "url": url_input, 
                        "is_upload": True, 
                        "query": ""
                    })
                    
                    # Grab the title from your state return and save it to the UI
                    video_title = result.get("title", "Unknown Video")
                    
                    # Prevent duplicates in the UI list
                    if video_title not in st.session_state.uploaded_videos:
                        st.session_state.uploaded_videos.append(video_title)
                    
                    st.success(f"Successfully added: {video_title}")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
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

    # Show a loading spinner while LangGraph does its retrieval and grading
    with st.chat_message("assistant"):
        with st.spinner("Searching transcripts and evaluating context..."):
            try:
                # Trigger the CRAG pipeline!
                # Notice we pass is_upload: False, so it skips the upload node
                result = workflow.invoke({
                    "url": "", 
                    "is_upload": False, 
                    "query": prompt
                })
                
                answer = result["answer"]
                
                # Display the answer
                st.markdown(answer)
                
                # If your graph decided to use DuckDuckGo, let the user know!
                if result.get("web_search_needed"):
                    st.caption("🔍 *This answer was supplemented with a live web search because the video didn't contain the answer.*")
                
                # Save the assistant's answer to the chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")