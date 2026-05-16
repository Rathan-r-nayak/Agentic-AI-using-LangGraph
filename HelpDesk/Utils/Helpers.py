import base64
import mimetypes
from langchain_core.messages import HumanMessage
from Utils.Logger import get_logger

# Import your primary model for vision tasks
from Config.LLMConfig import primary_llm 

logger = get_logger("HELPERS")

def format_chat_history(messages) -> str:
    """
    Converts a list of LangChain messages into a clean, readable string.
    This acts as the 'Short-Term Memory' for the Orchestrator and Workers.
    """
    if not messages:
        return "No previous conversation."
    
    # Grab the last 6 messages to keep the context window focused and cheap
    recent_msgs = messages[-6:]
    formatted = []
    
    for msg in recent_msgs:
        # Map LangChain message types to standard roles
        role = "User" if msg.type == "human" else "AI"
        formatted.append(f"{role}: {msg.content}")
        
    return "\n".join(formatted)

def analyze_image_context(image_path: str) -> str:
    """
    Uses the Vision LLM to analyze IT-specific images (errors, logs, diagrams).
    Extracts text, error codes, and system states to inject into the text prompt.
    """
    try:
        # 1. Dynamically detect mime type so PNGs and JPGs both work cleanly
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/jpeg"

        # 2. Encode the image to Base64 format
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 3. Build the multimodal message
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        "You are an L3 Technical Support Vision AI. Analyze this image. "
                        "1. If it's an error screen/terminal, extract the exact error codes and text. "
                        "2. If it's a software UI, describe the application state and any visible issues. "
                        "3. If it's a network/architecture diagram, list the connected components. "
                        "Focus entirely on actionable technical context. Do not describe aesthetic elements."
                    )
                },
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                }
            ]
        )
        
        logger.info("Analyzing attached image for technical context...")
        
        # 4. Invoke the model
        response = primary_llm.invoke([message])
        return response.content
        
    except Exception as e:
        logger.error(f"Vision Error: {e}")
        return "System failed to analyze the attached image."