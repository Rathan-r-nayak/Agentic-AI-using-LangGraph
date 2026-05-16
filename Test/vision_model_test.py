import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# 1. Helper function to read a local image file and encode it safely to Base64
def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image file and encodes it to a base64 string compatible with vision models."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    # Check suffix to format the exact MIME type (png, jpeg, webp)
    file_extension = image_path.split(".")[-1].lower()
    if file_extension == "png":
        mime_type = "image/png"
    elif file_extension in ["jpg", "jpeg"]:
        mime_type = "image/jpeg"
    else:
        mime_type = "image/webp"
        
    return f"data:{mime_type};base64,{encoded_string}"

def analyze_screenshot_via_openrouter(image_path: str, user_prompt: str) -> str:
    print(f"--- 👁️ PROCESSING IMAGE VIA MULTIMODAL VISION MODEL ---")
    
    if not os.path.exists(image_path):
        return f"Error: File not found at {image_path}"
        
    # Prepare the Base64 Data URL
    base64_image_url = encode_image_to_base64(image_path)
    
    # Initialize OpenRouter ChatOpenAI engine pulling a highly capable free vision tier
    vision_llm = ChatOpenAI(
        openai_api_key=API_KEY,
        base_url="https://openrouter.ai/api/v1",
        # model="google/gemma-4-31b-it:free",  # Target free multi-modal vision engine
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        temperature=0.0,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Relay AI Vision Hub"
        }
    )
    
    # 2. Structure the payload into standard text and image parts
    message_content = [
        {
            "type": "text", 
            "text": user_prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": base64_image_url  # Send the raw string data payload
            }
        }
    ]
    
    # Wrap it inside LangChain's HumanMessage container object
    messages = [HumanMessage(content=message_content)]
    
    try:
        response = vision_llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Vision invocation failed: {e}"

# ==========================================
# TEST EXECUTION TRACK
# ==========================================
if __name__ == "__main__":
    # Create a dummy image or path to an actual layout error screenshot
    TEST_IMAGE_PATH = "data/image-20.png" 
    PROMPT = "Analyze this image. Extract the details of this image and explain"
    
    if os.path.exists(TEST_IMAGE_PATH):
        analysis_report = analyze_screenshot_via_openrouter(TEST_IMAGE_PATH, PROMPT)
        print("\n--- 🚀 VISION COMPILATION OUTPUT ---")
        print(analysis_report)
    else:
        print(f"To test, please place a file named '{TEST_IMAGE_PATH}' in this workspace path.")