import os
import requests
import io
from PIL import Image
from dotenv import load_dotenv

# Load your HF token from the .env file
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    print("-> ⚠️ Error: HUGGINGFACEHUB_API_TOKEN missing from your environment variables.")
    exit(1)

# Targeting the lightning-fast FLUX.1 model
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def generate_production_image(prompt: str):
    print("--- 🎨 REQUESTING IMAGE VIA HUGGING FACE INFERENCE ---")
    
    payload = {
        "inputs": prompt,
        "parameters": {
            # Schnell is optimized to generate great images in very few steps (1-4)
            "num_inference_steps": 4 
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            print("--- 🚀 SUCCESS! SAVING IMAGE ---")
            # Hugging Face returns raw image bytes; we convert it and save it to disk
            image = Image.open(io.BytesIO(response.content))
            
            save_path = "generated_dashboard_asset.png"
            image.save(save_path)
            
            print(f"✅ Image successfully saved to: {save_path}")
            return save_path
            
        else:
            print(f"-> ⚠️ Generation Failed: {response.status_code}")
            try:
                print(response.json())
            except:
                print(response.text)
            return None
            
    except Exception as e:
        print(f"-> ⚠️ Network execution failed: {e}")
        return None

if __name__ == "__main__":
    test_prompt = "A high-fidelity modern dashboard icon for an enterprise IT helpdesk, clean vectors, isolated background, 3d render style"
    generate_production_image(test_prompt)