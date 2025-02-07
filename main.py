from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Securely fetch API key from environment variables
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise ValueError("Azure API key or endpoint is not set. Please configure it in the environment.")

# Initialize Azure client
client = ChatCompletionsClient(
    endpoint=AZURE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY),
)

# Request model
class ChatRequest(BaseModel):
    message: str

@app.post("/api/Llama")
async def chat(request: ChatRequest):
    TypeofImage = request.message
    
    if not TypeofImage:
        raise HTTPException(status_code=400, detail="Message is required")

    content = f"""
    Create a vivid and engaging description of a hypothetical image based on the following concept: {TypeofImage}.
    Depict a lively and charismatic character with expressive features, such as sparkling eyes and a warm, inviting smile. Their outfit should reflect a trendy, fashion-forward, or playful personality—consider bold color combinations, stylish layering, or unique accessories. Pay close attention to intricate details in their hairstyle, accessories, and overall aesthetic, ensuring a dynamic and visually striking presence.
    Set the scene in a vibrant and immersive environment—whether it's a neon-lit cityscape, a cozy aesthetic café, or a bustling festival—enhancing the energy and personality of the character. Focus on textures, lighting, and ambiance to create depth and visual appeal.
    Your response should be concise yet highly descriptive, capturing all essential visual elements in a single engaging paragraph. Start your response with 'Description: The image shows' and craft a medium-length paragraph that fully encapsulates the scene in a stylish and immersive way.
    """
    
    response = client.complete(
        messages=[
            SystemMessage(content=""),
            UserMessage(content=content),
        ],
        model="Llama-3.3-70B-Instruct",
        temperature=0.8,
        max_tokens=2048,
        top_p=0.1,
    )
    
    return {"response": response.choices[0].message.content}
