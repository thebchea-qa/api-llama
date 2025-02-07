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
    Your task is to create a description of a hypothetical image which is based on the following idea/keywords/instruction: {TypeofImage}\n\n
    Come up with a fully-formed idea based on that instruction, and write a description/caption which fully captures all visual aspects the image. 
    Depict a lively and cheerful character with expressive features, such as sparkling eyes and a warm smile. 
    Design a trendy or charming outfit that reflects their playful personality. 
    Include intricate details in their hairstyle, accessories, and surroundings to create a vibrant, engaging scene. 
    Your response should be crisp/lean/efficient, descriptive, and engaging.\n\n
    Start your response with \"Description: The image shows\" and then give a one-paragraph description which captures all visual details of the idea in a medium-length paragraph."
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
