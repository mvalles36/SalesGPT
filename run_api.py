import json
import os
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from salesgpt.salesgptapi import SalesGPTAPI

# Load environment variables
load_dotenv()

# Access environment variables and ensure they are set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not configured")

AUTH_KEY = os.getenv("AUTH_KEY")
if AUTH_KEY is None:
    raise ValueError("AUTH_KEY not configured")

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://react-frontend:80",
    "https://sales-gpt-frontend-git-main-filip-odysseypartns-projects.vercel.app",
    "https://sales-gpt-frontend.vercel.app"
]
CORS_METHODS = ["GET", "POST"]

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=CORS_METHODS,
    allow_headers=["*"],
)

class AuthenticatedResponse(BaseModel):
    message: str

def get_auth_key(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer ") or authorization != f"Bearer {AUTH_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/")
async def say_hello() -> dict:
    return {"message": "Hello World"}

class MessageList(BaseModel):
    session_id: str
    human_say: str

# Store active sessions
sessions = {}

@app.get("/botname", response_model=None)
async def get_bot_name(authorization: Optional[str] = Header(None)):
    if os.getenv("ENVIRONMENT") == "production":
        get_auth_key(authorization)
        
    sales_api = SalesGPTAPI(
        config_path=os.getenv("CONFIG_PATH", "examples/example_agent_setup.json"),
        product_catalog=os.getenv("PRODUCT_CATALOG", "examples/sample_product_catalog.txt"),
        verbose=True,
        model_name=os.getenv("GPT_MODEL", "gpt-3.5-turbo-0613"),
    )
    name = sales_api.sales_agent.salesperson_name
    return {"name": name, "model": sales_api.sales_agent.model_name}

@app.post("/chat")
async def chat_with_sales_agent(req: MessageList, stream: bool = Query(False), authorization: Optional[str] = Header(None)):
    """
    Handles chat interactions with the sales agent.
    
    Args:
        req (MessageList): A request object containing the session ID and the message from the human user.
        stream (bool, optional): Flag to indicate if the response should be streamed. Defaults to False.

    Returns:
        StreamingResponse or response from sales agent.
    """
    if os.getenv("ENVIRONMENT") == "production":
        get_auth_key(authorization)

    sales_api = sessions.get(req.session_id)
    if sales_api is None:
        sales_api = SalesGPTAPI(
            config_path=os.getenv("CONFIG_PATH", "examples/example_agent_setup.json"),
            verbose=True,
            product_catalog=os.getenv("PRODUCT_CATALOG", "examples/sample_product_catalog.txt"),
            model_name=os.getenv("GPT_MODEL", "gpt-3.5-turbo-0613"),
            use_tools=os.getenv("USE_TOOLS_IN_API", "True").lower() in ["true", "1", "t"],
        )
        sessions[req.session_id] = sales_api

    if stream:
        async def stream_response():
            try:
                stream_gen = sales_api.do_stream(req.human_say)
                async for message in stream_gen:
                    yield json.dumps({"token": message}).encode("utf-8") + b"\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}).encode("utf-8") + b"\n"

        return StreamingResponse(stream_response(), media_type="application/json")
    else:
        try:
            response = await sales_api.do(req.human_say)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
