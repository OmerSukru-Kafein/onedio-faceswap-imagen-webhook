from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx
import uuid
import asyncio
import os
from face_swapper import FaceSwapProcessor

app = FastAPI(title="FaceSwap Processing API", 
              description="API for processing face swap requests asynchronously with webhook callbacks")

# Configuration - Use environment variables for Cloud Run
CALLBACK_API_URL = os.environ.get("CALLBACK_API_URL", "http://127.0.0.1:8000/callback")
PORT = int(os.environ.get("PORT", 8001))
HOST = os.environ.get("HOST", "127.0.0.1")

class FaceSwapRequest(BaseModel):
    """Request model for face swap operation"""
    source_image_base64: str
    target_image_base64: str
    callback_url: str | None = None  # Optional callback URL parameter

class FaceSwapResult(BaseModel):
    """Result model for face swap operation"""
    status: str  # "success", "failed", "processing"
    swapped_image_base64: str = ""
    request_id: str | None = None
    error: str | None = None

face_swapper_instance = FaceSwapProcessor()

async def process_and_callback(payload: FaceSwapRequest, request_id: str):
    """Process face swap and send results to callback URL"""
    print(f"Background task started for request ID: {request_id}")
    callback_url = payload.callback_url or CALLBACK_API_URL
    
    try:
        print(f"Starting face swap processing for request ID: {request_id}")
        
        # Simulate 10 second processing delay
        print(f"Simulating 10 second processing delay for request ID: {request_id}")
        await asyncio.sleep(10)
        print(f"Processing delay completed for request ID: {request_id}")
        
        # Perform face swap
        swapped_base64 = face_swapper_instance.face_swap_function(
            payload.source_image_base64, payload.target_image_base64
        )
        print(f"Face swap completed successfully for request ID: {request_id}")
        result = FaceSwapResult(
            status="success", 
            swapped_image_base64=swapped_base64, 
            request_id=request_id
        )
    except Exception as e:
        print(f"Face swap failed for request ID: {request_id}, error: {str(e)}")
        result = FaceSwapResult(
            status="failed", 
            swapped_image_base64="", 
            request_id=request_id, 
            error=str(e)
        )

    # Send result to callback URL
    print(f"Sending callback to {callback_url} for request ID: {request_id}")
    async with httpx.AsyncClient() as client:
        try:
            json_response = await client.post(callback_url, json=result.model_dump())
            print(f"Callback sent successfully to {callback_url}. Status: {json_response.status_code}")
            # Save response in responses directory for debugging
            #os.makedirs("responses", exist_ok=True)
            #with open(f"responses/callback_response_{request_id}.json", "w") as f:
            #    f.write(json_response.text)
        except Exception as callback_error:
            print(f"Failed to send callback to {callback_url}: {callback_error}, request_id: {request_id}")
            # Save error for debugging
            #os.makedirs("responses", exist_ok=True)
            #with open(f"responses/callback_error_{request_id}.json", "w") as f:
            #    f.write(str(callback_error))

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "faceswap-processing-api"}

@app.post("/process")
async def process_face_swap(payload: FaceSwapRequest, background_tasks: BackgroundTasks):
    """Endpoint for asynchronous face swap processing with webhook callback"""
    request_id = str(uuid.uuid4())
    
    print(f"Request received with ID: {request_id}")
    print("Adding background task...")
    background_tasks.add_task(process_and_callback, payload, request_id)
    
    
    print(f"Returning immediate response for request ID: {request_id}")
    return {
        "message": "Request received. Face swap processing will continue in background.", 
        "request_id": request_id
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting FaceSwap Processing API on {HOST}:{PORT}")
    print(f"Callback URL: {CALLBACK_API_URL}")
    uvicorn.run(app, host=HOST, port=PORT)