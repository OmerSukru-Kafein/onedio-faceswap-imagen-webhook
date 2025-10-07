from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import httpx
import uuid
import asyncio
import os
from face_swapper import FaceSwapProcessor
from imagen import ImagenGenerator

app = FastAPI(title="Test Image Generation and Face Swap API", 
              description="API for generating test images with Imagen and swapping faces asynchronously with webhook callbacks")

# Configuration - Use environment variables for Cloud Run
CALLBACK_API_URL = os.environ.get("CALLBACK_API_URL", "http://127.0.0.1:8000/callback")
PORT = int(os.environ.get("PORT", 8001))
HOST = os.environ.get("HOST", "127.0.0.1")

class TestImageRequest(BaseModel):
    """Request model for test image generation and face swap operation"""
    # Test details
    test_sonucu: str
    test_adı: str
    test_aciklamasi: str
    gender: str
    age: int
    image_place: str = "The place must be relevant to the test"
    image_style: str = "The theme must be relevant to the test"
    
    # Face image to swap
    source_face_image_base64: str
    
    # Optional callback URL
    callback_url: str | None = None

class FaceSwapRequest(BaseModel):
    """Request model for face swap operation"""
    source_image_base64: str
    target_image_base64: str
    callback_url: str | None = None  # Optional callback URL parameter

class TestImageResult(BaseModel):
    """Result model for test image generation and face swap operation"""
    status: str  # "success", "failed", "processing"
    generated_image_base64: str = ""
    swapped_image_base64: str = ""
    request_id: str | None = None
    error: str | None = None

class FaceSwapResult(BaseModel):
    """Result model for face swap operation"""
    status: str  # "success", "failed", "processing"
    swapped_image_base64: str = ""
    request_id: str | None = None
    error: str | None = None

# Initialize processors
face_swapper_instance = FaceSwapProcessor()
imagen_generator_instance = ImagenGenerator()

async def process_test_image_and_callback(payload: TestImageRequest, request_id: str):
    """Process test image generation, face swap, and send results to callback URL"""
    print(f"Background task started for test image request ID: {request_id}")
    callback_url = payload.callback_url or CALLBACK_API_URL
    
    try:
        print(f"Starting image generation for request ID: {request_id}")
        
        # Step 1: Generate image using Imagen based on test data
        generated_image_base64 = imagen_generator_instance.generate_image_from_test(
            test_sonucu=payload.test_sonucu,
            test_adı=payload.test_adı,
            test_aciklamasi=payload.test_aciklamasi,
            gender=payload.gender,
            age=payload.age,
            image_place=payload.image_place,
            image_style=payload.image_style
        )
        print(f"Image generated successfully for request ID: {request_id}")
        
        # Step 2: Perform face swap - swap source face onto generated image
        print(f"Starting face swap for request ID: {request_id}")
        swapped_image_base64 = face_swapper_instance.face_swap_function(
            source_image_base64=payload.source_face_image_base64,
            target_image_base64=generated_image_base64
        )
        print(f"Face swap completed successfully for request ID: {request_id}")
        
        result = TestImageResult(
            status="success",
            generated_image_base64=generated_image_base64,
            swapped_image_base64=swapped_image_base64,
            request_id=request_id
        )
    except Exception as e:
        print(f"Test image processing failed for request ID: {request_id}, error: {str(e)}")
        result = TestImageResult(
            status="failed",
            generated_image_base64="",
            swapped_image_base64="",
            request_id=request_id,
            error=str(e)
        )

    # Send result to callback URL
    print(f"Sending callback to {callback_url} for request ID: {request_id}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            json_response = await client.post(callback_url, json=result.model_dump())
            print(f"Callback sent successfully to {callback_url}. Status: {json_response.status_code}")
        except Exception as callback_error:
            print(f"Failed to send callback to {callback_url}: {callback_error}, request_id: {request_id}")

async def process_face_swap_and_callback(payload: FaceSwapRequest, request_id: str):
    """Process face swap only and send results to callback URL"""
    print(f"Background task started for face swap request ID: {request_id}")
    callback_url = payload.callback_url or CALLBACK_API_URL
    
    try:
        print(f"Starting face swap processing for request ID: {request_id}")
        
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
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            json_response = await client.post(callback_url, json=result.model_dump())
            print(f"Callback sent successfully to {callback_url}. Status: {json_response.status_code}")
        except Exception as callback_error:
            print(f"Failed to send callback to {callback_url}: {callback_error}, request_id: {request_id}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "service": "test-image-faceswap-api"}

@app.post("/process-test-image")
async def process_test_image(payload: TestImageRequest, background_tasks: BackgroundTasks):
    """
    Endpoint for asynchronous test image generation and face swap with webhook callback.
    
    This endpoint:
    1. Generates an image using Imagen based on test details
    2. Swaps the provided face onto the generated image
    3. Sends results to the callback URL
    """
    request_id = str(uuid.uuid4())
    
    print(f"Test image request received with ID: {request_id}")
    print("Adding background task for test image processing...")
    background_tasks.add_task(process_test_image_and_callback, payload, request_id)
    
    print(f"Returning immediate response for request ID: {request_id}")
    return {
        "message": "Request received. Test image generation and face swap will continue in background.", 
        "request_id": request_id
    }

@app.post("/process-face-swap")
async def process_face_swap_endpoint(payload: FaceSwapRequest, background_tasks: BackgroundTasks):
    """
    Endpoint for asynchronous face swap processing only with webhook callback.
    
    This endpoint performs face swap between two provided images.
    """
    request_id = str(uuid.uuid4())
    
    print(f"Face swap request received with ID: {request_id}")
    print("Adding background task for face swap...")
    background_tasks.add_task(process_face_swap_and_callback, payload, request_id)
    
    print(f"Returning immediate response for request ID: {request_id}")
    return {
        "message": "Request received. Face swap processing will continue in background.", 
        "request_id": request_id
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Test Image and Face Swap API on {HOST}:{PORT}")
    print(f"Callback URL: {CALLBACK_API_URL}")
    uvicorn.run(app, host=HOST, port=PORT)
