from fastapi import FastAPI
from pydantic import BaseModel
from face_swapper import FaceSwapProcessor
import uvicorn
import json

class FaceSwapResult(BaseModel):
    """Result model for face swap operation"""
    status: str  # "success", "failed", "processing"
    swapped_image_base64: str = ""
    request_id: str | None = None
    error: str | None = None

class FaceSwapRequest(BaseModel):
    """Request model for face swap operation"""
    source_image_base64: str
    target_image_base64: str

app = FastAPI(title="FaceSwap API", 
              description="API for face swapping operations with direct and callback endpoints")

face_swapper_instance = FaceSwapProcessor()

@app.post("/callback")
async def receive_callback(result: FaceSwapResult):
    """Endpoint that receives callbacks from the processing API"""
    print(f"Callback received with status: {result.status}, request_id: {result.request_id}")
    
    if result.status == "success" and result.swapped_image_base64:
        print("Face swap completed successfully")
        # Save the response JSON to a file
        response_data = {
            "message": "Callback received",
            "status": result.status,
            "swapped_image_base64": result.swapped_image_base64,
            "request_id": result.request_id
        }
        #with open("face_swap_callback_response.json", "w") as f:
        #    json.dump(response_data, f, indent=4)
        return response_data
    elif result.status == "failed":
        print(f"Error: {result.error}")
        return {
            "message": "Callback received", 
            "status": result.status, 
            "error": result.error, 
            "request_id": result.request_id
        }
    else:
        return {
            "message": "Callback received", 
            "status": result.status, 
            "request_id": result.request_id
        }

@app.post("/direct")
async def direct_face_swap(request: FaceSwapRequest):
    """Endpoint for direct/synchronous face swap operations"""
    try:
        # Use the face swapper directly
        swapped_base64 = face_swapper_instance.face_swap_function(
            request.source_image_base64, request.target_image_base64
        )
        return {
            "status": "success", 
            "swapped_image_base64": swapped_base64,
            "message": "Face swap completed successfully"
        }
    except Exception as e:
        return {
            "status": "failed", 
            "error": str(e),
            "message": "Face swap failed"
        }

# For backwards compatibility
@app.post("/face-swap")
async def face_swap_compat(request: FaceSwapRequest):
    """Backwards compatibility endpoint for direct face swap"""
    return await direct_face_swap(request)

# Keep the test endpoint for backwards compatibility
@app.post("/test")
async def test_compat(result: FaceSwapResult):
    """Backwards compatibility for original /test endpoint"""
    return await receive_callback(result)

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="127.0.0.1")
