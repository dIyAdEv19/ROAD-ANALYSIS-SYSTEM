from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO  # Importing YOLOv8 class
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the YOLO model
model = YOLO("yolov8n.pt")

class DetectionResult(BaseModel):
    bounding_boxes: int

@app.post("/detect/", response_model=DetectionResult)
async def detect_objects(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    
    # Perform inference using the YOLO model
    results = model.predict(image)
    
    # Count bounding boxes (assuming one result per image)
    bounding_boxes = len(results[0].boxes) if results else 0
    
    # Return the count of bounding boxes
    return DetectionResult(bounding_boxes=bounding_boxes)

# Run the app using uvicorn
# Command: uvicorn app_name:app --reload
