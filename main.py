from fastapi import FastAPI
from fastapi.responses import FileResponse
import kagameDB
import json
import os

#Run with: uvicorn main:app --reload
app = FastAPI()
IMAGE_DIR = './images'

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/images")
def read_root():
    # Find all documents in the collection
    cursor = kagameDB.catalogue.find()

    response = []
    for document in cursor:
        response.append({'id': str(document['_id']), 'image_path': document['image_path'], 'retailer': document['retailer']})

    return response

@app.get("/images/{image_name}")
def get_image(image_name: str):
    # Construct the full image path
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    # Check if the image exists
    if not os.path.isfile(image_path):
        return {"error": "Image not found"}
    
    # Return the image file
    return FileResponse(image_path)