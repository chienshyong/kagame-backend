from fastapi import HTTPException, APIRouter, Request
from fastapi.responses import StreamingResponse
import services.mongodb as mongodb
from typing import List
import io
from PIL import Image
import mimetypes

router = APIRouter()

@router.get("/images")
def get_images():
    # Find all documents in the collection
    cursor = mongodb.catalogue.find({}, {"_id": 1, "image_path": 1, "retailer": 1})

    response = []
    for document in cursor:
        response.append({
            'id': str(document['_id']),
            'image_path': document.get('image_path', ''),
            'retailer': document.get('retailer', '')
        })

    return response

@router.get("/images/{image_path}")
def get_image(image_path: str):
    # Find the document with the given image_path
    document = mongodb.catalogue.find_one(
        {"image_path": image_path},
        {"image_data": 1, "image_path": 1}
    )

    if document and 'image_data' in document:
        # Get the image binary data
        image_binary = document['image_data']

        # Convert BSON binary to bytes
        image_bytes = bytes(image_binary)

        # Determine the MIME type
        mime_type, _ = mimetypes.guess_type(document.get('image_path', ''))
        if not mime_type:
            # If MIME type couldn't be guessed, use a default
            mime_type = 'application/octet-stream'

        # Serve the image as a streaming response
        return StreamingResponse(io.BytesIO(image_bytes), media_type=mime_type)
    else:
        raise HTTPException(status_code=404, detail="Image not found")
        
@router.get("/shop/items")
def get_items_by_retailer(retailer: str, include_embeddings: bool = False, limit: int = 0):
    try:
        # Define the filter to search by retailer
        filter_criteria = {"retailer": retailer}

        # Adjust the projection to include embeddings only if requested
        projection = {
            "name": 1,
            "category": 1,
            "price": 1,
            "image_url": 1,
            "product_link": 1
        }
        if include_embeddings:
            projection["embedding"] = 1

        # Query to find items by retailer with optional limit
        items_cursor = mongodb.catalogue.find(filter_criteria, projection)
        if limit > 0:
            items_cursor = items_cursor.limit(limit)

        # Prepare the response list
        response = []
        for item in items_cursor:
            # Construct the item data structure
            item_data = {
                "id": str(item["_id"]),  # Convert ObjectId to string
                "name": item.get("name", ""),
                "category": item.get("category", ""),
                "price": item.get("price", ""),
                "image_url": item.get("image_url", ""),
                "product_link": item.get("product_link", "")
            }
            # Include embeddings if requested
            if include_embeddings and "embedding" in item:
                item_data["embedding"] = item["embedding"]

            response.append(item_data)

        # Return the list of items for the specified retailer
        return response

    except Exception as e:
        # Handle any errors that occur
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching items: {str(e)}"
        )
