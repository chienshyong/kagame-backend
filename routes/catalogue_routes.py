from fastapi import HTTPException, APIRouter
from fastapi.responses import StreamingResponse
import services.kagameDB as kagameDB

router = APIRouter()

@router.get("/images")
def get_images():
    # Find all documents in the collection
    cursor = kagameDB.catalogue.find({}, {"_id": 1, "image_path": 1, "retailer": 1})

    response = []
    for document in cursor:
        response.append({'id': str(document['_id']), 'image_path': document['image_path'], 'retailer': document['retailer']})

    return response

@router.get("/images/{image_path}")
def get_image(image_path: str):
    image = kagameDB.get_image(image_path)

    if image != None:
        return StreamingResponse(image)
    else:
        raise HTTPException(status_code=404, detail="Image not found")