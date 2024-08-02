from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
import kagameDB
from services.auth import get_current_user
from PIL import Image
from io import BytesIO
from services.fashion_clip import generate_tags
from services.image import store_image

router = APIRouter()

@router.post("/wardrobe/item")
async def create_item(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Check if the file is an actual image
        image = Image.open(BytesIO(contents))  # Re-open to handle potential truncation issue
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    #If image is good, generate tags
    tags = generate_tags(image)
    image_formatted_for_db = store_image(image) #for now, just returns the binary of the image. Later on switch to returning a filepath.

    # Insert a document into the collection
    document = {
        "user_id": current_user['_id'],
        "type": tags['category'][0],
        "color": tags['color'][0],
        "description": tags['description'][:3],
        "image_data": image_formatted_for_db
    }
    
    insert_result = kagameDB.wardrobe.insert_one(document)
    print(f"Inserted document ID: {insert_result.inserted_id}")
    
    return tags


'''
GET /wardrobe/available_categories -> returns list of available categories. Don't hardcode colors and adjectives. 
GET /wardrobe/categories -> returns all categories for the user and a corresponding thumbnail image link
GET /wardrobe/category/{category} -> return all items in that category, and a corresponding thumbnail image   -Should collections and categories be saved under 'user' table?
GET /wardrobe/collections -> returns all collections for the user and a corresponding thumbnail image
GET /wardrobe/collection/{collection} -> return all items in that collection, and a corresponding thumbnail image
GET /wardrobe/item/{id} -> return the clothing item details from the id

POST /wardrobe/item -> new item into db. Automatically creates tags and returns them for editing.
PUT /wardrobe/item/{id} -> modify item in db
DELETE /wardrobe/item/{id} -> delete item in db

GET /wardrobe/search/{query}
'''