from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
import services.mongodb as mongodb
from services.auth import get_current_user
from PIL import Image
from io import BytesIO
from services.fashion_clip import generate_tags, category_labels
from services.image import store_blob, get_blob_url, DEFAULT_EXPIRY
from bson import ObjectId

router = APIRouter()

'''
done GET /wardrobe/available_categories -> returns list of available categories. Don't hardcode colors and adjectives. 
done GET /wardrobe/categories -> returns all categories for the user and a corresponding thumbnail image link
GET /wardrobe/category/{category} -> return all items in that category, and a corresponding thumbnail image   -Should collections and categories be saved under 'user' table?
GET /wardrobe/collections -> returns all collections for the user and a corresponding thumbnail image
GET /wardrobe/collection/{collection} -> return all items in that collection, and a corresponding thumbnail image
done GET /wardrobe/item/{id} -> return the clothing item details from the id

done POST /wardrobe/item -> new item into db. Automatically creates tags and returns them for editing.
PATCH /wardrobe/item/{id} -> modify item in db (name, type, color, description)
DELETE /wardrobe/item/{id} -> delete item in db

GET /wardrobe/search/{query} -> search function
'''


@router.post("/wardrobe/item")
async def create_item(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image.verify()  # Check if the file is an actual image
        image = Image.open(BytesIO(contents))  # Re-open to handle potential truncation issue
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )

    # If image is good, generate tags
    tags = generate_tags(image)
    # for now, just returns the binary of the image. Later on switch to returning a filepath.
    image_name = store_blob(contents, f"image/{image.format}")

    # Insert a document into the collection
    document = {
        "user_id": current_user['_id'],
        "name": "",
        "type": tags['category'][0],
        "color": tags['color'][0],
        "description": tags['description'][:3],
        "image_name": image_name
    }

    insert_result = mongodb.wardrobe.insert_one(document)
    print(f"Inserted document ID: {insert_result.inserted_id}")

    res = tags
    res['id'] = str(insert_result.inserted_id)

    return res


@router.get("/wardrobe/item/{item_id}")
async def get_item(item_id: str, current_user: dict = Depends(get_current_user)):
    try:
        query = {"_id": ObjectId(item_id), "user_id": current_user['_id']}
        res = mongodb.wardrobe.find_one(query)
        if res is not None:
            return res['type']  # Right now just returns the type, TODO: return all infos needed
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid wardrobe ID"
        )


@router.get("/wardrobe/categories")
async def get_categories(current_user: dict = Depends(get_current_user)):
    user_id = current_user['_id']
    res = {"categories": []}

    for category in category_labels:
        item = mongodb.wardrobe.find_one({"user_id": user_id, "type": category})
        if item == None:
            continue

        image_url = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
        res["categories"].append({"type": category, "url": image_url})

    return res


@router.get("/wardrobe/available_categories")
async def get_available_categories():
    return category_labels
