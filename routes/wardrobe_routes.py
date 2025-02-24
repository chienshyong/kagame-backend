from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
import services.mongodb as mongodb
from services.mongodb import UserItem
from services.user import get_current_user
from PIL import Image
from io import BytesIO
from services.image import store_blob, get_blob_url, DEFAULT_EXPIRY
from bson import ObjectId
from services.openai import generate_wardrobe_tags, category_labels, WardrobeTag, get_wardrobe_recommendation, clothing_tag_to_embedding, get_n_closest

router = APIRouter()

'''
POST /wardrobe/item -> new item into db. Automatically creates tags and returns them for editing.
GET /wardrobe/item/{id} -> return the clothing item details from the id
PATCH /wardrobe/item/{id} -> modify item in db (name, category, tags)
DELETE /wardrobe/item/{id} -> delete item in db

GET /wardrobe/categories -> returns all categories for the user and a corresponding thumbnail image link
GET /wardrobe/category/{category} -> return all items in that category, and a corresponding thumbnail image
GET /wardrobe/available_categories -> returns list of available categories. Don't hardcode colors and adjectives.

GET /wardrobe/search/{query} -> search function, returns all items which include the search term in name or tags
'''


@router.post("/wardrobe/item")
async def create_item(file: UploadFile = File(...), current_user: UserItem = Depends(get_current_user)):
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

    # Resize image
    new_width = 400
    new_height = image.size[1] * new_width // image.size[0]
    resized_image = image.resize((new_width, new_height))  # Downsize to reduce inference time (5s -> 2s)
    resized_image_arr = BytesIO()
    resized_image.save(resized_image_arr, format=image.format)

    # Upload image
    image_name = store_blob(resized_image_arr.getvalue(), f"image/{image.format}")
    image_url = get_blob_url(image_name, DEFAULT_EXPIRY)
    tags = generate_wardrobe_tags(image_url)

    # Insert a document into the collection
    document = {
        "user_id": current_user['_id'],
        "name": tags['name'],
        "category": tags['category'],
        "tags": tags['tags'],
        "image_name": image_name
    }
    insert_result = mongodb.wardrobe.insert_one(document)
    res = tags
    res['id'] = str(insert_result.inserted_id)

    return res


@router.get("/wardrobe/item/{_id}")
async def get_item(_id: str, current_user: UserItem = Depends(get_current_user)):
    query = {"_id": ObjectId(_id), "user_id": current_user['_id']}
    item = mongodb.wardrobe.find_one(query)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item_id specified"
        )

    res = {}
    res["image_url"] = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
    res["_id"] = str(item["_id"])
    for key in ["name", "category", "tags"]:
        res[key] = item[key]

    return res


@router.patch("/wardrobe/item/{_id}")
async def patch_item(_id: str, new_data: WardrobeTag, current_user: UserItem = Depends(get_current_user)):
    query = {"_id": ObjectId(_id), "user_id": current_user['_id']}
    new_data_query = {"$set": {"name": new_data.name, "category": new_data.category, "tags": new_data.tags}}
    result = mongodb.wardrobe.update_one(query, new_data_query)
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item_id specified"
        )
    else:
        return "Item updated successfully."


@router.delete("/wardrobe/item/{_id}")
async def delete_item(_id: str, current_user: UserItem = Depends(get_current_user)):
    query = {"_id": ObjectId(_id), "user_id": current_user['_id']}
    result = mongodb.wardrobe.delete_one(query)
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item_id specified"
        )
    else:
        return "Item deleted successfully."


@router.get("/wardrobe/categories")
async def get_categories(current_user: UserItem = Depends(get_current_user)):
    user_id = current_user['_id']
    res = {"categories": []}

    for category in category_labels:
        item = mongodb.wardrobe.find_one({"user_id": user_id, "category": category})
        if item == None:
            continue

        image_url = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
        res["categories"].append({"category": category, "url": image_url})

    return res


@router.get("/wardrobe/category/{category}")
async def get_categories(category: str, current_user: UserItem = Depends(get_current_user)):
    if category not in category_labels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid category specified"
        )

    user_id = current_user['_id']
    res = {"items": []}
    items = mongodb.wardrobe.find({"user_id": user_id, "category": category})
    for item in items:
        image_url = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
        res["items"].append({"_id": str(item["_id"]), "name": item["name"], "url": image_url})
    return res


@router.get("/wardrobe/available_categories")
async def get_available_categories():
    return category_labels


@router.get("/wardrobe/search/{search_term}")
async def function_name(search_term=str, current_user: UserItem = Depends(get_current_user)):
    # Find documents where 'name' contains the search term (case-insensitive)
    user_id = current_user['_id']
    items = mongodb.wardrobe.find({
        "user_id": user_id,
        "$or": [
            {"name": {"$regex": search_term, "$options": "i"}},
            {"tags": {"$elemMatch": {"$regex": search_term, "$options": "i"}}}
        ]
    })

    res = {"items": []}
    for item in items:
        image_url = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
        res["items"].append({"_id": str(item["_id"]), "name": item["name"], "url": image_url})

    return res


@router.get("/wardrobe/wardrobe_recommendation")
async def function_name(_id: str, additional_prompt: str = "", current_user: UserItem = Depends(get_current_user)):
    # Given the id of a user's wardrobe item, generate an outfit. You can add ptional constraints
    result = mongodb.wardrobe.find_one({"_id": ObjectId(_id), "user_id": current_user['_id']})

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item_id specified"
        )

    profile = await get_userdefined_profile(current_user)

    wardrobe_tag = WardrobeTag(name=result.get("name"), category=result.get("category"), tags=result.get("tags"))
    clothing_tags = get_wardrobe_recommendation(wardrobe_tag, profile, additional_prompt) #added user persona

    result = []
    for clothing_tag_embedded in map(clothing_tag_to_embedding, clothing_tags):
        rec = list(get_n_closest(clothing_tag_embedded, 1))[0]
        rec['_id'] = str(rec['_id'])
        result.append(rec)

    return result

@router.get("/wardrobe/userdefined_profile")
async def get_userdefined_profile(current_user: dict = Depends(get_current_user)):
    profile = current_user["userdefined_profile"]
    
    """return format: {
  "gender": "Female",
  "birthday": "09/09/2025",
  "location": "Singapore",
  "skin_tone": "Medium skin with neutral to warm undertones",
  "style": "Casual",
  "happiness_current_wardrobe": "7"
}"""
    return profile
