from fastapi import HTTPException, APIRouter, Depends, File, UploadFile, status
import services.mongodb as mongodb
from services.mongodb import UserItem
from services.user import get_current_user
from PIL import Image, ImageOps
from io import BytesIO
from services.image import store_blob, get_blob_url, DEFAULT_EXPIRY
from bson import ObjectId
from services.openai import generate_wardrobe_tags, category_labels, WardrobeTag, get_wardrobe_recommendation, clothing_tag_to_embedding, get_n_closest, get_user_feedback_recommendation, generate_embeddings, complementary_categories, complementary_wardrobe_item_vectorsearch_pipline, generate_wardrobe_outfit

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
        ImageOps.exif_transpose(image, in_place=True)
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

    text_descriptor = f"{tags['name']}, {', '.join(tags['tags'])}".lower()  # added embedding when an image is uploaded
    embedding = generate_embeddings(text_descriptor)

    # Insert a document into the collection
    document = {
        "user_id": current_user['_id'],
        "name": tags['name'],
        "category": tags['category'],
        "tags": tags['tags'],
        "image_name": image_name,
        "embedding": embedding
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
async def wardrobe_search(search_term=str, current_user: UserItem = Depends(get_current_user)):
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
async def wardrobe_recommendation(_id: str, additional_prompt: str = "", current_user: UserItem = Depends(get_current_user)):
    # Given the id of a user's wardrobe item, generate an outfit. You can add ptional constraints
    result = mongodb.wardrobe.find_one({"_id": ObjectId(_id), "user_id": current_user['_id']})
    user_object = mongodb.users.find_one({"_id": current_user['_id']})

    exclude_list = []
    if user_object.get('userdefined_profile')['clothing_dislikes'] != {}:
        exclude_list = list(user_object.get('userdefined_profile')['clothing_dislikes'].keys())[1:]

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid item_id specified"
        )

    profile = await get_userdefined_profile(current_user)

    wardrobe_tag = WardrobeTag(name=result.get("name"), category=result.get("category"), tags=result.get("tags"))
    clothing_recommendations = get_wardrobe_recommendation(
        wardrobe_tag, profile, additional_prompt)  # added user persona

    gender = profile['gender']

    result = []
    # for clothing_tag_embedded in map(clothing_tag_to_embedding, clothing_tags):
    #     rec = list(get_n_closest(clothing_tag_embedded, 1, gender_requirements=[gender,"U"]))[0]
    #     rec['_id'] = str(rec['_id'])
    #     result.append(rec)

    for clothing_tag, category in clothing_recommendations:  # Unpack tuple (ClothingTag, category)
        clothing_tag_embedded = clothing_tag_to_embedding(clothing_tag)  # Pass only ClothingTag

        if category == "Jackets":
            category = "Tops"

        # Retrieve the closest matching clothing item
        rec = list(get_n_closest(clothing_tag_embedded, 1, category_requirement=category,
                   gender_requirements=[gender, "U"], exclude_names=exclude_list))[0]
        rec['_id'] = str(rec['_id'])  # Ensure _id is a string

        result.append(rec)  # Append the final result

    return result


@router.get("/wardrobe/userdefined_profile")
async def get_userdefined_profile(current_user: dict = Depends(get_current_user)):
    profile = current_user["userdefined_profile"]

    if profile['gender'] == "Male":
        profile['gender'] = "M"
    else:
        profile['gender'] = "F"

    if profile['clothing_likes'] != {}:
        # just keeping the latest 5 entries and keeping the datatype as list
        profile['clothing_likes'] = list(profile['clothing_likes'].keys())[-1:-6:-1]
    else:
        profile['clothing_likes'] = []

    if profile['clothing_dislikes'] != {}:
        # get the last 5 dislikes (type:list, [category,item name, what they dislike(style/color/item), dislike reason])
        dislikes = profile['clothing_dislikes']['feedback'][-1:-6:-1]

        # get just the dislike reason from the list and add it to another list with just the dislike reasons
        dislikes_list = []
        for item in dislikes:
            dislikes_list.append(item[3])

        profile['clothing_dislikes'] = dislikes_list

    else:
        profile['clothing_dislikes'] = []

    return profile


@router.get("/wardrobe/feedback_recommendation")
async def get_feedback_recommendation(starting_id, previous_rec_id, dislike_reason: str, current_user: dict = Depends(get_current_user)):
    # previous_rec and starting should be the _id of the mongodb object for the previously rec clothing item and the starting item

    starting_mongodb_object = mongodb.wardrobe.find_one({"_id": ObjectId(starting_id), "user_id": current_user['_id']})
    disliked_mongodb_object = mongodb.catalogue.find_one({"_id": ObjectId(previous_rec_id)})
    user_obj = mongodb.users.find_one({"_id": current_user['_id']})

    starting_item = WardrobeTag(name=starting_mongodb_object.get(
        "name"), category=starting_mongodb_object.get("category"), tags=starting_mongodb_object.get("tags"))
    disliked_item = WardrobeTag(name=disliked_mongodb_object.get("name"), category=disliked_mongodb_object.get(
        "category"), tags=disliked_mongodb_object.get("other_tags"))

    profile = await get_userdefined_profile(current_user)
    gender = profile['gender']

    exclude_list = []
    if user_obj.get('userdefined_profile')['clothing_dislikes'] != {}:
        exclude_list = list(user_obj.get('userdefined_profile')['clothing_dislikes'].keys())[1:]

    recommended_ClothingTag = get_user_feedback_recommendation(starting_item, disliked_item, dislike_reason, profile)

    clothing_tag_embedded = clothing_tag_to_embedding(recommended_ClothingTag)
    rec = list(get_n_closest(clothing_tag_embedded, 1,
               exclude_names=exclude_list, gender_requirements=[gender, "U"]))[0]
    rec['_id'] = str(rec['_id'])

    # outputs the mongodb object for the clothing item from the catalogue as a dictionary.
    return rec


@router.post("/wardrobe/regenerate_tags")
async def regenerate_wardrobe_tags(current_user: UserItem = Depends(get_current_user), all_users: bool = False):
    """
    This endpoint regenerates wardrobe tags for all existing wardrobe items.
    If `all_users` is set to True, it regenerates tags for all users.
    Skips items that fail and resumes from where it broke.
    """
    if all_users:
        users = mongodb.users.find({}, {"_id": 1})
        user_ids = [user["_id"] for user in users]
    else:
        user_ids = [current_user['_id']]

    updated_items = []
    failed_items = []

    for user_id in user_ids:
        items = mongodb.wardrobe.find({"user_id": user_id})

        for item in items:
            try:
                image_url = get_blob_url(item["image_name"], DEFAULT_EXPIRY)
                new_tags = generate_wardrobe_tags(image_url)

                update_query = {
                    "$set": {
                        "name": new_tags['name'],
                        "category": new_tags['category'],
                        "tags": new_tags['tags']
                    }
                }

                mongodb.wardrobe.update_one({"_id": item["_id"]}, update_query)
                updated_items.append({
                    "_id": str(item["_id"]),
                    "user_id": str(user_id),
                    "name": new_tags['name'],
                    "category": new_tags['category'],
                    "tags": new_tags['tags']
                })
            except Exception as e:
                failed_items.append({
                    "_id": str(item["_id"]),
                    "user_id": str(user_id),
                    "error": str(e)
                })

    return {
        "message": "Wardrobe tags regeneration completed with some skips.",
        "updated_items": updated_items,
        "failed_items": failed_items
    }


@router.post("/wardrobe/complementary_items")
async def complementary_items(starting_id: str, current_user: dict = Depends(get_current_user)):
    # 1) Fetch the starting item
    starting_mongodb_object = mongodb.wardrobe.find_one({
        "_id": ObjectId(starting_id),
        "user_id": current_user['_id']
    })
    if not starting_mongodb_object:
        raise HTTPException(status_code=404, detail="Starting item not found")

    starting_category = starting_mongodb_object.get("category")
    embedding = starting_mongodb_object.get("embedding")
    user = ObjectId(current_user['_id'])

    # 2) Determine categories to search
    return_categories = complementary_categories(starting_category)

    results = []
    for category in return_categories:
        pipeline = complementary_wardrobe_item_vectorsearch_pipline(user, category, embedding)
        docs = list(mongodb.wardrobe.aggregate(pipeline))

        sub_list = []
        # Convert all ObjectIds to strings
        for doc in docs:
            if "_id" in doc and isinstance(doc["_id"], ObjectId):
                doc["_id"] = str(doc["_id"])
            # If user_id is also present as an ObjectId, convert that too:
            if "user_id" in doc and isinstance(doc["user_id"], ObjectId):
                doc["user_id"] = str(doc["user_id"])
            sub_list.append(doc)

        results.append(sub_list)

    return results


@router.get("/wardrobe/outfit_from_wardrobe")
async def outfit_from_wardrobe(starting_id: str, addn_prompt: str = "", current_user: dict = Depends(get_current_user)):
    # given the _id of a starting item to style ouftfit from wardrobe, get a list of dictionaries containing all the other items to complete the outfit
    """
    EXAMPLE:
    input: _id = "67dda1b467bf4ce7c6cfddef"

    output:
    [
    {
        "image_url": "https://storage.googleapis.com/kagame_bucket_1/a4124254-4df2-4aa5-b8ee-955cdeb76d72?Expires=1742835476&GoogleAccessId=default-621%40kagame-432309.iam.gserviceaccount.com&Signature=Be3yvKciDL6ogS3HSGEKNu6lpcoumf157LHyhXgzfxgeQEa91ojiNJ2XD7H6rI%2B2TU9QQ98ofaGiclOtm4r%2BQgbQ%2FoJ8%2FT7BVhvDsYAkLc971ZtqDwabtraA37iUyPvrcCuwpmWs8LNzCWIpavTJ0UJ%2Bc8e4dzB%2FLN9pRP8sxn%2B6KjV%2Bxpn917Ka05XYaxHLJ8slcpNLmg6RTCHHoTjTpCUukk%2FoN2krQo%2BwWbD8yJmaXO6uiurREY1tH4SkowHnb1%2Bv7e%2BqLeuncRehTvpt0WUSX%2FCK6rqgCRlGPJnv1Ebg55aeHJ1HXOF7aI5V769MyqI4KwrwaAhBO7bCiMFH1w%3D%3D",
        "_id": "67deef3af058e477e4f32f32",
        "name": "Comfortable Jogger Pants",
        "category": "Bottoms",
        "tags": [
        "sporty, casual",
        "casual",
        "regular fit",
        "green",
        "cotton blend"
        ]
    }
    ]
    note: This would have more items in the list but the user wardrobe was small so you only see pants in this example

    """

    closest_items = await complementary_items(starting_id, current_user)
    starting_item = await get_item(starting_id, current_user)

    starting_name = starting_item['name']
    starting_cat = starting_item['category']

    user_style = current_user['userdefined_profile']['style']

    outfit_ids = generate_wardrobe_outfit(user_style, closest_items, starting_name, starting_cat, addn_prompt)

    outfit = []
    for _id in outfit_ids:
        item = await get_item(_id, current_user)
        outfit.append(item)

    return outfit
