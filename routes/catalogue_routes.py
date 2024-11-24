from fastapi import APIRouter, Depends, HTTPException
import services.mongodb as mongodb
from services.openai import (
    str_to_clothing_tag,
    clothing_tag_to_embedding,
    get_n_closest,
    ClothingTagEmbed,
    user_style_to_embedding,
    StyleAnalysisResponse,
)
from bson import ObjectId
from services.user import get_current_user
from typing import Optional, List
from pydantic import BaseModel

router = APIRouter()

@router.get("/shop/items")
def get_items_by_retailer(retailer: str = None, include_embeddings: bool = False, limit: int = 0):
    try:
        # Define the filter for the retailer if provided
        filter_criteria = {}
        if retailer:
            filter_criteria["retailer"] = retailer

        # Adjust the projection to include new fields and embeddings if requested
        projection = {
            "name": 1,
            "category": 1,
            "price": 1,
            "image_url": 1,
            "product_url": 1,
            "clothing_type": 1,
            "color": 1,
            "material": 1,
            "other_tags": 1
        }
        if include_embeddings:
            projection["embedding"] = 1

        # Query for items with an optional limit
        items_cursor = mongodb.catalogue.find(filter_criteria, projection)
        if limit > 0:
            items_cursor = items_cursor.limit(limit)

        response = []
        for item in items_cursor:
            item_data = {
                "id": str(item["_id"]),
                "name": item.get("name", ""),
                "category": item.get("category", ""),
                "price": item.get("price", ""),
                "image_url": item.get("image_url", ""),
                "product_url": item.get("product_url", ""),
                "clothing_type": item.get("clothing_type", ""),
                "color": item.get("color", ""),
                "material": item.get("material", ""),
                "other_tags": item.get("other_tags", "")
            }
            if include_embeddings and "embedding" in item:
                item_data["embedding"] = item["embedding"]

            response.append(item_data)

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching items: {str(e)}"
        )

@router.get("/shop/search")
def get_search_result(search: str, n: int):
    clothing_tag = str_to_clothing_tag(search)
    embedding = clothing_tag_to_embedding(clothing_tag)
    recs = list(get_n_closest(embedding, n))

    for rec in recs:
        rec['_id'] = str(rec['_id'])

    return recs

@router.get("/shop/similar_items")
def get_similar_items(id: str, n: int = 5):
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid id format")

        # Fetch the product by _id
        product = mongodb.catalogue.find_one({"_id": object_id})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found with given id")

        # Ensure the product has embeddings
        required_fields = ['clothing_type_embed', 'color_embed', 'material_embed', 'other_tags_embed']
        if not all(field in product and product[field] for field in required_fields):
            raise HTTPException(status_code=500, detail="Product embeddings not found")

        # Create a ClothingTagEmbed object from the product's embeddings
        tag_embed = ClothingTagEmbed(
            clothing_type_embed=product.get('clothing_type_embed'),
            color_embed=product.get('color_embed'),
            material_embed=product.get('material_embed'),
            other_tags_embed=product.get('other_tags_embed')  # Use 'other_tags_embed' here
        )

        # Fetch the n + 1 closest items to account for possible inclusion of the original item
        recs = list(get_n_closest(tag_embed, n + 1))

        # Prepare the response, excluding the original item
        response = []
        for rec in recs:
            rec_id_str = str(rec["_id"])
            if rec_id_str == id:
                continue  # Skip the original item

            item_data = {
                "id": rec_id_str,
                "name": rec.get("name", ""),
                "category": rec.get("category", ""),
                "price": rec.get("price", ""),
                "image_url": rec.get("image_url", ""),
                "product_url": rec.get("product_url", ""),
                "clothing_type": rec.get("clothing_type", ""),
                "color": rec.get("color", ""),
                "material": rec.get("material", ""),
                "other_tags": rec.get("other_tags", "")
            }
            response.append(item_data)

            # Break the loop once we have n items
            if len(response) >= n:
                break

        return response

    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching similar items: {str(e)}"
        )

from fastapi import APIRouter, Depends, HTTPException, status
import services.mongodb as mongodb
from pydantic import BaseModel
from services.mongodb import UserItem
from typing import Optional
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
import json

router = APIRouter()

# Initialize the OpenAI client
openai_client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJ_ID,
    api_key=OPENAI_API_KEY
)

# Define the response models
class StyleSuggestion(BaseModel):
    style: str
    description: str
    reasoning: str

class StyleAnalysisResponse(BaseModel):
    top_styles: list[StyleSuggestion]

# Modify the User model to include 'user_style' as an optional field
class User(BaseModel):
    username: str
    password: str
    user_style: Optional[StyleAnalysisResponse] = None

@router.get("/user/style-analysis")
async def get_style_analysis(current_user: UserItem = Depends(get_current_user)):
    # Retrieve all wardrobe items for the user
    user_id = current_user['_id']
    items_cursor = mongodb.wardrobe.find({"user_id": user_id})
    wardrobe_items = []
    for item in items_cursor:
        wardrobe_item = {
            "name": item["name"],
            "category": item["category"],
            "tags": item["tags"]
        }
        wardrobe_items.append(wardrobe_item)
    
    if not wardrobe_items:
        raise HTTPException(status_code=400, detail="No wardrobe items found for the user.")

    # Prepare the data to send to OpenAI
    wardrobe_items_json = json.dumps(wardrobe_items)

    # Prepare the prompt
    prompt = (
        "Based on my wardrobe items, suggest the top 3 styles that fit me, "
        "along with descriptions and reasoning. The output should be a JSON "
        "object matching the following schema:\n"
        "{\n"
        "  \"top_styles\": [\n"
        "    {\n"
        "      \"style\": \"string\",\n"
        "      \"description\": \"string\",\n"
        "      \"reasoning\": \"string\"\n"
        "    },\n"
        "    ... (2 more styles)\n"
        "  ]\n"
        "}"
    )

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": "You are a fashion expert assistant."
        },
        {
            "role": "user",
            "content": f"{prompt}\nHere are my wardrobe items:\n{wardrobe_items_json}"
        }
    ]

    # Call the OpenAI API
    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=StyleAnalysisResponse,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Extract and return the parsed response
    style_analysis = completion.choices[0].message.parsed

    # Update the user's 'user_style' field in the database
    mongodb.users.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"user_style": style_analysis.dict()}}
    )

    return style_analysis.dict()

@router.get("/user/style-embedding")
def get_user_style_embedding(current_user: UserItem = Depends(get_current_user)):
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        raise HTTPException(status_code=400, detail="User style not found.")

    user_style_data = user["user_style"]
    # Parse user_style_data into StyleAnalysisResponse
    try:
        user_style = StyleAnalysisResponse.parse_obj(user_style_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error parsing user style: {str(e)}"
        )

    # Get the embedding
    try:
        embedding = user_style_to_embedding(user_style)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating embedding: {str(e)}"
        )

    # Update the user's 'style_embedding' field in the database
    mongodb.users.update_one(
        {"_id": current_user["_id"]}, {"$set": {"style_embedding": embedding}}
    )

    return {"embedding": embedding}

from fastapi import APIRouter, Depends, HTTPException
import services.mongodb as mongodb
from services.user import get_current_user
from services.mongodb import UserItem
from typing import List

router = APIRouter()

@router.get("/shop/recommendations")
def get_recommendations(current_user: UserItem = Depends(get_current_user)):
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or 'style_embedding' not in user:
        raise HTTPException(status_code=400, detail="User style embedding not found.")
    style_embedding = user['style_embedding']
    
    try:
        pipeline = [
            # Compute dot products between user's style embedding and item's embeddings
            {
                "$addFields": {
                    "clothing_type_score": {
                        "$let": {
                            "vars": {
                                "user_embedding": style_embedding,
                                "item_embedding": "$clothing_type_embed"
                            },
                            "in": {
                                "$sum": {
                                    "$map": {
                                        "input": {"$range": [0, {"$size": "$$user_embedding"}]},
                                        "as": "idx",
                                        "in": {
                                            "$multiply": [
                                                {"$arrayElemAt": ["$$user_embedding", "$$idx"]},
                                                {"$arrayElemAt": ["$$item_embedding", "$$idx"]}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "color_score": {
                        "$let": {
                            "vars": {
                                "user_embedding": style_embedding,
                                "item_embedding": "$color_embed"
                            },
                            "in": {
                                "$sum": {
                                    "$map": {
                                        "input": {"$range": [0, {"$size": "$$user_embedding"}]},
                                        "as": "idx",
                                        "in": {
                                            "$multiply": [
                                                {"$arrayElemAt": ["$$user_embedding", "$$idx"]},
                                                {"$arrayElemAt": ["$$item_embedding", "$$idx"]}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "material_score": {
                        "$let": {
                            "vars": {
                                "user_embedding": style_embedding,
                                "item_embedding": "$material_embed"
                            },
                            "in": {
                                "$sum": {
                                    "$map": {
                                        "input": {"$range": [0, {"$size": "$$user_embedding"}]},
                                        "as": "idx",
                                        "in": {
                                            "$multiply": [
                                                {"$arrayElemAt": ["$$user_embedding", "$$idx"]},
                                                {"$arrayElemAt": ["$$item_embedding", "$$idx"]}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "other_tags_score": {
                        "$let": {
                            "vars": {
                                "user_embedding": style_embedding,
                                "item_embeddings": "$other_tags_embed"
                            },
                            "in": {
                                "$sum": {
                                    "$map": {
                                        "input": "$$item_embeddings",
                                        "as": "item_embed",
                                        "in": {
                                            "$sum": {
                                                "$map": {
                                                    "input": {"$range": [0, {"$size": "$$user_embedding"}]},
                                                    "as": "idx",
                                                    "in": {
                                                        "$multiply": [
                                                            {"$arrayElemAt": ["$$user_embedding", "$$idx"]},
                                                            {"$arrayElemAt": ["$$item_embed", "$$idx"]}
                                                        ]
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            # Calculate the overall similarity score
            {
                "$addFields": {
                    "similarity_score": {
                        "$add": [
                            {"$multiply": [0.4, "$clothing_type_score"]},
                            {"$multiply": [0.3, "$color_score"]},
                            {"$multiply": [0.2, "$material_score"]},
                            {"$multiply": [0.1, "$other_tags_score"]}
                        ]
                    }
                }
            },
            # Sort items by similarity score in descending order
            {
                "$sort": {"similarity_score": -1}
            },
            # Limit to top 50 items
            {
                "$limit": 50
            },
            # Project the necessary fields
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "category": 1,
                    "clothing_type": 1,
                    "color": 1,
                    "material": 1,
                    "other_tags": 1,
                    "price": 1,
                    "image_url": 1,
                    "product_url": 1,
                    "retailer": 1,
                    "gender": 1,
                    "similarity_score": 1  # Include for debugging
                }
            }
        ]

        recommendations = list(mongodb.catalogue.aggregate(pipeline))
        
        # Convert ObjectId to string and remove '_id' field
        for rec in recommendations:
            rec['id'] = str(rec['_id'])
            del rec['_id']

        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching recommendations: {str(e)}"
        )
