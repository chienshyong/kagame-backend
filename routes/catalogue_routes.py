from fastapi import APIRouter, Depends, HTTPException, Query
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
from pydantic import BaseModel, ValidationError
from services.mongodb import UserItem
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
import json
from tqdm import tqdm  # <-- Added tqdm import

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


@router.get("/shop/recommendations")
def get_recommendations(current_user: UserItem = Depends(get_current_user), n: int = 50):
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        raise HTTPException(status_code=400, detail="User style not found.")
    
    user_style_data = user["user_style"]
    try:
        user_style = StyleAnalysisResponse.parse_obj(user_style_data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error parsing user style: {str(e)}"
        )
    
    recommendations = []
    item_ids = set()
    styles_count = len(user_style.top_styles)
    if styles_count == 0:
        raise HTTPException(status_code=400, detail="No styles found in user style analysis.")

    items_per_style = max(n // styles_count, 1)
    
    for style_suggestion in user_style.top_styles:
        style_prompt = f"{style_suggestion.style}: {style_suggestion.description} {style_suggestion.reasoning}"
        clothing_tag = str_to_clothing_tag(style_prompt)
        tag_embed = clothing_tag_to_embedding(clothing_tag)
        recs = list(get_n_closest(tag_embed, items_per_style))
        for rec in recs:
            rec_id_str = str(rec["_id"])
            if rec_id_str in item_ids:
                continue  # Skip duplicates
            item_ids.add(rec_id_str)
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
            recommendations.append(item_data)
            if len(recommendations) >= n:
                break
        if len(recommendations) >= n:
            break
    return recommendations

# -------------------------------------------------------------------
# 1) "Loose" models for OpenAI parse (no advanced constraints)
# -------------------------------------------------------------------

class LooseRecommendedItem(BaseModel):
    name: str
    color: List[str]
    material: List[str]
    other_tags: List[str]
    reason: str

class LooseOutfitRecommendation(BaseModel):
    style: str
    description: str
    name: str
    items: List[LooseRecommendedItem]

class LooseStylingLLMResponse(BaseModel):
    outfits: List[LooseOutfitRecommendation]

# -------------------------------------------------------------------
# 2) "Strict" models for post-parse validation (with constraints)
# -------------------------------------------------------------------

class StrictRecommendedItem(BaseModel):
    name: str
    color: List[str]
    material: List[str]
    other_tags: List[str]
    reason: str

    def validate_tags(self):
        # At least 1 color tag
        if len(self.color) < 1:
            raise ValueError(
                f"'color' must have at least 1 tag, got {len(self.color)}."
            )
        # At least 1 material tag
        if len(self.material) < 1:
            raise ValueError(
                f"'material' must have at least 1 tag, got {len(self.material)}."
            )
        # At least 5 'other_tags'
        if len(self.other_tags) < 5:
            raise ValueError(
                f"'other_tags' must have at least 10 tags, got {len(self.other_tags)}."
            )

class StrictOutfitRecommendation(BaseModel):
    style: str
    description: str
    name: str
    items: List[StrictRecommendedItem]

    def validate_items(self):
        for item in self.items:
            item.validate_tags()

class StrictStylingLLMResponse(BaseModel):
    outfits: List[StrictOutfitRecommendation]

    def validate_outfits(self):
        for outfit in self.outfits:
            outfit.validate_items()

##############################
# Input Models for Outfit Search
##############################

class RecommendedItemInput(BaseModel):
    name: str
    color: List[str]
    material: List[str]
    other_tags: List[str]
    reason: str

class OutfitInput(BaseModel):
    style: str
    description: str
    name: str
    items: List[RecommendedItemInput]

class OutfitSearchRequest(BaseModel):
    outfits: List[OutfitInput]

##############################
# Output Models for Matched Catalog Items
##############################

class CatalogItem(BaseModel):
    """
    A minimal set of fields from your catalogue for a matched item.
    The fields color, material, and other_tags are lists of strings.
    """
    id: str
    name: str
    category: str
    price: float
    image_url: str
    product_url: str
    clothing_type: str
    color: List[str]
    material: List[str]
    other_tags: List[str]

class OutfitSearchMatchedItem(BaseModel):
    """
    Wraps the original LLM-recommended item and its matched catalogue item.
    """
    original: RecommendedItemInput
    match: CatalogItem

class OutfitSearchResult(BaseModel):
    style: str
    description: str
    name: str
    items: List[OutfitSearchMatchedItem]

class OutfitSearchResponse(BaseModel):
    outfits: List[OutfitSearchResult]

##############################
# Helper Function
##############################

def ensure_list(value) -> List[str]:
    """
    Ensures that the input value is returned as a list of strings.
    If the value is already a list, it is returned unchanged.
    If it is a string, it is wrapped into a list.
    Otherwise, an empty list is returned.
    """
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return [value]
    else:
        return []

@router.get("/shop/item-outfit-search")
def item_outfit_search(item_id: str, current_user: UserItem = Depends(get_current_user)) -> OutfitSearchResponse:
    """
    1) Generate 3 outfits for the specified item (like /shop/item-styling).
    2) For each recommended item in each outfit, retrieve the best catalog match (like /shop/outfit-search).
    3) Return the final structure with matched items (including image_url, etc.).
    """
    # -------------------------------------------------------
    #  Fetch the chosen item from the catalogue
    # -------------------------------------------------------
    try:
        object_id = ObjectId(item_id)
        print(f"[DEBUG] Converted item_id '{item_id}' to ObjectId: {object_id}")
    except Exception as e:
        print(f"[DEBUG] Failed to convert item_id '{item_id}' to ObjectId: {e}")
        raise HTTPException(status_code=400, detail="Invalid item_id format")

    product = mongodb.catalogue.find_one({"_id": object_id})
    if not product:
        print(f"[DEBUG] Product not found for _id: {object_id}")
        raise HTTPException(status_code=404, detail="Product not found with given id")
    print(f"[DEBUG] Retrieved product: {product}")

    # -------------------------------------------------------
    #  Fetch the user's style
    # -------------------------------------------------------
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        print(f"[DEBUG] User style not found for user: {current_user['_id']}")
        raise HTTPException(status_code=400, detail="User style not found.")
    user_style_data = user["user_style"]
    print(f"[DEBUG] Retrieved user style data: {user_style_data}")
    try:
        user_style = StyleAnalysisResponse.parse_obj(user_style_data)
    except Exception as e:
        print(f"[DEBUG] Error parsing user style: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing user style: {e}")

    # -------------------------------------------------------
    #  Determine pairing instructions based on clothing_type
    # -------------------------------------------------------
    clothing_type = (product.get("clothing_type") or "").lower()
    print(f"[DEBUG] Product clothing_type: {clothing_type}")
    if clothing_type in ["top", "shirt", "blouse", "tank top", "camisole top", "t-shirt"]:
        pairing_instructions = "Because this item is a top, recommend (1) a bottom and (2) shoes."
    elif clothing_type in ["bottom", "pants", "jeans", "leggings", "shorts", "skirt"]:
        pairing_instructions = "Because this item is a bottom, recommend (1) a top and (2) shoes."
    elif clothing_type in ["shoes", "heels", "sneakers", "boots"]:
        pairing_instructions = "Because this item is shoes, recommend (1) a top and (2) a bottom."
    else:
        pairing_instructions = "Please recommend two items that complement this piece for a cohesive outfit."
    print(f"[DEBUG] Pairing instructions: {pairing_instructions}")

    # -------------------------------------------------------
    #  Prepare text for user’s top styles
    # -------------------------------------------------------
    top_styles_instructions = []
    for style_info in tqdm(user_style.top_styles, desc="Processing user styles"):
        top_styles_instructions.append(
            f"Style: {style_info.style}\nDescription: {style_info.description}"
        )
    styles_text = "\n\n".join(top_styles_instructions)
    print(f"[DEBUG] Constructed styles_text:\n{styles_text}")

    # -------------------------------------------------------
    #  Build the LLM prompts
    # -------------------------------------------------------
    system_message = {
        "role": "system",
        "content": (
            "You are a helpful fashion assistant. "
            "We have exactly 3 user styles, plus 1 chosen clothing item. "
            "For each of the 3 styles, build an outfit that:\n"
            "  1) Includes the already-chosen item.\n"
            "  2) Recommends 2 additional items, each with:\n"
            "     - name (string)\n"
            "     - color (array of strings, at least 3)\n"
            "     - material (array of strings)\n"
            "     - other_tags (array of strings, at least 10)\n"
            "     - reason (string)\n"
            "  3) The top-level JSON must have 'outfits' array. Each outfit has:\n"
            "     - 'style', 'description', 'name', 'items'.\n\n"
            "No extra keys, no commentary; only a valid JSON object that matches:\n"
            "{\n"
            "  \"outfits\": [\n"
            "    {\n"
            "      \"style\": \"...\",\n"
            "      \"description\": \"...\",\n"
            "      \"name\": \"...\",\n"
            "      \"items\": [\n"
            "        {\"name\": \"...\", \"color\": [...], ...},\n"
            "        {...}\n"
            "      ]\n"
            "    },\n"
            "    ... (2 more outfits)...\n"
            "  ]\n"
            "}"
        )
    }

    user_message_content = (
        f"User's top 3 styles:\n\n{styles_text}\n\n"
        f"Chosen item:\n"
        f"- Name: {product.get('name', '')}\n"
        f"- Clothing Type: {product.get('clothing_type', '')}\n"
        f"- Color: {product.get('color', '')}\n"
        f"- Material: {product.get('material', '')}\n"
        f"- Other Tags: {product.get('other_tags', [])}\n\n"
        f"{pairing_instructions}\n\n"
        "We only want the JSON response with the structure described above. No additional commentary."
    )
    user_message = {"role": "user", "content": user_message_content}

    messages = [system_message, user_message]
    print("[DEBUG] LLM messages:")
    print(json.dumps(messages, indent=2))

    # -------------------------------------------------------
    #  Step 1: LLM call for LooseStylingLLMResponse
    # -------------------------------------------------------
    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=LooseStylingLLMResponse
        )
        print(f"[DEBUG] LLM raw completion: {completion}")
    except Exception as e:
        print(f"[DEBUG] Error during LLM call: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not completion or not completion.choices:
        print("[DEBUG] No response from OpenAI")
        raise HTTPException(status_code=500, detail="No response from OpenAI")

    loose_data = completion.choices[0].message.parsed  # LooseStylingLLMResponse
    print(f"[DEBUG] Loose LLM response: {loose_data}")

    # -------------------------------------------------------
    #  Step 2: Validate with StrictStylingLLMResponse
    # -------------------------------------------------------
    data_dict = loose_data.dict()
    print(f"[DEBUG] Loose data dict: {data_dict}")
    try:
        strict_response = StrictStylingLLMResponse.parse_obj(data_dict)
        strict_response.validate_outfits()  # Additional custom validations
        print("[DEBUG] Strict response validated successfully.")
    except Exception as e:
        print(f"[DEBUG] Strict response validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Pydantic validation error: {e}")

    # -------------------------------------------------------
    #  Step 3: Perform outfit-search-like logic to find best matches
    # -------------------------------------------------------
    request_payload = OutfitSearchRequest(outfits=[
        OutfitInput(
            style=o.style,
            description=o.description,
            name=o.name,
            items=[
                RecommendedItemInput(
                    name=i.name,
                    color=i.color,
                    material=i.material,
                    other_tags=i.other_tags,
                    reason=i.reason
                )
                for i in o.items
            ]
        )
        for o in strict_response.outfits
    ])
    print(f"[DEBUG] Request payload for outfit search: {request_payload.model_dump_json(indent=2)}")

    final_outfits: List[OutfitSearchResult] = []
    # Wrap the outfit processing with tqdm for progress indication
    for outfit in tqdm(request_payload.outfits, desc="Processing outfits"):
        matched_items: List[OutfitSearchMatchedItem] = []
        # Wrap the inner loop for recommended items
        for recommended_item in tqdm(outfit.items, desc="Matching recommended items", leave=False):
            text_parts = [
                recommended_item.name,
                *recommended_item.color,
                *recommended_item.material,
                *recommended_item.other_tags,
            ]
            prompt_text = " ".join(text_parts)
            print(f"[DEBUG] Prompt text for matching: {prompt_text}")

            clothing_tag = str_to_clothing_tag(prompt_text)
            embedding = clothing_tag_to_embedding(clothing_tag)
            recs = list(get_n_closest(embedding, 1))
            if recs:
                best = recs[0]
                match_item = CatalogItem(
                    id=str(best["_id"]),
                    name=best.get("name", ""),
                    category=best.get("category", ""),
                    price=best.get("price", 0.0),
                    image_url=best.get("image_url", ""),
                    product_url=best.get("product_url", ""),
                    clothing_type=best.get("clothing_type", ""),
                    color=ensure_list(best.get("color", [])),
                    material=ensure_list(best.get("material", [])),
                    other_tags=ensure_list(best.get("other_tags", [])),
                )
                print(f"[DEBUG] Best match found: {match_item}")
            else:
                match_item = CatalogItem(
                    id="",
                    name="No Match Found",
                    category="",
                    price=0.0,
                    image_url="",
                    product_url="",
                    clothing_type="",
                    color=[],
                    material=[],
                    other_tags=[]
                )
                print(f"[DEBUG] No match found for prompt: {prompt_text}")

            matched_items.append(
                OutfitSearchMatchedItem(
                    original=recommended_item,
                    match=match_item
                )
            )
        final_outfits.append(
            OutfitSearchResult(
                style=outfit.style,
                description=outfit.description,
                name=outfit.name,
                items=matched_items
            )
        )
    print(f"[DEBUG] Final outfits prepared: {final_outfits}")

    return OutfitSearchResponse(outfits=final_outfits)

@router.get("/shop/item/{item_id}")
def get_item_by_id(item_id: str):
    """
    Retrieves a single item from the 'catalogue' collection by ID, returning all fields.
    """
    try:
        object_id = ObjectId(item_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid item_id format")

    item_doc = mongodb.catalogue.find_one({"_id": object_id})
    if not item_doc:
        raise HTTPException(status_code=404, detail="Item not found with given id")

    # Convert MongoDB's _id to a friendlier field
    item_doc["id"] = str(item_doc["_id"])
    del item_doc["_id"]  # Optionally remove the original _id field

    return item_doc
