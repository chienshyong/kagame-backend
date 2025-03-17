from fastapi import APIRouter, Depends, HTTPException, Query
import services.mongodb as mongodb
from services.openai import (
    ClothingTag,
    str_to_clothing_tag,
    clothing_tag_to_embedding,
    get_n_closest,
    ClothingTagEmbed,
    user_style_to_embedding,
    StyleAnalysisResponse,
    get_n_closest_with_filter
)
from bson import ObjectId
from services.user import get_current_user
from typing import Optional, List
from pydantic import BaseModel
from services.mongodb import UserItem
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
import json
from pydantic import parse_obj_as

router = APIRouter()


@router.get("/shop/items")
def get_items_by_retailer(retailer: str = None, include_embeddings: bool = False, limit: int = 0):
    try:
        filter_criteria = {}
        if retailer:
            filter_criteria["retailer"] = retailer

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


# @router.get("/shop/search")
# def get_search_result(search: str, n: int):
#     clothing_tag = str_to_clothing_tag(search)
#     embedding = clothing_tag_to_embedding(clothing_tag)
#     recs = list(get_n_closest(embedding, n))

#     # Convert _id to string
#     for rec in recs:
#         rec['_id'] = str(rec['_id'])
#     return recs


@router.get("/shop/similar_items")
def get_similar_items(id: str, n: int = 5):
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid id format")

        product = mongodb.catalogue.find_one({"_id": object_id})
        if not product:
            raise HTTPException(status_code=404, detail="Product not found with given id")

        required_fields = ['clothing_type_embed', 'color_embed', 'material_embed', 'other_tags_embed']
        if not all(field in product and product[field] for field in required_fields):
            raise HTTPException(status_code=500, detail="Product embeddings not found")

        tag_embed = ClothingTagEmbed(
            clothing_type_embed=product.get('clothing_type_embed'),
            color_embed=product.get('color_embed'),
            material_embed=product.get('material_embed'),
            other_tags_embed=product.get('other_tags_embed')
        )

        recs = list(get_n_closest(tag_embed, n + 1))

        response = []
        for rec in recs:
            rec_id_str = str(rec["_id"])
            if rec_id_str == id:
                continue  # skip original

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
            if len(response) >= n:
                break

        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching similar items: {str(e)}"
        )

# ------------------------------------------
# Initialize the OpenAI client
# ------------------------------------------
openai_client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJ_ID,
    api_key=OPENAI_API_KEY
)

# ------------------------------------------
# Response models
# ------------------------------------------
class StyleSuggestion(BaseModel):
    style: str
    description: str
    reasoning: str


class StyleAnalysisResponse(BaseModel):
    top_styles: List[StyleSuggestion]

class User(BaseModel):
    username: str
    password: str
    user_style: Optional[StyleAnalysisResponse] = None


@router.get("/user/style-analysis")
async def get_style_analysis(current_user: UserItem = Depends(get_current_user)):
    try:
        user_id = current_user['_id']
        items_cursor = mongodb.wardrobe.find({"user_id": user_id})
        wardrobe_items = []
        for item in items_cursor:
            wardrobe_items.append({
                "name": item["name"],
                "category": item["category"],
                "tags": item["tags"]
            })

        if not wardrobe_items:
            raise HTTPException(status_code=400, detail="No wardrobe items found for the user.")

        wardrobe_items_json = json.dumps(wardrobe_items)
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
        messages = [
            {"role": "system", "content": "You are a fashion expert assistant."},
            {"role": "user", "content": f"{prompt}\nHere are my wardrobe items:\n{wardrobe_items_json}"}
        ]

        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=StyleAnalysisResponse,
        )
        style_analysis = completion.choices[0].message.parsed

        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {"user_style": style_analysis.dict()}}
        )
        return style_analysis.dict()

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/style-embedding")
def get_user_style_embedding(current_user: UserItem = Depends(get_current_user)):
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        raise HTTPException(status_code=400, detail="User style not found.")

    try:
        user_style_data = user["user_style"]
        user_style = StyleAnalysisResponse.parse_obj(user_style_data)
        embedding = user_style_to_embedding(user_style)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    mongodb.users.update_one(
        {"_id": current_user["_id"]}, {"$set": {"style_embedding": embedding}}
    )
    return {"embedding": embedding}

@router.get("/shop/recommendations")
def get_recommendations(current_user: UserItem = Depends(get_current_user), n: int = 50):
    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        raise HTTPException(status_code=400, detail="User style not found.")
    
    # Parse user style
    try:
        user_style_data = user["user_style"]
        user_style = parse_obj_as(StyleAnalysisResponse, user_style_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing user style: {str(e)}")
    
    
    # Ensure the recommendations_refresh_needed field exists in the DB
    # If it doesn't, create it now, set to True.
    if "recommendations_refresh_needed" not in user:
        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {"recommendations_refresh_needed": True}}
        )
        refresh_needed = True
    else:
        refresh_needed = user["recommendations_refresh_needed"]

    # Either grab existing style_recommendations or regenerate them
    style_recommendations = user.get("style_recommendations", [])

    if refresh_needed:
        # Build new style_recommendations from the user's style
        new_recommendations = []
        for style_suggestion in user_style.top_styles:
            # Combine the style and reasoning into text for the embedding
            style_prompt = f"{style_suggestion.style}: {style_suggestion.description} {style_suggestion.reasoning}"
            clothing_tag = str_to_clothing_tag(style_prompt)
            tag_embed = clothing_tag_to_embedding(clothing_tag)
            new_recommendations.append({
                'style_name': style_suggestion.style,
                'clothing_tag': clothing_tag.dict(),
                'tag_embed': tag_embed.dict()
            })

        # Store them in the user record
        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {
                "$set": {
                    "style_recommendations": new_recommendations,
                    "recommendations_refresh_needed": False
                }
            }
        )
        style_recommendations = new_recommendations

    # Now do the usual logic to get recommended items via vector search
    # (either from fresh or cached style_recommendations).
    if not style_recommendations:
        raise HTTPException(status_code=400, detail="No style recommendations available.")

    recommendations = []
    item_ids = set()
    styles_count = len(style_recommendations)
    items_per_style = max(n // styles_count, 1)

    for style_rec in style_recommendations:
        # Rebuild the ClothingTagEmbed from the dict
        tag_embed_dict = style_rec['tag_embed']
        tag_embed = parse_obj_as(ClothingTagEmbed, tag_embed_dict)

        # Perform vector search
        recs = list(get_n_closest(tag_embed, items_per_style))
        for rec in recs:
            rec_id_str = str(rec["_id"])
            if rec_id_str in item_ids:
                continue
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

class LooseRecommendedItem(BaseModel):
    reason: str
    name: str
    color: List[str]
    material: List[str]
    other_tags: List[str]
    category: str
    clothing_type: str

class LooseOutfitRecommendation(BaseModel):
    style: str
    description: str
    name: str
    items: List[LooseRecommendedItem]

class LooseStylingLLMResponse(BaseModel):
    outfits: List[LooseOutfitRecommendation]

class RecommendedItemInput(BaseModel):
    name: str
    color: List[str]
    material: List[str]
    other_tags: List[str]
    reason: str
    category: str
    clothing_type: str

class OutfitInput(BaseModel):
    style: str
    description: str
    name: str
    items: List[RecommendedItemInput]

class OutfitSearchRequest(BaseModel):
    outfits: List[OutfitInput]

class CatalogItem(BaseModel):
    id: str

class OutfitSearchMatchedItem(BaseModel):
    original: RecommendedItemInput
    match: CatalogItem

class OutfitSearchResult(BaseModel):
    style: str
    description: str
    name: str
    items: List[OutfitSearchMatchedItem]

class OutfitSearchResponse(BaseModel):
    outfits: List[OutfitSearchResult]

def ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return [value]
    else:
        return []

@router.get("/shop/item-outfit-search")
def item_outfit_search(item_id: str, current_user: UserItem = Depends(get_current_user)) -> OutfitSearchResponse:
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item_id format")

    product = mongodb.catalogue.find_one({"_id": object_id})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found with given id")

    user = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user or "user_style" not in user:
        raise HTTPException(status_code=400, detail="User style not found.")

    user_style_data = user["user_style"]
    try:
        user_style = StyleAnalysisResponse.parse_obj(user_style_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing user style: {e}"
        )

    # Build the pairing instructions from product category
    category = (product.get("category") or "").lower()
    if category == "tops":
        pairing_instructions = "Because this item is a top, recommend (1) a bottom and (2) shoes."
    elif category == "bottoms":
        pairing_instructions = "Because this item is a bottom, recommend (1) a top and (2) shoes."
    elif category == "shoes":
        pairing_instructions = "Because this item are shoes, recommend (1) a top and (2) a bottom."
    else:
        pairing_instructions = "Please recommend two items that complement this piece for a cohesive outfit."

    # Summarize top styles
    top_styles_instructions = []
    for style_info in user_style.top_styles:
        top_styles_instructions.append(
            f"Style: {style_info.style}\nDescription: {style_info.description}"
        )
    styles_text = "\n\n".join(top_styles_instructions)

    system_message = {
        "role": "system",
        "content": (
            "You are an expert fashion stylist. We have exactly 3 user styles, plus 1 chosen clothing item. "
            "For each of the 3 styles, recommend 2 additional items for that chosen clothing item, to make a cohesive outfit. "
            "Each recommended item must have:\n"
            "- category: one of 'Tops', 'Bottoms', or 'Shoes'\n (case sensitive)"
            "- clothing_type\n"
            "- a short reasoning\n"
            "- name\n"
            "- color[]\n"
            "- material[]\n"
            "- other_tags[]\n\n"
            "The top-level JSON must have 'outfits'. Each outfit has 'style', 'description', 'name', 'items'. The outfit name should be something cute!"
            "Each item in 'items' must have exactly these fields: "
            "[category, reason, name, color[], material[], other_tags[], clothing_type]."
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
    )

    user_message = {"role": "user", "content": user_message_content}
    messages = [system_message, user_message]

    try:
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=LooseStylingLLMResponse,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not completion or not completion.choices:
        raise HTTPException(status_code=500, detail="No response from OpenAI")

    loose_data = completion.choices[0].message.parsed
    data_dict = loose_data.dict()
    outfits = data_dict.get("outfits", [])

    final_outfits: List[OutfitSearchResult] = []

    for outfit in outfits:
        style = outfit.get("style", "")
        description = outfit.get("description", "")
        name = outfit.get("name", "")
        items = outfit.get("items", [])

        matched_items: List[OutfitSearchMatchedItem] = []
        for recommended_item in items:
            # Build the recommended item
            rec_input = RecommendedItemInput(
                name=recommended_item.get("name", ""),
                color=recommended_item.get("color", []),
                material=recommended_item.get("material", []),
                other_tags=recommended_item.get("other_tags", []),
                reason=recommended_item.get("reason", ""),
                category=recommended_item.get("category", "").capitalize(),
                clothing_type=recommended_item.get("clothing_type", "").lower(),
            )

            color_str = ",".join(rec_input.color) if rec_input.color else "nil"
            material_str = ",".join(rec_input.material) if rec_input.material else "nil"

            new_tag = ClothingTag(
                clothing_type=rec_input.clothing_type,
                color=color_str,
                material=material_str,
                other_tags=rec_input.other_tags
            )

            # Then get embedding
            embedding = clothing_tag_to_embedding(new_tag)
            # Use category-based filter
            recs = list(get_n_closest_with_filter(embedding, rec_input.category, 1))
            if recs:
                best = recs[0]
                match_item = CatalogItem(id=str(best["_id"]))
            else:
                match_item = CatalogItem(id="")

            matched_items.append(
                OutfitSearchMatchedItem(original=rec_input, match=match_item)
            )

        final_outfits.append(
            OutfitSearchResult(
                style=style,
                description=description,
                name=name,
                items=matched_items
            )
        )

    return OutfitSearchResponse(outfits=final_outfits)

@router.get("/shop/item/{item_id}")
def get_item_by_id(item_id: str):
    try:
        object_id = ObjectId(item_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid item_id format")

    item_doc = mongodb.catalogue.find_one({"_id": object_id})
    if not item_doc:
        raise HTTPException(status_code=404, detail="Item not found with given id")

    item_doc["id"] = str(item_doc["_id"])
    del item_doc["_id"]
    return item_doc

@router.post("/wardrobe/check-style-analysis")
async def check_style_analysis(current_user = Depends(get_current_user)):
    """
    Checks the user's wardrobe count. If we've never run analysis before
    (last_analysis_count == 0) OR they've added 5 or more items since last analysis,
    then run style analysis and style embedding.
    Otherwise, do nothing.
    """
    try:
        user_id = current_user["_id"]
        
        # 1) Count how many items are in the user's wardrobe
        total_count = mongodb.wardrobe.count_documents({"user_id": user_id})
        
        # 2) Fetch the user's doc and check last_analysis_count
        user_doc = mongodb.users.find_one({"_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found.")
        
        last_analysis_count = user_doc.get("last_analysis_count", 0)
        
        # 3) Decide if we need to run style analysis
        #    Condition 1: first-time analysis -> last_analysis_count == 0
        #    Condition 2: user added >=5 items since last analysis
        if last_analysis_count == 0 or (total_count - last_analysis_count >= 5):
            # Run style analysis
            style_analysis = await get_style_analysis(current_user)
            
            # Run style embedding
            _ = get_user_style_embedding(current_user)
            
            # Update last_analysis_count and set the refresh flag
            mongodb.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "last_analysis_count": total_count,
                        "recommendations_refresh_needed": True
                    }
                }
            )
            return {"message": "Style analysis performed. Recommendations refresh flagged."}
        
        else:
            return {"message": "No style analysis triggered. Not enough new items."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
