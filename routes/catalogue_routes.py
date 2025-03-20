# catalogue_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
import services.mongodb as mongodb
from services.openai import (
    ClothingTag,
    str_to_clothing_tag,
    clothing_tag_to_embedding,
    get_n_closest,
    ClothingTagEmbed,
    StyleAnalysisResponse,
    compile_tags_and_embeddings_for_item,
    get_all_catalogue_ids,
    WardrobeTag, 
    generate_base_catalogue_recommendation
)
from bson import ObjectId
from services.user import get_current_user
from typing import Optional, List
from pydantic import BaseModel, parse_obj_as, ValidationError
from services.mongodb import UserItem
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
import json
from datetime import datetime

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

# --------------------------
# Response Models
# --------------------------
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


# ----------------------------------------------------------------------------
# HELPER FUNCTION to run style analysis (calls OpenAI) without using a route
# ----------------------------------------------------------------------------
def run_style_analysis_logic(user_id: ObjectId) -> StyleAnalysisResponse:
    """
    Gathers the user's wardrobe items from MongoDB, calls OpenAI to analyze,
    and returns the resulting StyleAnalysisResponse (top 3 styles).
    """
    # 1) Gather the user's wardrobe items
    wardrobe_items = []
    items_cursor = mongodb.wardrobe.find({"user_id": user_id})
    for item in items_cursor:
        wardrobe_items.append({
            "name": item["name"],
            "category": item["category"],
            "tags": item["tags"]
        })

    if not wardrobe_items:
        raise HTTPException(status_code=400, detail="No wardrobe items found for the user.")

    # 2) Prepare the prompt/messages for OpenAI
    wardrobe_items_json = json.dumps(wardrobe_items)
    prompt = (
        "Based on my wardrobe items, suggest the top 3 styles that fit me, "
        "along with descriptions and reasoning. The output should be a JSON "
        "object matching the following schema:\n"
        "{\n"
        '  "top_styles": [\n'
        "    {\n"
        '      "style": "string",\n'
        '      "description": "string",\n'
        '      "reasoning": "string"\n'
        "    },\n"
        "    ... (2 more styles)\n"
        "  ]\n"
        "}"
    )
    messages = [
        {"role": "system", "content": "You are a fashion expert assistant."},
        {"role": "user", "content": f"{prompt}\nHere are my wardrobe items:\n{wardrobe_items_json}"}
    ]

    # 3) Call OpenAI with the prompt and parse the response into StyleAnalysisResponse
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        response_format=StyleAnalysisResponse,
    )

    # Return the parsed style analysis
    return completion.choices[0].message.parsed

# --------------------------
# Endpoint: GET /user/style-analysis
# --------------------------

@router.get("/user/style-analysis")
async def get_style_analysis(current_user: UserItem = Depends(get_current_user)):
    """
    Manually trigger style analysis. Calls the helper function run_style_analysis_logic.
    """
    try:
        style_analysis = run_style_analysis_logic(current_user["_id"])
        # Store the result in the user doc
        mongodb.users.update_one(
            {"_id": current_user["_id"]},
            {"$set": {"user_style": style_analysis.dict()}}
        )
        return style_analysis.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/wardrobe/check-style-analysis")
async def check_style_analysis(current_user: UserItem = Depends(get_current_user)):
    """
    Checks how many items are in the user's wardrobe and compares against last_analysis_count.
    If the user added >=5 new items since the last analysis (or if they've never run analysis before),
    runs style analysis and regenerates style recommendations. Otherwise, does nothing.
    """
    try:
        user_id = current_user["_id"]
        
        # 1) Count how many items are in the user's wardrobe
        total_count = mongodb.wardrobe.count_documents({"user_id": user_id})
        print(f"[DEBUG] Total wardrobe count: {total_count}")
        
        # 2) Fetch the user's document and check last_analysis_count
        user_doc = mongodb.users.find_one({"_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found.")
        
        last_analysis_count = user_doc.get("last_analysis_count", 0)
        print(f"[DEBUG] last_analysis_count: {last_analysis_count}")
        
        # 3) Decide if we need to run style analysis
        #    Condition: user added >=5 items since last analysis, or no prior analysis
        if (total_count - last_analysis_count >= 5) or (last_analysis_count == 0):
            print("[DEBUG] Running style analysis...")
            style_analysis = run_style_analysis_logic(user_id)

            # Build new style_recommendations from the style analysis
            new_recommendations = []
            for style_suggestion in style_analysis.top_styles:
                style_prompt = f"{style_suggestion.style}: {style_suggestion.description} {style_suggestion.reasoning}"
                clothing_tag = str_to_clothing_tag(style_prompt)
                tag_embed = clothing_tag_to_embedding(clothing_tag)
                new_recommendations.append({
                    'style_name': style_suggestion.style,
                    'clothing_tag': clothing_tag.dict(),
                    'tag_embed': tag_embed.dict()
                })

            print(f"[DEBUG] Generated {len(new_recommendations)} new style recommendations.")

            # Update the user document with new recommendations
            mongodb.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "style_recommendations": new_recommendations,
                        "recommendations_refresh_needed": False,
                        "last_analysis_count": total_count
                    }
                }
            )
            return {"message": "Style analysis performed. Recommendations updated."}
        
        # If we did not trigger new analysis
        return {"message": "No style analysis triggered. Not enough new items."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/shop/recommendations")
def get_recommendations(current_user: UserItem = Depends(get_current_user), n: int = 10):
    print("[DEBUG] Entering /shop/recommendations route")
    
    user_id = current_user["_id"]
    print(f"[DEBUG] Current user_id: {user_id}")

    # Fetch the user document from MongoDB
    user = mongodb.users.find_one({"_id": user_id})
    print(f"[DEBUG] Found user doc: {user}")

    # Extract style_recommendations (array of objects, each with a 'tag_embed')
    style_recommendations = user.get("style_recommendations", [])
    print(f"[DEBUG] style_recommendations: {style_recommendations}")

    # If empty, we cannot produce any recommendations
    if not style_recommendations:
        print("[DEBUG] style_recommendations is empty. Throwing 400 error.")
        raise HTTPException(status_code=400, detail="No style recommendations available.")

    # We will collect items from each style_recommendation until we reach 'n'
    recommendations = []
    item_ids = set()

    styles_count = len(style_recommendations)
    items_per_style = max(n // styles_count, 1)
    print(f"[DEBUG] styles_count: {styles_count}, n: {n}, items_per_style: {items_per_style}")

    # For each style_recommendation, run a vector search using its 'tag_embed'
    for i, style_rec in enumerate(style_recommendations, start=1):
        print(f"[DEBUG] style_rec #{i}: {style_rec}")

        # Extract the dict containing the embedding
        embed_dict = style_rec.get("tag_embed", {})
        if not embed_dict:
            print("[DEBUG] No 'tag_embed' in style_recommendation; skipping.")
            continue

        # Parse embed_dict into a ClothingTagEmbed
        tag_embed = parse_obj_as(ClothingTagEmbed, embed_dict)
        print("[DEBUG] Created ClothingTagEmbed from style_rec")

        # Call your get_n_closest function
        recs = list(get_n_closest(tag_embed, items_per_style))
        print(f"[DEBUG] get_n_closest returned {len(recs)} items for style #{i}")

        # Accumulate results
        for rec in recs:
            rec_id_str = str(rec["_id"])
            if rec_id_str in item_ids:
                print(f"[DEBUG] Already have item {rec_id_str}, skipping.")
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
            print(f"[DEBUG] Added item {rec_id_str} to recommendations. Total so far: {len(recommendations)}")

            if len(recommendations) >= n:
                print("[DEBUG] Reached 'n' items in recommendations, stopping.")
                break

        if len(recommendations) >= n:
            break

    print(f"[DEBUG] Final recommendations count: {len(recommendations)}")
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
            recs = list(get_n_closest(embedding, rec_input.category, 1))
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

# @router.post("/catalogue/generate-base-recommendations")
# async def generate_base_recommendations(
#     before_date: Optional[datetime] = Query(None, description="Process items where recommendations were created before this date"),
#     n: Optional[int] = Query(None, description="Number of items to process (all if not specified)"),
#     current_user: UserItem = Depends(get_current_user)
# ):
#     # Check permissions if necessary; example:
#     # if not current_user.get('is_admin', False):
#     #     raise HTTPException(status_code=403, detail="Not authorized")

#     # Build the query
#     if before_date:
#         query = {
#             "$or": [
#                 {"base_recommendations": {"$exists": False}},
#                 {"base_recommendations_created_at": {"$lt": before_date}}
#             ]
#         }
#     else:
#         query = {"base_recommendations": {"$exists": False}}

#     # Fetch items
#     items_cursor = mongodb.catalogue.find(query)
#     if n is not None and n > 0:
#         items_cursor = items_cursor.limit(n)

#     processed_count = 0

#     for item in items_cursor:
#         try:
#             # Construct WardrobeTag from item data
#             name = item.get('name', '')
#             category = item.get('category', '')
#             clothing_type = item.get('clothing_type', '')
#             color = item.get('color', '')
#             material = item.get('material', '')
#             other_tags = item.get('other_tags', [])

#             tags = []
#             if clothing_type:
#                 tags.append(clothing_type)
#             if color:
#                 tags.append(color)
#             if material:
#                 tags.append(material)
#             tags.extend([tag for tag in other_tags if tag])

#             wardrobe_tag = WardrobeTag(
#                 name=name,
#                 category=category,
#                 tags=tags
#             )

#             # Generate recommendations
#             recommendations = generate_base_catalogue_recommendation(wardrobe_tag)

#             # Prepare update data
#             update_data = {
#                 "base_recommendations": [rec.dict() for rec in recommendations],
#                 "base_recommendations_created_at": datetime.now()
#             }

#             # Update the catalogue item
#             mongodb.catalogue.update_one(
#                 {"_id": item["_id"]},
#                 {"$set": update_data}
#             )

#             processed_count += 1
#         except Exception as e:
#             print(f"Error processing item {item['_id']}: {str(e)}")
#             continue

#     return {"message": f"Successfully processed {processed_count} catalogue items."}


# import threading

# MAX_THREADS = 20  # Limit to avoid exceeding API rate limits

# def process_data(items):
#     """
#     Process and embed `base_recommendations` inside catalogue items.
#     Runs in multiple threads for parallel execution.
#     """
#     for item in items:
#         try:
#             base_recs = item.get("base_recommendations", [])
#             if not base_recs:
#                 continue  # Skip items with no recommendations

#             updated_recommendations = []
#             for recommendation in base_recs:
#                 clothing_tag = ClothingTag(
#                     clothing_type=recommendation.get("clothing_type", ""),
#                     color=recommendation.get("color", ""),
#                     material=recommendation.get("material", ""),
#                     other_tags=recommendation.get("other_tags", [])
#                 )

#                 # Generate embeddings
#                 embeds = clothing_tag_to_embedding(clothing_tag)

#                 # Store the embeddings directly as fields inside each recommendation object
#                 recommendation["clothing_type_embed"] = embeds.clothing_type_embed
#                 recommendation["color_embed"] = embeds.color_embed
#                 recommendation["material_embed"] = embeds.material_embed
#                 recommendation["other_tags_embed"] = embeds.other_tags_embed
#                 updated_recommendations.append(recommendation)

#             # Update MongoDB with modified recommendations
#             mongodb.catalogue.update_one(
#                 {"_id": item["_id"]},
#                 {"$set": {"base_recommendations": updated_recommendations}}
#             )

#         except Exception as e:
#             print(f"❌ Error processing item {item['_id']}: {str(e)}")


# @router.post("/catalogue/embed-base-recommendations")
# async def embed_base_recommendations(
#     force_recompute: bool = Query(False, description="If True, force re-embedding of all items even if embeddings already exist"),
#     n: Optional[int] = Query(None, description="Number of items to process (default: all)"),
# ):
#     """
#     Finds catalogue items with `base_recommendations` and generates embeddings
#     for each recommended item, inserting them **directly** as new fields.

#     - Uses Python's `threading` module to run 20 threads for parallel processing.
#     - If `force_recompute=True`, it ignores existing embeddings and re-embeds everything.
#     - If `n` is provided, it limits the number of items processed.
#     """

#     # Query for relevant items
#     query_filters = {"base_recommendations": {"$exists": True, "$ne": []}}

#     if not force_recompute:
#         # Only process items missing embeddings inside `base_recommendations`
#         query_filters["base_recommendations.clothing_type_embed"] = {"$exists": False}

#     items_cursor = mongodb.catalogue.find(query_filters)

#     if n is not None and n > 0:
#         items_cursor = items_cursor.limit(n)

#     items_to_process = list(items_cursor)  # Convert cursor to list
#     total_items = len(items_to_process)

#     # If no items to process, return immediately
#     if total_items == 0:
#         return {"message": "No items require embedding."}

#     print(f"🔄 Starting embedding for {total_items} items using {MAX_THREADS} threads...")

#     # Create threads for parallel processing
#     threads = []
#     chunk_size = total_items // MAX_THREADS  # Split data into chunks

#     for i in range(MAX_THREADS):
#         start = i * chunk_size
#         end = (i + 1) * chunk_size if i < MAX_THREADS - 1 else total_items  # Last thread takes remaining items
#         thread = threading.Thread(target=process_data, args=(items_to_process[start:end],))
#         threads.append(thread)
#         thread.start()

#     # Wait for all threads to finish
#     for thread in threads:
#         thread.join()

#     print(f"🎉 Finished processing {total_items} items using {MAX_THREADS} threads.")
#     return {"message": f"Processed {total_items} items with embeddings stored directly as fields inside base_recommendations."}


@router.post("/catalogue/update-tags-embeddings/{item_id}")
def update_tags_for_one_item(item_id: str):
    """
    Updates tags & embeddings for a single catalogue item by ID.
    Pulls all tags from the item’s `other_tags` & `base_recommendations`,
    and inserts them into `tag_embeddings`.
    """
    inserted_count = compile_tags_and_embeddings_for_item(item_id)
    return {
        "message": f"Processed item {item_id}. Inserted {inserted_count} new tags into 'tag_embeddings'."
    }

# @router.post("/catalogue/update-tags-embeddings-all")
# def update_tags_for_all_items():
#     """
#     Updates tags & embeddings for *all* items in the catalogue.
#     Iterates over every item ID and calls `compile_tags_and_embeddings_for_item`.
#     """
#     all_ids = get_all_catalogue_ids()
#     total_inserted = 0
#     for cid in all_ids:
#         inserted_count = compile_tags_and_embeddings_for_item(cid)
#         total_inserted += inserted_count

#     return {
#         "message": f"Processed {len(all_ids)} items in catalogue. Inserted {total_inserted} new tags total."
#     }


def store_combined_embedding_for_item(item_id: ObjectId):
    """
    Loads an item, computes a 'combined_embed', and updates the doc.
    
    Assumes the item already has these fields:
      - clothing_type_embed
      - color_embed
      - material_embed
      - other_tags_embed
    """
    item = mongodb.catalogue.find_one({"_id": item_id})
    if not item:
        return

    ctype = item.get("clothing_type_embed", [])
    color = item.get("color_embed", [])
    material = item.get("material_embed", [])
    others = item.get("other_tags_embed", [])

    combined = build_combined_embedding(
        ctype, color, material, others,
        w_ctype=1.0,   # Heaviest weight
        w_color=0.2,
        w_material=0.2,
        w_others=0.2
    )

    mongodb.catalogue.update_one(
        {"_id": item_id},
        {"$set": {"combined_embed": combined}}
    )

# @router.post("/catalogue/update-combined-embeds-all")
# def update_combined_embeddings_for_all():
#     """
#     Iterates over every item in the 'catalogue' collection, 
#     computes its combined_embed, and updates the item.
#     Returns how many items were processed.
#     """
#     # Optionally apply a filter if you only want to process items that do not yet have combined_embed
#     # filter_query = {"combined_embed": {"$exists": False}}
#     filter_query = {}

#     items_cursor = mongodb.catalogue.find(filter_query, {"_id": 1})
#     count = 0
#     for doc in items_cursor:
#         store_combined_embedding_for_item(doc["_id"])
#         count += 1

#     return {"message": f"Processed {count} items, updated combined_embed on each."}

def build_combined_embedding(
    clothing_type_embed: List[float],
    color_embed: List[float],
    material_embed: List[float],
    other_tags_embed: List[List[float]],
    w_ctype: float = 1.0,
    w_color: float = 0.2,
    w_material: float = 0.2,
    w_others: float = 0.2
) -> List[float]:
    """
    Produces a single vector that merges all sub-embeddings using a weighted sum.
    (All embeddings must be of the same dimension.)
    """
    # If any of the first three embeddings are missing, assume a zero vector.
    # We try to get dimension from one that is non-empty.
    dim = None
    for vec in (clothing_type_embed, color_embed, material_embed):
        if vec:
            dim = len(vec)
            break
    # If still no dimension found but other_tags_embed exists, use its first vector's length.
    if dim is None and other_tags_embed:
        dim = len(other_tags_embed[0])
    if dim is None:
        # Fallback: cannot determine dimension
        raise ValueError("No embedding data provided to determine dimension.")

    # If any of the three are empty, replace them with a zero vector of length dim.
    if not clothing_type_embed:
        clothing_type_embed = [0.0] * dim
    if not color_embed:
        color_embed = [0.0] * dim
    if not material_embed:
        material_embed = [0.0] * dim

    # Sum up other_tags vectors.
    combined_others = [0.0] * dim
    if other_tags_embed:
        for vec in other_tags_embed:
            # Assume each vector is of length dim.
            for i in range(dim):
                combined_others[i] += vec[i]
        for i in range(dim):
            combined_others[i] *= w_others

    combined = [0.0] * dim
    for i in range(dim):
        combined[i] = (clothing_type_embed[i] * w_ctype +
                       color_embed[i] * w_color +
                       material_embed[i] * w_material +
                       combined_others[i])
    return combined

def build_other_tags_only_embedding(other_tags_embed: List[List[float]], dim: int, w_others: float = 0.2) -> List[float]:
    """
    Builds an embedding using only the other_tags vectors.
    If no vectors are provided, returns a zero vector of length 'dim'.
    """
    if not other_tags_embed:
        return [0.0] * dim
    combined = [0.0] * dim
    for vec in other_tags_embed:
        for i in range(dim):
            combined[i] += vec[i]
    for i in range(dim):
        combined[i] *= w_others
    return combined

@router.get("/fast-item-outfit-search-with-style")
def fast_item_outfit_search_with_style(
    item_id: str,
    current_user: UserItem = Depends(get_current_user),
    top_k: int = 20,
    candidate_pool: int = 100
):
    """
    Similar to /fast-item-outfit-search but now incorporates each of the user's
    style_recommendations. For each style, we ignore (do not add) the style’s 
    clothing_type, color, and material embeddings—instead, we only add the 
    style's other_tags_embed to the base recommendation's combined embed.
    The response is grouped by style.
    """

    def merge_vectors(vecA: List[float], vecB: List[float]) -> List[float]:
        """Element-wise sum of two vectors of equal length."""
        if len(vecA) != len(vecB):
            # In a production system, you might handle this more gracefully.
            raise ValueError("Vector dimensions do not match")
        return [a + b for a, b in zip(vecA, vecB)]

    # 1) Validate item_id
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID format")

    # 2) Fetch the main item
    item = mongodb.catalogue.find_one({"_id": object_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    category = item.get("category", "")
    # 3) Ensure base_recommendations exist (generate if missing)
    if "base_recommendations" not in item or not item["base_recommendations"]:
        clothing_type = item.get("clothing_type", "")
        color = item.get("color", "")
        material = item.get("material", "")
        tags = item.get("other_tags", [])

        wtag = WardrobeTag(
            name=item.get("name", ""),
            category=item.get("category", ""),
            tags=[clothing_type, color, material, *tags]
        )

        base_recs = generate_base_catalogue_recommendation(wtag)
        updated_recs = []
        for rec in base_recs:
            embeds = clothing_tag_to_embedding(rec)
            rec_dict = rec.dict()
            rec_dict.update({
                "clothing_type_embed": embeds.clothing_type_embed,
                "color_embed": embeds.color_embed,
                "material_embed": embeds.material_embed,
                "other_tags_embed": embeds.other_tags_embed
            })
            updated_recs.append(rec_dict)

        mongodb.catalogue.update_one(
            {"_id": object_id},
            {"$set": {"base_recommendations": updated_recs}}
        )
        item["base_recommendations"] = updated_recs

    # 4) Build a combined embed for each base recommendation
    base_recommendations = item["base_recommendations"]
    base_recs_with_combined = []
    for rec in base_recommendations:
        required = ["clothing_type_embed", "color_embed", "material_embed", "other_tags_embed"]
        if not all(k in rec for k in required):
            continue

        base_combined = build_combined_embedding(
            clothing_type_embed=rec["clothing_type_embed"],
            color_embed=rec["color_embed"],
            material_embed=rec["material_embed"],
            other_tags_embed=rec["other_tags_embed"],
            w_ctype=1.1,
            w_color=0.2,
            w_material=0.2,
            w_others=0.2
        )
        base_recs_with_combined.append({
            "base_recommendation": {
                "clothing_type": rec.get("clothing_type", ""),
                "color": rec.get("color", ""),
                "material": rec.get("material", ""),
                "other_tags": rec.get("other_tags", []),
            },
            "base_combined_embed": base_combined
        })

    if not base_recs_with_combined:
        raise HTTPException(status_code=500, detail="No valid base recommendation embeddings found.")

    # Determine default dimension from the first base recommendation's combined embed
    default_dim = len(base_recs_with_combined[0]["base_combined_embed"])

    # 5) Fetch user's style_recommendations
    user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    style_recs = user_doc.get("style_recommendations", [])
    if not style_recs:
        raise HTTPException(status_code=400, detail="No style_recommendations found for this user.")

    # 6) For each style, build a "style-only" embed from only other_tags_embed
    styles_output = []
    for style_rec in style_recs:
        style_name = style_rec.get("style_name", "Unknown Style")
        embed_dict = style_rec.get("tag_embed", {})
        # We now ignore clothing_type_embed, color_embed, and material_embed from the style.
        other_tags_embed = embed_dict.get("other_tags_embed", [])
        if not other_tags_embed:
            # If nothing available, use a zero vector of the default dimension.
            style_only_embed = [0.0] * default_dim
        else:
            style_only_embed = build_other_tags_only_embedding(other_tags_embed, default_dim, w_others=0.4) # weight of style embedding

        # 7) For each base recommendation, merge the base combined embed with the style-only embed,
        # then run a vector search.
        style_outfits = []
        for base_obj in base_recs_with_combined:
            base_embed = base_obj["base_combined_embed"]
            merged_embed = merge_vectors(base_embed, style_only_embed)

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "combined_embed_index",
                        "path": "combined_embed",
                        "queryVector": merged_embed,
                        "filter": {"category": {"$ne": category}},  # Exclude items with the same category
                        "limit": candidate_pool,
                        "numCandidates": candidate_pool * 10
                    }
                },
                {"$match": {"_id": {"$ne": ObjectId(item_id)}}},
                {
                    "$project": {
                        "_id": 1,
                        "name": 1,
                        "category": 1,
                        "price": 1,
                        "image_url": 1,
                        "product_url": 1,
                        "clothing_type": 1,
                        "color": 1,
                        "material": 1,
                        "other_tags": 1,
                        "score": {"$meta": "vectorSearchScore"},
                        "cropped_image_url": 1
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": top_k}
            ]

            cursor = mongodb.catalogue.aggregate(pipeline)
            top_items = list(cursor)

            style_outfits.append({
                "base_recommendation": base_obj["base_recommendation"],
                "top_items": [
                    {
                        "id": str(doc["_id"]),
                        "name": doc.get("name", ""),
                        "category": doc.get("category", ""),
                        "price": doc.get("price", ""),
                        "image_url": doc.get("image_url", ""),
                        "product_url": doc.get("product_url", ""),
                        "clothing_type": doc.get("clothing_type", ""),
                        "color": doc.get("color", ""),
                        "material": doc.get("material", ""),
                        "other_tags": doc.get("other_tags", []),
                        "score": doc["score"],
                        "cropped_image_url": doc.get("cropped_image_url", ""),
                    }
                    for doc in top_items
                ]
            })

        styles_output.append({
            "style_name": style_name,
            "style_outfits": style_outfits
        })

    return {
        "item_id": item_id,
        "name": item.get("name", ""),
        "styles": styles_output
    }

@router.get("/fast-item-outfit-search-with-style-using-get-n-closest")
def fast_item_outfit_search_with_style_using_get_n_closest(
    item_id: str,
    current_user: UserItem = Depends(get_current_user),
    top_k: int = 6,
):
    # 1) Validate item_id
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID format")

    # 2) Fetch the main item
    item = mongodb.catalogue.find_one({"_id": object_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    category = item.get("category", "")
    
    # 3) Ensure base_recommendations exist (same as original implementation)
    if "base_recommendations" not in item or not item["base_recommendations"]:
        clothing_type = item.get("clothing_type", "")
        color = item.get("color", "")
        material = item.get("material", "")
        tags = item.get("other_tags", [])

        wtag = WardrobeTag(
            name=item.get("name", ""),
            category=item.get("category", ""),
            tags=[clothing_type, color, material, *tags]
        )

        base_recs = generate_base_catalogue_recommendation(wtag)
        updated_recs = []
        for rec in base_recs:
            embeds = clothing_tag_to_embedding(rec)
            rec_dict = rec.dict()
            rec_dict.update({
                "clothing_type_embed": embeds.clothing_type_embed,
                "color_embed": embeds.color_embed,
                "material_embed": embeds.material_embed,
                "other_tags_embed": embeds.other_tags_embed
            })
            updated_recs.append(rec_dict)

        mongodb.catalogue.update_one(
            {"_id": object_id},
            {"$set": {"base_recommendations": updated_recs}}
        )
        item["base_recommendations"] = updated_recs

    # 4) Fetch user's style recommendations
    user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    style_recs = user_doc.get("style_recommendations", [])
    if not style_recs:
        raise HTTPException(status_code=400, detail="No style_recommendations found for this user.")

    # 5) Process styles and base recommendations
    styles_output = []
    for style_rec in style_recs:
        style_name = style_rec.get("style_name", "Unknown Style")
        style_tags = style_rec.get("tag_embed", {}).get("other_tags", [])
        
        style_outfits = []
        for base_rec in item["base_recommendations"]:
            # Create combined clothing tag
            combined_tag = ClothingTag(
                clothing_type=base_rec.get("clothing_type", ""),
                color=base_rec.get("color", ""),
                material=base_rec.get("material", ""),
                other_tags=base_rec.get("other_tags", []) + style_tags
            )
            
            # Get embedding for the combined tag
            tag_embed = clothing_tag_to_embedding(combined_tag)
            
            # Get closest items using the new method
            closest_items = get_n_closest(
                tag_embed=tag_embed,
                n=top_k,
                category_requirement=category,  # Category to exclude
                exclude_category=True,  # New flag for exclusion
                gender_requirements=["M", "F", "U"]
            )
            
            style_outfits.append({
                "base_recommendation": {
                    "clothing_type": base_rec.get("clothing_type", ""),
                    "color": base_rec.get("color", ""),
                    "material": base_rec.get("material", ""),
                    "other_tags": base_rec.get("other_tags", []),
                },
                "top_items": [{
                    "id": str(doc["_id"]),
                    "name": doc.get("name", ""),
                    "category": doc.get("category", ""),
                    "price": doc.get("price", ""),
                    "image_url": doc.get("image_url", ""),
                    "product_url": doc.get("product_url", ""),
                    "clothing_type": doc.get("clothing_type", ""),
                    "color": doc.get("color", ""),
                    "material": doc.get("material", ""),
                    "other_tags": doc.get("other_tags", []),
                    "score": doc.get("other_tag_match_count", 0),
                    "cropped_image_url": doc.get("cropped_image_url", ""),
                } for doc in closest_items]
            })

        styles_output.append({
            "style_name": style_name,
            "style_outfits": style_outfits
        })

    return {
        "item_id": item_id,
        "name": item.get("name", ""),
        "styles": styles_output
    }