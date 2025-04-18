# catalogue_routes.py
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
import services.mongodb as mongodb
from services.openai import (
    ClothingTag,
    style_to_clothing_tag,
    clothing_tag_to_embedding,
    get_n_closest,
    ClothingTagEmbed,
    StyleAnalysisResponse,
    compile_tags_and_embeddings_for_item,
    get_all_catalogue_ids,
    WardrobeTag,
    generate_base_catalogue_recommendation,
    get_n_closest_no_other_tags,
    get_user_feedback_recommendation
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
import time
import random
from services.image import store_blob, get_blob_url, LONG_EXPIRY

router = APIRouter()

def merge_vectors(vecA: List[float], vecB: List[float]) -> List[float]:
    if not vecA or not vecB: # Handle empty vectors gracefully
        return vecA or vecB or []
    if len(vecA) != len(vecB):
        # You might want to log this error instead of raising for production robustness
        print(f"Warning: Vector dimensions do not match during merge. Len A: {len(vecA)}, Len B: {len(vecB)}. Returning Vec A.")
        # Decide on fallback: return vecA, vecB, or raise error
        return vecA # Or handle more gracefully depending on requirements
        # raise ValueError("Vector dimensions do not match")
    return [a + b for a, b in zip(vecA, vecB)]

@router.get("/shop/items")
def get_items_by_retailer(
    retailer: str = None,
    include_embeddings: bool = False,
    limit: int = 0,
    gender: str = None  # Added gender parameter
):
    try:
        filter_criteria = {}
        if retailer:
            filter_criteria["retailer"] = retailer

        # Add gender filter if provided
        if gender:
            # Check if gender is valid
            if gender not in ['M', 'F', 'U']:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid gender value. Must be 'M', 'F', or 'U'."
                )

            # Filter by gender (if unisex is selected, include unisex items only)
            # If M is selected, show M and U items, same for F
            if gender == 'U':
                filter_criteria["gender"] = 'U'
            else:
                filter_criteria["$or"] = [{"gender": gender}, {"gender": "U"}]

        projection = {
            "name": 1,
            "category": 1,
            "price": 1,
            "image_url": 1,
            "product_url": 1,
            "clothing_type": 1,
            "color": 1,
            "material": 1,
            "other_tags": 1,
            "gender": 1  # Ensure gender is included in response
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
                "other_tags": item.get("other_tags", ""),
                "gender": item.get("gender", "")  # Default to 'U' if not specified
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


@router.get("/shop/similar_items")
def get_similar_items(id: str, n: int = 5, current_user: UserItem = Depends(get_current_user)):
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

        gender_requirement = ['U', 'F', 'M']

        if "userdefined_profile" in current_user and "gender" in current_user["userdefined_profile"]:
            user_gender = current_user["userdefined_profile"]["gender"]
            if user_gender == 'Female':
                gender_requirement.remove('M')
            elif user_gender == 'Male':
                gender_requirement.remove('F')

        recs = list(get_n_closest(tag_embed, n + 1, gender_requirements=gender_requirement))

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
        # 2) Fetch the user's document and check last_analysis_count
        user_doc = mongodb.users.find_one({"_id": user_id})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found.")

        last_analysis_count = user_doc.get("last_analysis_count", 0)
        # 3) Decide if we need to run style analysis
        #    Condition: user added >=5 items since last analysis, or no prior analysis
        if (total_count - last_analysis_count >= 5) or (last_analysis_count == 0):
            style_analysis = run_style_analysis_logic(user_id)

            # Build new style_recommendations from the style analysis
            new_recommendations = []
            for style_suggestion in style_analysis.top_styles:
                style_prompt = f"{style_suggestion.style}: {style_suggestion.description} {style_suggestion.reasoning}"
                clothing_tag = style_to_clothing_tag(style_prompt)
                tag_embed = clothing_tag_to_embedding(clothing_tag)
                new_recommendations.append({
                    'style_name': style_suggestion.style,
                    'clothing_tag': clothing_tag.dict(),
                    'tag_embed': tag_embed.dict()
                })

            # Update the user document with new recommendations
            mongodb.users.update_one(
                {"_id": user_id},
                {
                    "$set": {
                        "style_recommendations": new_recommendations,
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
def get_recommendations(current_user: UserItem = Depends(get_current_user), n: int = 25):
    user_id = current_user["_id"]
    # Fetch the user document from MongoDB
    user = mongodb.users.find_one({"_id": user_id})
    # Extract style_recommendations (array of objects, each with a 'tag_embed')
    style_recommendations = user.get("style_recommendations", [])
    # If empty, we cannot produce any recommendations
    if not style_recommendations:
        raise HTTPException(status_code=400, detail="No style recommendations available.")

    # We will collect items from each style_recommendation until we reach 'n'
    recommendations = []
    item_ids = set()

    styles_count = len(style_recommendations)
    items_per_style = max(n // styles_count, 1)
    # For each style_recommendation, run a vector search using its 'tag_embed'
    for i, style_rec in enumerate(style_recommendations, start=1):
        # Extract the dict containing the embedding
        embed_dict = style_rec.get("tag_embed", {})
        if not embed_dict:
            continue

        # Parse embed_dict into a ClothingTagEmbed
        tag_embed = parse_obj_as(ClothingTagEmbed, embed_dict)
        # Call your get_n_closest function
        recs = list(get_n_closest(tag_embed, items_per_style))
        # Accumulate results
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


@router.get("/shop/recommendations-fast")
def get_recommendations_fast(
    current_user: UserItem = Depends(get_current_user),
    n: int = 25,
    category: Optional[str] = Query(None, description="Filter by category (e.g., 'Tops', 'Bottoms', 'Shoes')"),
    gender: Optional[str] = Query(None, description="Filter by gender (M, F, or U)"),
    candidate_pool: int = 100
):
    """
    Returns style-based recommendations using a single, combined embedding vector
    created by merging all the user's style preferences together.

    - n: Total number of recommendations to return
    - category: Optional category filter
    - gender: Optional gender filter (M, F, or U)
    - candidate_pool: Size of initial vector search results (before filtering)
    """
    start_time = time.perf_counter()
    timings = {}

    # 1) Fetch user's style recommendations
    user_fetch_start = time.perf_counter()
    user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")

    style_recs = user_doc.get("style_recommendations", [])
    if not style_recs:
        raise HTTPException(status_code=400, detail="No style recommendations found for this user.")
    timings["fetch_user"] = time.perf_counter() - user_fetch_start

    # 2) Extract all style names and combine all style embeddings
    style_names = []
    combined_vector = None
    vector_dim = None

    # First, determine the embedding dimension
    combine_start = time.perf_counter()
    for style_rec in style_recs:
        style_names.append(style_rec.get("style_name", "Unknown Style"))

        # Extract other_tags_embed (this is what we want to combine)
        embed_dict = style_rec.get("tag_embed", {})
        other_tags_embed = embed_dict.get("other_tags_embed", [])

        # If we find a non-empty embedding, get its dimension
        if other_tags_embed and not vector_dim:
            vector_dim = len(other_tags_embed[0]) if other_tags_embed else 0

    # If we couldn't determine the dimension, return an error
    if not vector_dim:
        raise HTTPException(status_code=500, detail="Could not determine embedding dimension")

    # Initialize a zero vector with the correct dimension
    combined_vector = [0.0] * vector_dim

    # Now combine all embeddings by summing them together
    for style_rec in style_recs:
        embed_dict = style_rec.get("tag_embed", {})
        other_tags_embed = embed_dict.get("other_tags_embed", [])

        if other_tags_embed:
            # Sum all vectors in other_tags_embed
            for vec in other_tags_embed:
                for i in range(vector_dim):
                    combined_vector[i] += vec[i]

    # Normalize the combined vector to maintain consistent magnitudes
    # Skip if the vector is all zeros
    if any(combined_vector):
        magnitude = sum(x**2 for x in combined_vector) ** 0.5
        if magnitude > 0:
            combined_vector = [x/magnitude for x in combined_vector]

    timings["combine_embeddings"] = time.perf_counter() - combine_start

    # 3) Build filter criteria
    filter_criteria = {}
    if category:
        filter_criteria["category"] = category

    # Add gender filter if provided
    if gender:
        # Check if gender is valid
        if gender not in ['M', 'F', 'U']:
            raise HTTPException(
                status_code=400,
                detail="Invalid gender value. Must be 'M', 'F', or 'U'."
            )

        # Filter by gender (if unisex is selected, include unisex items only)
        # If M is selected, show M and U items, same for F
        if gender == 'U':
            filter_criteria["gender"] = 'U'
        else:
            filter_criteria["$or"] = [{"gender": gender}, {"gender": "U"}]

    # 4) Perform a single vector search with the combined embedding
    search_start = time.perf_counter()

    # Build the vector search pipeline - only include filter if we have criteria
    vector_search_params = {
        "index": "combined_embed_index",
        "path": "combined_embed",
        "queryVector": combined_vector,
        "limit": candidate_pool,
        "numCandidates": candidate_pool * 10
    }

    # Only add filter if we have filter criteria
    if filter_criteria:
        vector_search_params["filter"] = filter_criteria

    pipeline = [
        {
            "$vectorSearch": vector_search_params
        },
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
                "cropped_image_url": 1,
                "gender": 1  # Include gender in projection
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": n}
    ]

    cursor = mongodb.catalogue.aggregate(pipeline)
    search_results = list(cursor)
    timings["search_combined"] = time.perf_counter() - search_start

    # 5) Process results and shuffle
    process_start = time.perf_counter()
    recommendations = []

    for item in search_results:
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
            "other_tags": item.get("other_tags", ""),
            "score": item.get("score", 0),
            "cropped_image_url": item.get("cropped_image_url", ""),
            "gender": item.get("gender", "U")  # Include gender in the response
        }
        recommendations.append(item_data)

    # Shuffle recommendations for variety
    import random
    random.shuffle(recommendations)

    timings["process_results"] = time.perf_counter() - process_start

    # Log performance metrics
    total_time = time.perf_counter() - start_time
    print(f"===== Fast Recommendations Performance =====")
    print(f"Total time: {total_time:.4f} seconds")
    for step, elapsed in timings.items():
        print(f"  {step}: {elapsed:.4f} seconds")

    return {
        "styles": style_names,
        "recommendations": recommendations,
        "total_items": len(recommendations)
    }


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
    top_k: int = 10,
    candidate_pool: int = 50
):

    # STEP 1) Validate item_id
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item ID format")

    # STEP 2) Fetch the main item
    item = mongodb.catalogue.find_one({"_id": object_id})
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    category = item.get("category", "")
    # STEP 3) Ensure base_recommendations exist (generate if missing)
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

    # STEP 4) Build a combined embed for each base recommendation
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
    default_dim = len(base_recs_with_combined[0]["base_combined_embed"])
    user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found.")
    style_recs = user_doc.get("style_recommendations", [])
    if not style_recs:
        raise HTTPException(status_code=400, detail="No style_recommendations found for this user.")
    styles_output = []

    for style_rec in style_recs:
        style_name = style_rec.get("style_name", "Unknown Style")
        embed_dict = style_rec.get("tag_embed", {})

        # Only other_tags_embed is used from the style
        other_tags_embed = embed_dict.get("other_tags_embed", [])
        if not other_tags_embed:
            style_only_embed = [0.0] * default_dim
        else:
            style_only_embed = build_other_tags_only_embedding(
                other_tags_embed, default_dim, w_others=0.4
            )

        style_outfits = []
        for base_obj in base_recs_with_combined:
            # --- TIME the vector merging ---
            merge_start = time.perf_counter()
            base_embed = base_obj["base_combined_embed"]
            merged_embed = merge_vectors(base_embed, style_only_embed)
            merge_time = time.perf_counter() - merge_start

            # --- TIME the vector search ---
            search_start = time.perf_counter()
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "combined_embed_index",
                        "path": "combined_embed",
                        "queryVector": merged_embed,
                        "filter": {"category": {"$ne": category}},  # Exclude same-category
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
            search_time = time.perf_counter() - search_start

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

    # Return the final data without timings
    return {
        "item_id": item_id,
        "name": item.get("name", ""),
        "styles": styles_output,
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


@router.get("/fast-item-outfit-search-with-style-stream")
def fast_item_outfit_search_with_style_stream(
    item_id: str,
    current_user: UserItem = Depends(get_current_user),
    top_k: int = 10,
    candidate_pool: int = 50,
    gender: Optional[str] = Query(None, description="Filter by gender (M, F, or U)")
) -> StreamingResponse:
    """
    Example streaming version (SSE) of the outfit search. 
    Returns partial results for each style as soon as they're ready.
    Optional gender filter can be applied to only show items appropriate for that gender.
    """

    def merge_vectors(vecA: List[float], vecB: List[float]) -> List[float]:
        if len(vecA) != len(vecB):
            raise ValueError("Vector dimensions do not match")
        return [a + b for a, b in zip(vecA, vecB)]

    def sse_event_generator():
        yield "data: {\"test\":true, \"message\":\"Connection established\"}\n\n"
        # 1) Validate item_id
        try:
            object_id = ObjectId(item_id)
        except:
            raise HTTPException(status_code=400, detail="Invalid item_id format")

        # 2) Fetch the item
        item = mongodb.catalogue.find_one({"_id": object_id})
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        category = item.get("category", "")

        # 3) Ensure base_recommendations exist (or generate if needed)
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

        base_recommendations = item["base_recommendations"]
        if not base_recommendations:
            raise HTTPException(
                status_code=500,
                detail="No base_recommendations for this item"
            )

        # 4) Build combined embeds for each base recommendation
        base_recs_with_combined = []
        for rec in base_recommendations:
            if not all(k in rec for k in ("clothing_type_embed", "color_embed", "material_embed", "other_tags_embed")):
                continue

            base_combined = build_combined_embedding(
                clothing_type_embed=rec["clothing_type_embed"],
                color_embed=rec["color_embed"],
                material_embed=rec["material_embed"],
                other_tags_embed=rec["other_tags_embed"],
                w_ctype=0.3,
                w_color=0.2,
                w_material=0.2,
                w_others=0.4
            )
            base_recs_with_combined.append({
                "base_recommendation": {
                    "clothing_type": rec.get("clothing_type", ""),
                    "color": rec.get("color", ""),
                    "material": rec.get("material", ""),
                    "other_tags": rec.get("other_tags", [])
                },
                "base_combined_embed": base_combined
            })

        if not base_recs_with_combined:
            raise HTTPException(status_code=500, detail="No valid base rec embeddings")

        default_dim = len(base_recs_with_combined[0]["base_combined_embed"])

        # 5) Fetch user & style_recommendations
        user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
        style_recs = user_doc.get("style_recommendations", [])
        if not style_recs:
            raise HTTPException(status_code=400, detail="No style_recommendations found")

        # 6) For each style, build a partial result, then yield SSE data
        for style_idx, style_rec in enumerate(style_recs):
            style_name = style_rec.get("style_name", f"Style {style_idx+1}")
            embed_dict = style_rec.get("tag_embed", {})

            # We only use 'other_tags_embed' from style
            other_tags_embed = embed_dict.get("other_tags_embed", [])
            if not other_tags_embed:
                style_only_embed = [0.0] * default_dim
            else:
                style_only_embed = build_other_tags_only_embedding(
                    other_tags_embed, default_dim, w_others=0.4
                )

            # We'll combine each base embed with style embed, run the search, and gather top items
            style_outfits = []
            for base_obj in base_recs_with_combined:
                base_embed = base_obj["base_combined_embed"]
                merged_embed = merge_vectors(base_embed, style_only_embed)

                # Build filter criteria
                filter_criteria = {"category": {"$ne": category}}

                # Add gender filter if provided
                if gender:
                    # Check if gender is valid
                    if gender not in ['M', 'F', 'U']:
                        raise HTTPException(
                            status_code=400,
                            detail="Invalid gender value. Must be 'M', 'F', or 'U'."
                        )

                    # If unisex is selected, include unisex items only
                    # If M is selected, show M and U items, same for F
                    if gender == 'U':
                        filter_criteria["gender"] = 'U'
                    else:
                        filter_criteria["$or"] = [{"gender": gender}, {"gender": "U"}]

                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "combined_embed_index",
                            "path": "combined_embed",
                            "queryVector": merged_embed,
                            "filter": filter_criteria,
                            "limit": candidate_pool,
                            "numCandidates": candidate_pool * 10
                        }
                    },
                    {"$match": {"_id": {"$ne": ObjectId(item_id)}}},
                    {
                        "$project": {
                            "_id": 1, "name": 1, "category": 1, "price": 1,
                            "image_url": 1, "product_url": 1,
                            "clothing_type": 1, "color": 1, "material": 1,
                            "other_tags": 1, "score": {"$meta": "vectorSearchScore"},
                            "cropped_image_url": 1, "gender": 1
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
                            "gender": doc.get("gender", "U"),
                        }
                        for doc in top_items
                    ]
                })

            # Build partial result for this style
            partial_data = {
                "style_name": style_name,
                "style_index": style_idx,
                "style_outfits": style_outfits
            }

            # SSE chunk = "data: <json>\n\n"
            sse_chunk = "data: " + json.dumps(partial_data) + "\n\n"
            yield sse_chunk

        # After all styles, send a final "done" event
        done_chunk = "data: {\"done\":true}\n\n"
        yield done_chunk

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Prevents Nginx buffering if you're using it
        }
    )


@router.get("/user/gender")
async def get_user_gender(current_user: UserItem = Depends(get_current_user)):
    """
    Returns the user's gender in a format suitable for filtering products.
    Returns 'M' for Male, 'F' for Female, and null for "Prefer not to say".
    """
    try:
        user_id = current_user["_id"]
        user_doc = mongodb.users.find_one({"_id": user_id})

        if not user_doc or "userdefined_profile" not in user_doc:
            return {"gender_code": None}

        user_gender = user_doc.get("userdefined_profile", {}).get("gender", "")

        # Map user-friendly gender values to API codes
        gender_code = None
        if user_gender == "Male":
            gender_code = "M"
        elif user_gender == "Female":
            gender_code = "F"
        # For "Prefer not to say" or any other value, keep as None

        return {"gender_code": gender_code}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching user gender: {str(e)}"
        )


@router.get("/catalogue/feedback_recommendation")
async def get_feedback_recommendation(starting_id, previous_rec_id, dislike_reason: str, current_user: dict = Depends(get_current_user)):
    # previous_rec and starting should be the _id of the mongodb object for the previously rec clothing item and the starting item

    starting_mongodb_object = mongodb.catalogue.find_one({"_id": ObjectId(starting_id)})
    disliked_mongodb_object = mongodb.catalogue.find_one({"_id": ObjectId(previous_rec_id)})

    starting_item = WardrobeTag(name=starting_mongodb_object.get("name"), category=starting_mongodb_object.get(
        "category"), tags=list(starting_mongodb_object.get("other_tags")))
    disliked_item = WardrobeTag(name=disliked_mongodb_object.get("name"), category=disliked_mongodb_object.get(
        "category"), tags=list(disliked_mongodb_object.get("other_tags")))

    # formatting profile
    profile = current_user["userdefined_profile"]

    exclude_names = []

    if profile['gender'] == "Male":
        profile['gender'] = "M"
    else:
        profile['gender'] = "F"

    if profile['clothing_likes'] != None:
        if len(profile['clothing_likes'].keys()) < 5:
            profile['clothing_likes'] = list(profile['clothing_likes'].keys())
        # just keeping the latest 5 entries and keeping the datatype as list
        if len(profile['clothing_likes']) >= 5:
            profile['clothing_likes'] = list(profile['clothing_likes'].keys())[-1:-6:-1]
    else:
        profile['clothing_likes'] = []

    if profile['clothing_dislikes'] != None:
        # used to filter disliked items from the recommendation
        exclude_names = list(profile['clothing_dislikes'].keys())[1:]
        if len(profile['clothing_dislikes']) < 5:
            dislikes = profile['clothing_dislikes']['feedback']
        # get the last 5 dislikes (type:list, [category,item name, what they dislike(style/color/item), dislike reason])
        if len(profile['clothing_dislikes']) >= 5:
            dislikes = profile['clothing_dislikes']['feedback'][-1:-6:-1]

        # get just the dislike reason from the list and add it to another list with just the dislike reasons
        dislikes_list = []
        for item in dislikes:
            dislikes_list.append(item[3])

        profile['clothing_dislikes'] = dislikes_list

    else:
        profile['clothing_dislikes'] = []

    exclude_names.append(starting_mongodb_object.get('name'))

    recommended_ClothingTag = get_user_feedback_recommendation(starting_item, disliked_item, dislike_reason, profile)

    clothing_tag_embedded = clothing_tag_to_embedding(recommended_ClothingTag)
    rec = list(get_n_closest(clothing_tag_embedded, 1, exclude_names=exclude_names))[0]
    rec['_id'] = str(rec['_id'])

    # outputs the mongodb object for the clothing item from the catalogue as a dictionary.
    return rec


@router.post("/catalogue/track-click/{item_id}")
async def track_product_click(item_id: str, current_user: UserItem = Depends(get_current_user)):
    """
    Tracks a click on a product. Increments the click_count field in MongoDB.
    """
    try:
        try:
            object_id = ObjectId(item_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid item ID format")

        # Update document with atomic increment of click_count
        # If field doesn't exist yet, it will be created with value 1
        result = mongodb.catalogue.update_one(
            {"_id": object_id},
            {"$inc": {"click_count": 1}}
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Product not found")

        return {"success": True, "message": "Click tracked successfully"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error tracking product click: {str(e)}"
        )


@router.get("/shop/text-search")
async def text_search(
    query: str,
    gender: Optional[str] = Query(None, description="Filter by gender (M, F, or U)"),
    limit: int = Query(200, description="Maximum number of results to return"),
    current_user: UserItem = Depends(get_current_user)
):
    """
    Performs search on catalog items using MongoDB Atlas Search.
    Returns items sorted by search score.
    """
    try:
        # Create the pipeline for Atlas Search
        pipeline = []

        # Only add $search stage if query is provided
        if query and query.strip():
            # Add Atlas Search stage
            search_stage = {
                "$search": {
                    "index": "default",  # The Atlas Search index name
                    "text": {
                        "query": query,
                        "path": {
                            "wildcard": "*"  # Search in all indexed fields
                        },
                    }
                }
            }
            pipeline.append(search_stage)

            # Add score metadata to results
            pipeline.append({
                "$addFields": {
                    "score": {"$meta": "searchScore"}
                }
            })

        # Add gender filter if provided
        match_criteria = {}
        if gender:
            if gender not in ['M', 'F', 'U']:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid gender value. Must be 'M', 'F', or 'U'."
                )

            if gender == 'U':
                # Only show unisex items
                match_criteria["gender"] = 'U'
            else:
                # Show items that match gender OR are unisex
                match_criteria["$or"] = [{"gender": gender}, {"gender": "U"}]

        # Add $match stage if we have gender or non-search case
        if match_criteria or not query or not query.strip():
            pipeline.append({"$match": match_criteria})

        # Project only the fields we need
        pipeline.append({
            "$project": {
                "name": 1,
                "category": 1,
                "price": 1,
                "image_url": 1,
                "product_url": 1,
                "clothing_type": 1,
                "color": 1,
                "material": 1,
                "other_tags": 1,
                "gender": 1,
                "score": 1
            }
        })

        # Add sorting - by search score for search queries, or default sort for non-search
        if query and query.strip():
            pipeline.append({"$sort": {"score": -1}})
        else:
            # For non-search queries, sort by something reasonable like name
            pipeline.append({"$sort": {"name": 1}})

        # Add limit
        pipeline.append({"$limit": limit})

        # Execute the aggregation pipeline
        cursor = mongodb.catalogue.aggregate(pipeline)

        # Process results
        results = []
        for item in cursor:
            results.append({
                "id": str(item["_id"]),
                "name": item.get("name", ""),
                "category": item.get("category", ""),
                "price": item.get("price", ""),
                "image_url": item.get("image_url", ""),
                "product_url": item.get("product_url", ""),
                "clothing_type": item.get("clothing_type", ""),
                "color": item.get("color", ""),
                "material": item.get("material", ""),
                "other_tags": item.get("other_tags", []),
                "gender": item.get("gender", "U"),
                "score": item.get("score", 0) if query and query.strip() else None
            })

        # Debug log the filter criteria and result count
        print(f"Search query: '{query}'")
        print(f"Gender filter: {gender}")
        print(f"Match criteria: {match_criteria}")
        print(f"Found {len(results)} results")

        return results

    except Exception as e:
        print(f"Error in text_search: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )

@router.get("/catalogue/matching-wardrobe-items/{item_id}")
async def get_matching_wardrobe_items(
    item_id: str,
    current_user: UserItem = Depends(get_current_user),
    top_k: int = 3,
    candidate_pool: int = 30,
):
    start_time = time.time()
    print(f"Starting matching wardrobe items search for item: {item_id}")

    # STEP 1: Validate item_id and Fetch Shop Item
    try:
        object_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid shop item ID format")

    shop_item = mongodb.catalogue.find_one({"_id": object_id})
    if not shop_item:
        raise HTTPException(status_code=404, detail="Shop item not found")
    shop_item_category = shop_item.get("category", "")
    print(f"Shop item '{shop_item.get('name')}' fetched. Category: {shop_item_category}")

    # STEP 2: Ensure Base Recommendations Exist on Shop Item
    if "base_recommendations" not in shop_item or not shop_item["base_recommendations"]:
        print(f"Generating base recommendations for item: {item_id}")
        # Logic copied from fast-item-outfit-search-with-style-stream
        clothing_type = shop_item.get("clothing_type", "")
        color = shop_item.get("color", "")
        material = shop_item.get("material", "")
        tags = shop_item.get("other_tags", [])

        wtag = WardrobeTag(
            name=shop_item.get("name", ""),
            category=shop_item_category, # Use fetched category
            tags=[t for t in [clothing_type, color, material] if t] + tags # Filter out empty strings
        )

        try:
            base_recs_tags = generate_base_catalogue_recommendation(wtag)
            updated_recs_data = []
            for rec_tag in base_recs_tags:
                embeds = clothing_tag_to_embedding(rec_tag)
                rec_dict = rec_tag.dict()
                rec_dict.update({
                    "clothing_type_embed": embeds.clothing_type_embed,
                    "color_embed": embeds.color_embed,
                    "material_embed": embeds.material_embed,
                    "other_tags_embed": embeds.other_tags_embed
                })
                updated_recs_data.append(rec_dict)

            mongodb.catalogue.update_one(
                {"_id": object_id},
                {"$set": {"base_recommendations": updated_recs_data}}
            )
            shop_item["base_recommendations"] = updated_recs_data
            print(f"Base recommendations generated and saved for item: {item_id}")
        except Exception as e:
             print(f"Error generating/embedding base recommendations for {item_id}: {e}")
             raise HTTPException(status_code=500, detail=f"Could not generate base recommendations: {e}")

    base_recommendations = shop_item.get("base_recommendations", [])
    if not base_recommendations:
         # This should ideally not happen after the generation step, but check anyway
        raise HTTPException(status_code=500, detail="Failed to obtain base recommendations for the shop item.")
    
    # STEP 3: Build Combined Embeddings for Base Recs
    base_recs_with_combined = []
    default_dim = None
    print("Building combined embeddings for base recommendations...")
    for rec in base_recommendations:
        required = ["clothing_type_embed", "color_embed", "material_embed", "other_tags_embed"]
        if not all(k in rec and rec[k] for k in required):
            print(f"Skipping base rec due to missing embeds: {rec.get('clothing_type')}")
            continue
        try:
            base_combined = build_combined_embedding( # Weights from stream example
                rec.get("clothing_type_embed", []), rec.get("color_embed", []),
                rec.get("material_embed", []), rec.get("other_tags_embed", []),
                w_ctype=0.3, w_color=0.2, w_material=0.2, w_others=0.4
            )
            if not default_dim and base_combined: default_dim = len(base_combined)
            base_recs_with_combined.append({
                "base_recommendation": {k: rec.get(k) for k in ["clothing_type", "color", "material", "other_tags"]},
                "base_combined_embed": base_combined
            })
        except Exception as e:
            print(f"Error building combined embed for base rec: {e}")

    if not base_recs_with_combined or not default_dim:
        raise HTTPException(status_code=500, detail="Could not process base recommendations or determine dimension.")
    print(f"Built combined embeddings for {len(base_recs_with_combined)} base recs. Dim: {default_dim}")

    # STEP 4: Fetch User and Style Recommendations
    user_doc = mongodb.users.find_one({"_id": current_user["_id"]})
    if not user_doc: raise HTTPException(status_code=404, detail="User not found.")
    style_recs = user_doc.get("style_recommendations", [])
    if not style_recs: raise HTTPException(status_code=400, detail="User has no style recommendations.")
    print(f"Fetched {len(style_recs)} style recommendations for user.")

    # STEP 5: Iterate Through Styles, Base Recs, and Search Wardrobe
    styles_output = []
    total_searches = 0
    total_search_time = 0

    for style_idx, style_rec in enumerate(style_recs):
        style_name = style_rec.get("style_name", f"Style {style_idx+1}")
        print(f"\nProcessing Style: {style_name} ({style_idx+1}/{len(style_recs)})")
        embed_dict = style_rec.get("tag_embed", {})
        other_tags_embed = embed_dict.get("other_tags_embed", [])

        try:
            style_only_embed = build_other_tags_only_embedding(
                other_tags_embed, default_dim, w_others=0.4
            )
        except Exception as e:
             print(f"Error building style embedding for {style_name}: {e}. Skipping.")
             continue

        style_outfits = []
        for base_idx, base_obj in enumerate(base_recs_with_combined):
            base_rec_details = base_obj["base_recommendation"]
            base_embed = base_obj["base_combined_embed"]
            merged_embed = merge_vectors(base_embed, style_only_embed)

            if not merged_embed: continue # Skip if merge failed

            search_start_time = time.time()
            total_searches += 1

            wardrobe_filter = { "user_id": current_user["_id"] }
            if shop_item_category:
                 wardrobe_filter["category"] = {"$ne": shop_item_category}

            # ***** CORRECTED VECTOR SEARCH *****
            wardrobe_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "wardrobe_vector_index", # <-- VERIFY THIS NAME
                        "path": "embedding",              # <-- USE CORRECT PATH
                        "queryVector": merged_embed,
                        "filter": wardrobe_filter,
                        "limit": candidate_pool,
                        "numCandidates": candidate_pool * 10
                    }
                },
                { # Project needed fields, exclude embedding
                    "$project": {
                        "_id": 1, "name": 1, "category": 1, "tags": 1,
                        "image_name": 1, 
                        "score": {"$meta": "vectorSearchScore"}
                    }
                },
                {"$sort": {"score": -1}},
                {"$limit": top_k}
            ]
            # ***** END CORRECTION *****

            try:
                cursor = mongodb.wardrobe.aggregate(wardrobe_pipeline)
                top_wardrobe_items = list(cursor)
                search_duration = time.time() - search_start_time
                total_search_time += search_duration
                print(f"  - Searched for base rec '{base_rec_details.get('clothing_type')}': Found {len(top_wardrobe_items)} wardrobe items in {search_duration:.3f}s")

                formatted_items = []
                for doc in top_wardrobe_items:
                    image_name = doc.get('image_name')
                    image_url = None
                    if image_name:
                        try:
                            # Use expiry=15 (minutes) as specified
                            image_url = get_blob_url(image_name, LONG_EXPIRY)
                        except Exception as blob_err:
                            print(f"Error getting blob URL for {image_name}: {blob_err}")
                            # Decide: return None, or a placeholder URL?

                    formatted_items.append({
                        "id": str(doc["_id"]),
                        "name": doc.get("name", ""),
                        "category": doc.get("category", ""),
                        "tags": doc.get("tags", []), # Wardrobe items have 'tags' field
                        "image_url": image_url,      # Use the generated SAS URL
                        "image_name": image_name,    # Keep original name if needed
                        "score": doc.get("score", 0)
                    })

                style_outfits.append({
                    "base_recommendation": base_rec_details,
                    "matching_wardrobe_items": formatted_items
                })
            except Exception as e:
                 print(f"Error during wardrobe vector search aggregate: {e}")
                 style_outfits.append({
                     "base_recommendation": base_rec_details,
                     "matching_wardrobe_items": [],
                     "error": f"Search failed: {e}"
                 })

        styles_output.append({
            "style_name": style_name,
            "style_outfits": style_outfits
        })

    end_time = time.time()
    total_duration = end_time - start_time
    avg_search_time = total_search_time / total_searches if total_searches > 0 else 0
    print(f"\nFinished matching wardrobe items search. Total time: {total_duration:.3f}s")
    print(f"Performed {total_searches} vector searches. Avg search time: {avg_search_time:.3f}s")

    # STEP 6: Return Final JSON Response
    shop_item_response_data = {
            "id": item_id,
            "name": shop_item.get("name", ""),
            "category": shop_item_category,
            "image_url": shop_item.get("image_url", ""), # Use direct URL for shop item
            "product_url": shop_item.get("product_url", ""),
    }
    # Add _is_shop_item flag for frontend clarity, maybe not needed if frontend handles based on structure
    # shop_item_response_data["_is_shop_item"] = True

    return {
        "shop_item": shop_item_response_data,
        "matching_styles": styles_output
    }