# openai.py
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
from services.mongodb import catalogue, db, tag_embeddings
from services.metadata import get_catalogue_metadata
from pydantic import BaseModel
import random
import json
from bson import ObjectId
from typing import Dict, List, Optional, Literal, Tuple

# Hardcoded available categories
category_labels = ["Tops", "Bottoms", "Dresses", "Shoes", "Jackets", "Accessories"]

# Fixed formats which GPT4 will force to return


class ClothingTag(BaseModel):  # For catalogue
    clothing_type: str
    color: str
    material: str
    other_tags: list[str]

# store the recommendations given to the user, to allow for feedback to be given - then store the feedback to make next recommendations better
# I have no idea how to store this for each user and how to load in when logging in


class UserPersona(BaseModel):
    age: int
    gender: str
    height: int
    skin_tone: str
    style: list[str]

    # recommendations: {'item':ClothingTag, recommended: [{'top':clothingTag, 'bottom':clothingTag, 'shoes':clothingTag}]}
    recommendations: dict[str, list[dict[str, ClothingTag]] | ClothingTag]

    # preferences: {'tops': ["casual tee shirts","textual graphics tee shirts"],'bottoms': ["jeans","trousers"],'shoes': ["sneakers","boots"]}}
    # idea is to populate this when the user gives fedback on the recommendations, keep maybe 3 preferences for each category -> to be updated with each feedback
    preferences: dict[str, list[str]]


class ClothingTagEmbed(BaseModel):  # For catalogue
    clothing_type_embed: list[float]
    color_embed: list[float]
    material_embed: list[float]
    other_tags_embed: list[list[float]]


class WardrobeTag(BaseModel):  # For wardrobe
    name: str
    category: str
    tags: list[str]


openai_client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJ_ID,
    api_key=OPENAI_API_KEY
)


# Util function for dot products. Make sure the vectors are of the same length
def calculate_dot_product(vector_name_in_mongodb: str, vector_we_have: list[int]):
    return {
        "$sum": {
            "$map": {
                "input": {"$range": [0, len(vector_we_have)]},
                "as": "index",
                "in": {
                    "$multiply": [
                        {"$arrayElemAt": [vector_name_in_mongodb, "$$index"]},
                        {"$arrayElemAt": [vector_we_have, "$$index"]}
                    ]
                }
            }
        }
    }


def generate_wardrobe_tags(image_url: str) -> WardrobeTag:  # Generate tags from user uploaded image
    prompt = f"Give a name description of this clothing item (5 words or less) and include important styling features and colors in the name, choose category from {category_labels}, and tag with other descriptors as a list in this order [styles, ocassion, fit, color, material]. If there are multiple styles separate them by commas. Add the word 'fit' in the fit descriptor (e.g. regular fit). Give tags in all lowercase."
    output = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":  image_url
                        },
                    },
                ],
            }
        ],
        response_format=WardrobeTag,
    )

    # TODO(maybe?): Convert it to lowercase by code, in case chatgpt ignores the prompt and puts uppercase chars
    return json.loads(output.choices[0].message.content)


def str_to_clothing_tag(search: str) -> ClothingTag:
    print("DEPRECIATED, PLEASE USE ANOTHER FUNCTION")
    output = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You'll be given a description for an outfit. Generate descriptor tags for this as a list in this order [styles, ocassion, fit, color, material]. If there are multiple styles separate them by commas. Add the word 'fit' in the fit descriptor (e.g. regular fit). If nothing can be inferred for the type or color or material, use \"NIL\". Give tags in all lowercase."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": search
                    }
                ]
            },
        ],
        response_format=ClothingTag
    )
    # TODO(maybe): Convert it to lowercase by code, in case chatgpt ignores the prompt and puts uppercase chars.
    return ClothingTag(**json.loads(output.choices[0].message.content))

def style_to_clothing_tag(search: str) -> ClothingTag:
    output = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You'll be given a style. Generate accurate tags to describe the style"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": search
                    }
                ]
            },
        ],
        response_format=ClothingTag
    )
    # TODO(maybe): Convert it to lowercase by code, in case chatgpt ignores the prompt and puts uppercase chars.
    return ClothingTag(**json.loads(output.choices[0].message.content))

# TODO: safety to not prompt inject the additional prompt

def clothing_tag_to_embedding(tag: ClothingTag) -> ClothingTagEmbed:
    # TODO(maybe): Handle case where the tag is 'NIL'. Use [0,0,0....0]? Keep it as is?
    input = [tag.clothing_type, tag.color, tag.material] + tag.other_tags

    embedding_data = openai_client.embeddings.create(input=input, model="text-embedding-3-large").data

    clothing_type_embed = embedding_data[0].embedding
    color_embed = embedding_data[1].embedding
    material_embed = embedding_data[2].embedding
    other_tags_embed = []
    for i in range(3, len(embedding_data)):
        other_tags_embed.append(embedding_data[i].embedding)
    return ClothingTagEmbed(clothing_type_embed=clothing_type_embed, color_embed=color_embed, material_embed=material_embed, other_tags_embed=other_tags_embed)


def get_n_closest(
    tag_embed: ClothingTagEmbed, 
    n: int, 
    category_requirement: Optional[Literal['Tops', 'Bottoms', 'Shoes', 'Dresses']] = None,
    gender_requirements: List[Literal['M', 'F', 'U']] = None,
    exclude_category: bool = False  # If True, filter excludes specified item from being returned in results
):    # If category_requirement is empty, don't use filters. Otherwise, clothing must be of the specified type.
    # Random bucketing disabled for now.

    # LIMITATIONS: other_tag matching is not by percentage of match, but by raw number of matches. This means catalogue objects with more "other_tags" will have a higher likelihood to score more.
    # Also, after the first and second filtering stage, best match is purely by tag_match_count. (If any are equal, score from second stage is used as tie breaker)

    # From n * candidate * first_stage candidates:
    # find the n * first_stage closest matching clothing_types
    # then find n * second_stage closest score (0.7*clothing_type + 0.3*color) (or whatever the multiplier is)
    # then among those, find the n items with the largest number of similar tags. (See synonym_count_per_tag)
    CANDIDATE_TO_LIMIT_RATIO = 10
    FIRST_STAGE_FILTER_RATIO = 100
    SECOND_STAGE_FILTER_RATIO = 20
    CLOTHING_TYPE_WEIGHT = 0.7
    COLOR_WEIGHT = 0.3

    # How many synonyms to "exact match" for each other_tag. Eg: If it's 3, then ['simple', 'sleeveless'] might match ['simple', 'easy', 'simple design', 'sleeveless', 'no sleeves', 'short sleeves']
    SYNONYM_COUNT_PER_TAG = 25

    final_tags = []
    for embed in tag_embed.other_tags_embed:
        cursor = tag_embeddings.aggregate([
            {
                '$vectorSearch': {
                    'index': 'embedding',
                    'path': 'embedding',
                    'queryVector': embed,
                    'numCandidates': SYNONYM_COUNT_PER_TAG * CANDIDATE_TO_LIMIT_RATIO,
                    'limit': SYNONYM_COUNT_PER_TAG,
                }
            },
            {
                '$project': {
                    '_id': 0,
                    'tag': 1
                }
            }
        ])
        final_tags += [item['tag'] for item in list(cursor)]

    # bucket_count = get_catalogue_metadata().bucket_count
    # bucket_num = random.randint(1, bucket_count)

    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_search_with_category_filter',
                'path': 'clothing_type_embed',
                'queryVector': tag_embed.clothing_type_embed,
                'numCandidates': n * FIRST_STAGE_FILTER_RATIO * CANDIDATE_TO_LIMIT_RATIO,
                'limit': n * FIRST_STAGE_FILTER_RATIO,
                # 'filter': {'bucket_num': bucket_num}
            }
        },
        {
            "$addFields": {
                "color_score": calculate_dot_product("$color_embed", tag_embed.color_embed)
            }
        },
        {
            "$addFields": {
                "combined_score": {
                    "$add": [
                        {"$multiply": [CLOTHING_TYPE_WEIGHT, {"$meta": "vectorSearchScore"}]},
                        {"$multiply": [COLOR_WEIGHT, "$color_score"]}
                    ]
                }
            }
        },
        {
            "$sort": {"combined_score": -1}
        },
        {
            "$limit": n * SECOND_STAGE_FILTER_RATIO
        },
        {
            '$addFields': {
                'other_tag_match_count': {
                    '$size': {
                        '$setIntersection': ["$other_tags", final_tags]
                    }
                }
            }
        },
        {
            '$sort': {'other_tag_match_count': -1, "combined_score": -1}
        },
        {
            '$limit': n
        },
        {
            '$project': {
                '_id': 1,
                'name': 1,
                'category': 1,
                'color': 1,
                'clothing_type': 1,
                'material': 1,
                'other_tags': 1,
                'price': 1,
                'image_url': 1,
                'product_url': 1,
                'retailer': 1,
                'gender': 1,
                'other_tag_match_count': 1,
                'cropped_image_url': 1,
            }
        }
    ]
    temp_dict = {}
    # Modified category filtering logic
    if category_requirement is not None:
        if exclude_category:
            temp_dict['category'] = {'$ne': category_requirement}
        else:
            temp_dict['category'] = category_requirement
            
    if gender_requirements is not None:
        temp_dict['gender'] = {'$in': gender_requirements}
        
    if len(temp_dict) > 0:
        pipeline[0]['$vectorSearch']['filter'] = temp_dict

    return catalogue.aggregate(pipeline)

# Define the StyleSuggestion and StyleAnalysisResponse models

class StyleSuggestion(BaseModel):
    style: str
    description: str
    reasoning: str


class StyleAnalysisResponse(BaseModel):
    top_styles: List[StyleSuggestion]


def get_wardrobe_recommendation(tag: WardrobeTag, profile: dict, additional_prompt: str = "") -> list[ClothingTag]:
    # added user persona 23/02

    tag_dict = tag.model_dump()
    if additional_prompt != "":
        tag_dict["additional_prompt"] = additional_prompt

    prompt = json.dumps(tag_dict)
    openai_output = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": f"""You are a personal stylist, your task is to generate a complete outfit based off a starting item. You will be given a starting item as JSON input containing: name (description of item), category (top, bottom, shoes, layer) and tags (style tags).\n\nA complete outfit:\n- Either (dress + shoes) \n- Or (one top + one bottom + shoes)\n- Optionally include one layering piece (e.g., jacket) if it matches the style and context.\n\nCreating an outfit:\n1.) analyse the style of the given item and additional context to select an outfit style. \n2.) based off the category of the item and the definition of an outfit, identify the other categories required to complete an outfit. You can only have 1 item from each category. There should be a maximum 3 unique categories in the output.\n3.)recommend clothing items in these categories that match the overall outfit style. Output the different items of the outfit in JSON with the tags: clothing_type (descriptor of the item) , color, material, and other_tags (category, comma separated styles, ocassion, fit, color, material), category ("Tops", "Bottoms", "Shoes", "Dresses") \n\nCarefully consider the style of the input, the users preferences defined below and the additional context (if any) when choosing the overall style of the outfit. Take inspiration from user preferences but include some variation. Ensure only one item per category in the outfit. Do not include the starting item or the same category in the output. Ensure a complete outfit can be created with the recommended items.\n\n
                    User Persona: {profile['age']}-year-old {profile['gender']} in {profile['location']}, {profile['skin_tone']} skin, {profile['style']} style.
                    Likes: {profile['clothing_likes']}.
                    Likes: {profile['clothing_likes']}.
                    Dislikes: {profile['clothing_dislikes']}.
                    """}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        },
          tools=[
            {
            "type": "function",
            "function": {
                "name": "recommend_clothing",
                "description": "Generates clothing recommendations based on user preferences",
                "parameters": {
                "type": "object",
                "required": [
                    "recommendations"
                ],
                "properties": {
                    "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                        "clothing_type",
                        "color",
                        "material",
                        "other_tags",
                        "category"
                        ],
                        "properties": {
                        "color": {
                            "type": "string",
                            "description": "Color of the clothing item"
                        },
                        "material": {
                            "type": "string",
                            "description": "Material of the clothing item"
                        },
                        "other_tags": {
                            "type": "array",
                            "items": {
                            "type": "string",
                            "description": "Additional tags related to clothing characteristics"
                            },
                            "description": "Additional tags describing the clothing"
                        },
                        "clothing_type": {
                            "type": "string",
                            "description": "Type of clothing item"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category of the clothing item. Can be [Tops, Bottoms, Shoes, Dresses]"
                        }
                        },
                        "additionalProperties": False
                    },
                    "description": "Array of clothing recommendations"
                    }
                },
                "additionalProperties": False
                },
                "strict": True
            }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "recommend_clothing"}},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    clothing_recommendations: List[Tuple[ClothingTag, str]] = []

    for tool_call_str in openai_output.choices[0].message.tool_calls:
        tool_call_json = json.loads(tool_call_str.function.arguments)
        for clothing_item in tool_call_json["recommendations"]:
            category = clothing_item["category"]  # Extract category
            clothing_data = {key: value for key, value in clothing_item.items() if key != "category"}  # Exclude category
            
            clothing_tag = ClothingTag(**clothing_data)  # Create ClothingTag object
            clothing_recommendations.append((clothing_tag, category))  # Store as tuple
    print(clothing_recommendations)
    return clothing_recommendations

def get_user_feedback_recommendation(starting_item: WardrobeTag, disliked_item: WardrobeTag, dislike_reason: str, profile: dict):
    # generates a new recommendation given a disliked previous one. Should be called when user dislikes a recommended item from the wardrobe page.
    # TODO need to figure out how to pass the outfit style of the starting item
    # TODO include additional prompts in the recommendation

    outfit_style = starting_item.tags[:3]
    # [styles, ocassion, fit, color, material] --> hardcoded order in generate_wardrobe_tags, we only use the first 3 to determine the outfit style

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "text": f"You are a fashion stylist recommending clothes to a user. The user wants to generate an alternative for a single item from an outfit because of some dislike reason. You will be given the disliked item as a JSON with the following tags: \"name\" (description of the item), \"category\" (top, bottom, shoes), \"tags\" (style descriptors). You will also be given style descriptors of the outfit.\n\n1.) Store the category of the disliked. You should only recommend an item from the same category.\n2.) Use the style descriptors of the outfit to gauge the style of the item you will be generating.\n3.) The dislike reason will be given as one of the tags of the item, change that when generating the new outfit. Only the dislike reason should be drastically changed, try to keep other factors the same. You should act as a sales rep suggesting an alternative, not a completely new item.\n\nOutput the new recommendation in JSON with the tags: clothing_type (descriptor of the item) , color, material, and other_tags (category, comma separated styles, ocassion, fit, color, material)\n\nUser Persona: {profile['age']}-year-old {profile['gender']} in {profile['location']}, with {profile['style']} style. Likes: {profile['clothing_likes']}. Dislikes: {profile['clothing_dislikes']}.",
                        "type": "text"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"disliked_item ({disliked_item})\n\noutfit_style ({outfit_style})\n\ndislike_reason({dislike_reason})"
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        },
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "feedback_single_item",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": [
                            "clothing_type",
                            "color",
                            "material",
                            "other_tags"
                        ],
                        "properties": {
                            "color": {
                                "type": "string",
                                "description": "color of clothing item"
                            },
                            "material": {
                                "type": "string",
                                "description": "Material of the clothing item (e.g., cotton, wool, polyester)"
                            },
                            "other_tags": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "A style tag (e.g., casual, formal, vintage)"
                                },
                                "description": "Any other tags that describe the item's style"
                            },
                            "clothing_type": {
                                "type": "string",
                                "description": "Type of clothing item (e.g., shirt, pants, dress)"
                            }
                        },
                        "additionalProperties": False
                    },
                    "description": "Describes a clothing item with its features."
                }
            }
        ],
        tool_choice={
            "type": "function",
            "function": {
                "name": "feedback_single_item"
            }
        },
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    output = response.choices[0].message.tool_calls[0].function.arguments
    output_json = json.loads(output)
    clothing_item = ClothingTag(**output_json)
    return clothing_item

def generate_base_catalogue_recommendation(tag: WardrobeTag, additional_prompt: str = "") -> list[ClothingTag]:
    tag_dict = tag.model_dump()
    if additional_prompt != "":
        tag_dict["additional_prompt"] = additional_prompt

    prompt = json.dumps(tag_dict)
    openai_output = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": """You are a personal stylist, your task is to generate a complete outfit based off a starting item. You will be given a starting item as JSON input containing: name (description of item), category (top, bottom, shoes, layer) and tags (style tags).\n\nA complete outfit:\n- Either (dress + shoes) \n- Or (one top + one bottom + shoes)\n- Optionally include one layering piece (e.g., jacket) if it matches the style and context.\n\nCreating an outfit:\n1.) analyse the style of the given item and additional context to select an outfit style. \n2.) based off the category of the item and the definition of an outfit, identify the other categories required to complete an outfit. You can only have 1 item from each category. There should be a maximum 3 unique categories in the output.\n3.)recommend clothing items in these categories that match the overall outfit style. Output the different items of the outfit in JSON with the tags: clothing_type (descriptor of the item) , color, material, and other_tags (category, comma separated styles, ocassion, fit, color, material)\n\nCarefully consider the style of the input and the additional context (if any) when choosing the overall style of the outfit. Ensure only one item per category in the outfit. Do not include the starting item or the same category in the output. Ensure a complete outfit can be created with the recommended items."""
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        response_format={
            "type": "json_object"
        },
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "recommend_clothing",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "required": [
                            "recommendations"
                        ],
                        "properties": {
                            "recommendations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": [
                                        "clothing_type",
                                        "color",
                                        "material",
                                        "other_tags"
                                    ],
                                    "properties": {
                                        "color": {
                                            "type": "string",
                                            "description": "Color of the clothing item"
                                        },
                                        "material": {
                                            "type": "string",
                                            "description": "Material of the clothing item"
                                        },
                                        "other_tags": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "Additional tags related to clothing characteristics"
                                            },
                                            "description": "Additional tags describing the clothing"
                                        },
                                        "clothing_type": {
                                            "type": "string",
                                            "description": "Type of clothing item"
                                        }
                                    },
                                    "additionalProperties": False
                                },
                                "description": "Array of clothing recommendations"
                            }
                        },
                        "additionalProperties": False
                    },
                    "description": "Generates clothing recommendations based on user preferences"
                }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "recommend_clothing"}},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    clothing_tags: list[ClothingTag] = []

    for tool_call_str in openai_output.choices[0].message.tool_calls:
        tool_call_json = json.loads(tool_call_str.function.arguments)
        for clothing_tag in tool_call_json["recommendations"]:
            clothing_tags.append(ClothingTag(**clothing_tag))

    return clothing_tags

def compile_tags_and_embeddings_for_item(item_id: str) -> int:
    """
    For a single catalogue item (specified by `item_id`), 
    gather the unique (tag -> embedding) pairs from:
      - item['other_tags'] / item['other_tags_embed']
      - each base_recommendation in item['base_recommendations'], 
        specifically rec['other_tags'] / rec['other_tags_embed'].
    Insert them into 'tag_embeddings' collection (one doc per unique tag).

    Returns the number of **new** tags inserted (i.e. if the tag already exists in tag_embeddings, skip).
    """
    catalogue_col = db.catalogue
    tags_col = db.tag_embeddings

    try:
        object_id = ObjectId(item_id)
    except:
        # Not a valid ObjectId; you could raise an Exception or return 0
        return 0

    # 1) Fetch the catalogue item
    item = catalogue_col.find_one({"_id": object_id})
    if not item:
        return 0  # not found, nothing to do

    # 2) Prepare a dictionary for tag->embedding from this single item
    item_tag_dict: Dict[str, list[float]] = {}

    # A) Process the item’s own other_tags
    other_tags = item.get("other_tags", [])
    other_tags_embed = item.get("other_tags_embed", [])
    if len(other_tags) == len(other_tags_embed):
        for tag, embed_vec in zip(other_tags, other_tags_embed):
            item_tag_dict[tag] = embed_vec  # override if repeated in the same item

    # B) Process each base_recommendation
    base_recs = item.get("base_recommendations", [])
    for rec in base_recs:
        rec_tags = rec.get("other_tags", [])
        rec_tags_embed = rec.get("other_tags_embed", [])
        if len(rec_tags) == len(rec_tags_embed):
            for tag, embed_vec in zip(rec_tags, rec_tags_embed):
                item_tag_dict[tag] = embed_vec

    # 3) For each unique tag from this item, check if it already exists in tag_embeddings
    #    and insert if missing
    new_insert_count = 0
    for tag, embed_vec in item_tag_dict.items():
        # If you want to ensure uniqueness by 'tag', you can attempt an upsert or findOne check:
        existing = tags_col.find_one({"tag": tag})
        if existing:
            # Already in the collection; skip if you don't want to update
            # or optionally upsert with new embedding:
            # tags_col.update_one({"_id": existing["_id"]}, {"$set": {"embedding": embed_vec}})
            continue
        else:
            tags_col.insert_one({"tag": tag, "embedding": embed_vec})
            new_insert_count += 1

    return new_insert_count


def get_all_catalogue_ids() -> list[str]:
    """
    Returns a list of all catalogue item string IDs from `db.catalogue`.
    """
    catalogue_col = db.catalogue
    ids = catalogue_col.find({}, {"_id": 1})
    return [str(doc["_id"]) for doc in ids]


def get_n_closest_no_other_tags(tag_embed: ClothingTagEmbed, n: int):
    FIRST_STAGE_FILTER_RATIO = 10
    CANDIDATE_TO_LIMIT_RATIO = 10
    CLOTHING_TYPE_WEIGHT = 0.7
    COLOR_WEIGHT = 0.3

    bucket_count = get_catalogue_metadata().bucket_count
    bucket_num = random.randint(1, bucket_count)

    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_index',
                'path': 'clothing_type_embed',
                'queryVector': tag_embed.clothing_type_embed,
                'numCandidates': n * FIRST_STAGE_FILTER_RATIO * CANDIDATE_TO_LIMIT_RATIO,
                'limit': n * FIRST_STAGE_FILTER_RATIO,
                'filter': {'bucket_num': bucket_num}
            }
        },
        {
            "$addFields": {
                "color_score": calculate_dot_product("$color_embed", tag_embed.color_embed)
            }
        },
        {
            "$addFields": {
                "combined_score": {
                    "$add": [
                        {"$multiply": [CLOTHING_TYPE_WEIGHT, {"$meta": "vectorSearchScore"}]},
                        {"$multiply": [COLOR_WEIGHT, "$color_score"]}
                    ]
                }
            }
        },
        {
            "$sort": {"combined_score": -1}
        },
        {
            "$limit": n
        },
        {
            '$project': {
                '_id': 1,
                'name': 1,
                'category': 1,
                'color': 1,
                'clothing_type': 1,
                'material': 1,
                'other_tags': 1,
                'price': 1,
                'image_url': 1,
                'product_url': 1,
                'retailer': 1,
                'gender': 1
            }
        }
    ]
    return catalogue.aggregate(pipeline)

#generating recommendations within the user's wardrobe
def generate_embeddings(text_description):
    response = openai_client.embeddings.create(
    input=text_description,
    model="text-embedding-3-large"
    )
    embedding_vector = response.data[0].embedding
    return embedding_vector


def complementary_categories(category:str) -> List[str]:
    #category_labels = ["Tops", "Bottoms", "Dresses", "Shoes", "Jackets", "Accessories"]
    category_map = {
        "Tops":["Bottoms","Shoes","Jackets","Accessories"],
        "Bottoms":["Tops","Shoes","Jackets","Accessories"],
        "Dresses":["Shoes","Jackets","Accessories"],
        "Jackets":["Tops","Shoes","Bottoms","Accessories"],
        "Accessories":["Tops","Shoes","Bottoms","Jackets"],
        "Shoes":["Tops","Bottoms","Jackets","Accessories"],

    }
    return category_map[category]

def complementary_wardrobe_item_vectorsearch_pipline(user, category, embedding: List[float], n: int = 3):
    """
    Returns MongoDB Atlas Search aggregation pipeline for vector similarity search.
    """
    return [
        {
            "$vectorSearch": {
                # Index name must match what you created
                "index": "wardrobe_vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": n * 10,   # Oversample for better re-ranking
                "limit": n * 2,
                "filter": {
                    "$and": [
                        {"user_id": {"$eq": user}},
                        {"category": {"$eq": category}}
                    ]
                }
            }
        },
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "category": 1,
                "tags": 1,
                "image_name": 1,
                # Show the vector similarity score
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        # If you want exactly n final results:
        {
            "$limit": n
        }
    ]

def generate_wardrobe_outfit(user_style:str, closest_items:list, starting_name:str, starting_cat:str,addn_prompt:str):
    candidate_tops = []
    candidate_bottoms = []
    candidate_dresses = []
    candidate_dresses = []
    candidate_jackets = []
    candidate_shoes = []
    candidate_accessories = []

    candidate_items_list = []
    
    for candidates in closest_items:
        for candidate in candidates:
            candidate_items_list.append(candidate)
            if candidate["category"] == "Tops":
                candidate_tops.append(candidate['name'])
            elif candidate["category"] == "Bottoms":
                candidate_bottoms.append(candidate['name'])
            elif candidate["category"] == "Dresses":
                candidate_dresses.append(candidate['name'])
            elif candidate["category"] == "Jackets":
                candidate_jackets.append(candidate['name'])
            elif candidate["category"] == "Shoes":
                candidate_shoes.append(candidate['name'])
            elif candidate["category"] == "Accessories":
                candidate_accessories.append(candidate['name'])
    
    response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": f"Role: You are a fashion stylist tasked with creating a stylish and cohesive outfit from a set of clothes. This outfit must align with the user's style preference. The user's style preference is: {user_style}\n\nObjective: You will receive a starting item to build the outfit around along with a list of similarly styled items to choose from. These items will be divided by their category [\"Tops\", \"Bottoms\", \"Dresses\", \"Shoes\", \"Jackets\", \"Accessories\"]. You will receive a list of up to 3 items from each category.\n\nThe user input will be provided as follows:\n- starting item: category: \"category_name\", item: \"item_name\"\n- candidate clothes for each category\n-Optionally an additional ocassion or styling prompt\n\nYour goal is to select the name of exactly one item from each category (if that category exists in the input) to create a single cohesive outfit that aligns with the user’s style preference and the additonal prompt(if any). Ignore the additional prompt if it is not related. Give the output in JSON."
            }
        ]
        },
        {
         "role":"user",
         "content" :[
            {
            "type": "text",
            "text": f"starting_item: {starting_name}, starting_category: {starting_cat}\nCandidate clothes:\nTops: {candidate_tops},Bottoms: {candidate_bottoms}, Dresses: {candidate_dresses}, Jackets: {candidate_jackets}, Shoes: {candidate_shoes}, Accessories: {candidate_accessories}\nAdditional Prompt: ```{addn_prompt}```"
            }
         ]  
        }
    ],
    response_format={
        "type": "text"
    },
    tools=[
        {
        "type": "function",
        "function": {
            "name": "generate_clothing_output",
            "description": "Outputs selected clothing items for each of the categories, leave the description as empty string if there is no item to select.",
            "parameters": {
            "type": "object",
            "required": [
                "Tops",
                "Bottoms",
                "Dresses",
                "Shoes",
                "Jackets",
                "Accessories"
            ],
            "properties": {
                "Tops": {
                "type": "string",
                "description": "Name of the top clothing item"
                },
                "Bottoms": {
                "type": "string",
                "description": "Name of the bottom clothing item"
                },
                "Dresses": {
                "type": "string",
                "description": "Name of the dress"
                },
                "Shoes": {
                "type": "string",
                "description": "Name of the shoes"
                },
                "Jackets": {
                "type": "string",
                "description": "Name of the jacket"
                },
                "Accessories": {
                "type": "string",
                "description": "Name of the accessories"
                }
            },
            "additionalProperties": False
            },
            "strict": True
        }
        }
    ],
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    store=False
    )

    output = response.choices[0].message.tool_calls[0].function.arguments
    output_json = json.loads(output)

    top, bottoms, dresses, shoes, jackets, accessories = output_json["Tops"],output_json["Bottoms"],output_json["Dresses"],output_json["Shoes"],output_json["Jackets"],output_json["Accessories"]

    outfit_ids = []
    for candidate in candidate_items_list:
        if candidate['name'] == top:
            outfit_ids.append(candidate['_id'])
        elif candidate['name'] == bottoms:
            outfit_ids.append(candidate['_id'])
        elif candidate['name'] == dresses:
            outfit_ids.append(candidate['_id'])
        elif candidate['name'] == shoes:
            outfit_ids.append(candidate['_id'])
        elif candidate['name'] == jackets:
            outfit_ids.append(candidate['_id'])
        elif candidate['name'] == accessories:
            outfit_ids.append(candidate['_id'])

    return outfit_ids
