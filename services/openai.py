from typing import List, Optional, Literal, Tuple
from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
from services.mongodb import catalogue, tag_embeddings
from services.metadata import get_catalogue_metadata
from pydantic import BaseModel
from bson import ObjectId
import random
import json

# Hardcoded available categories
category_labels = ["Tops", "Bottoms", "Dresses", "Shoes", "Jackets", "Accessories"]

# Fixed formats which GPT4 will force to return

create_outfit_tool = {
    "type": "function",
    "function": {
        "name": "create_outfit",
        "description": "Assembles an outfit consisting of top, bottom, and shoes with specified attributes.",
        "parameters": {
            "type": "object",
            "required": [
                "top",
                "shoes",
                "bottom"
            ],
            "properties": {
                "top": {
                    "type": "object",
                    "required": [
                        "color",
                        "material",
                        "other_tags",
                        "clothing_type"
                    ],
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "Color of the top garment"
                        },
                        "material": {
                            "type": "string",
                            "description": "Material from which the top is made"
                        },
                        "other_tags": {
                            "type": "array",
                            "description": "List of tags describing the style of the top",
                            "items": {
                                "type": "string"
                            }
                        },
                        "clothing_type": {
                            "type": "string",
                            "description": "Type of clothing for the top"
                        }
                    },
                    "additionalProperties": False
                },
                "shoes": {
                    "type": "object",
                    "required": [
                        "color",
                        "material",
                        "other_tags",
                        "clothing_type"
                    ],
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "Color of the shoes"
                        },
                        "material": {
                            "type": "string",
                            "description": "Material from which the shoes are made"
                        },
                        "other_tags": {
                            "type": "array",
                            "description": "List of tags describing the style of the shoes",
                            "items": {
                                "type": "string"
                            }
                        },
                        "clothing_type": {
                            "type": "string",
                            "description": "Type of shoes"
                        }
                    },
                    "additionalProperties": False
                },
                "bottom": {
                    "type": "object",
                    "required": [
                        "color",
                        "material",
                        "other_tags",
                        "clothing_type"
                    ],
                    "properties": {
                        "color": {
                            "type": "string",
                            "description": "Color of the bottom garment"
                        },
                        "material": {
                            "type": "string",
                            "description": "Material from which the bottom is made"
                        },
                        "other_tags": {
                            "type": "array",
                            "description": "List of tags describing the style of the bottom",
                            "items": {
                                "type": "string"
                            }
                        },
                        "clothing_type": {
                            "type": "string",
                            "description": "Type of clothing for the bottom"
                        }
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        },
        "strict": True
    }
}


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
    prompt = f"Give a name description of this clothing item (5 words or less), choose category from {category_labels}, and tag with other descriptors as a list in this order [styles, ocassion, fit, color, material]. If there are multiple styles separate them by commas. Add the word 'fit' in the fit descriptor (e.g. regular fit). Give tags in all lowercase."
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

# TODO: safety to not prompt inject the additional prompt


def generate_outfit_recommendations(item: ClothingTag, additional_prompt: str, user) -> ClothingTag:
    age = user.age
    gender = user.gender
    skin_tone = user.skin_tone
    style = user.style
    preferences = user.preferences

    # making a nice sentence about user preferences to inclcude in prompt
    parts = []
    if preferences == {}:
        user_preferences = ""
    else:
        if preferences['tops']:
            parts.append(f"{', '.join(preferences['tops'])}")
        if preferences['bottoms']:
            parts.append(f"{', '.join(preferences['bottoms'])}")
        if preferences['shoes']:
            parts.append(f"{', '.join(preferences['shoes'])}")
        user_preferences = f"The user prefers wearing: {', '.join(parts)}."

    # if there is no additional prompt, we don't want to include it in the prompt
    if not additional_prompt.strip():
        item_description = f"{item.color} {item.material} {item.clothing_type} ({item.other_tags}). Additional prompt: {additional_prompt}"
    else:
        item_description = f"{item.color} {item.material} {item.clothing_type} ({item.other_tags})"

    system_message = f"You are a fashion stylist creating an outfit for a {age} year old {skin_tone} skin {gender} who likes wearing {style} outfits. \
        {user_preferences} \
        Given a description of a clothing item (example: white polo tee shirt with black printed text) recommend complementary clothes to complete the outfit. \
        Follow these steps to recommend an outfit: 1) consider the user persona mentioned to analyse the style of clothes to be recommend.\
        2) if the given item is a top (tee shirt, polo, shirt, dress, tank top etc), give recommendations for bottoms (pants, shorts, trousers, jeans skirts, leggings etc) and for shoes. \
        If the given item is a shoe, give recommendations for tops and bottoms. If the given item is a bottom, give recommendations for the top and shoes. \
        You may also be given additional prompts in text to constrain the style of the matched item. \
        Compile the output into a JSON which contains the description of all the items in the completed outfit. Generate 3 such outfits. "

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": item_description},
        ],
        tools=[create_outfit_tool],
        tool_choice={"type": "function", "function": {"name": "create_outfit"}})

    parsed_response = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    top = parsed_response['top']
    bottom = parsed_response['bottom']
    shoes = parsed_response['shoes']

    top = ClothingTag(**{**top, 'other_tags': list(top['other_tags'])})
    bottom = ClothingTag(**{**bottom, 'other_tags': list(bottom['other_tags'])})
    shoes = ClothingTag(**{**shoes, 'other_tags': list(shoes['other_tags'])})

    # recommended = {'item':item, recommended: [{'top':top, 'bottom':bottom, 'shoes':shoes}]}
    if user.recommendations == {}:
        recommended = []
    else:
        recommended = user.recommendations['recommended']

    # #we only want to keep information about the last 3 recommendations
    if len(recommended) < 3:
        recommended.append({'top': top, 'bottom': bottom, 'shoes': shoes})
    else:
        recommended.pop(0)
        recommended.append({'top': top, 'bottom': bottom, 'shoes': shoes})

    user.recommendations = {'item': item, 'recommended': recommended}
    return top, bottom, shoes


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


def get_n_closest(tag_embed: ClothingTagEmbed, n: int, category_requirement: Optional[Literal['Tops', 'Bottoms', 'Shoes', 'Dresses']] = None, gender_requirements: List[Literal['M', 'F', 'U']] = None):
    # If category_requirement is empty, don't use filters. Otherwise, clothing must be of the specified type.
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
                'other_tag_match_count': 1
            }
        }
    ]
    temp_dict = {}
    if category_requirement is not None:
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


def user_style_to_embedding(user_style: StyleAnalysisResponse) -> list[float]:
    # Combine the styles and descriptions into a single text
    style_texts = []
    for style_suggestion in user_style.top_styles:
        style_texts.append(f"{style_suggestion.style}: {style_suggestion.description}")
    combined_style_text = " ".join(style_texts)

    # Get the embedding
    embedding_response = openai_client.embeddings.create(
        input=combined_style_text, model="text-embedding-3-large"
    )
    embedding = embedding_response.data[0].embedding
    return embedding


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


def get_n_closest_with_filter(tag_embed: ClothingTagEmbed, category: str, n: int):
    print("############ IMPORTANT #########\n\n\n'get_n_closest_with_filter' is deprecated. Use get_n_closest instead, it has an optional argument.")
    """
    Returns the n closest items in the specified category using vector search.
    """
    # You can reuse the FIRST_STAGE_FILTER_RATIO, weighting, etc. from get_n_closest.
    # The main difference is the pipeline's filter includes "category": category.
    FIRST_STAGE_FILTER_RATIO = 10
    CANDIDATE_TO_LIMIT_RATIO = 10
    CLOTHING_TYPE_WEIGHT = 0.7
    COLOR_WEIGHT = 0.3

    from services.metadata import get_catalogue_metadata
    import random
    import services.mongodb as mongodb

    bucket_count = get_catalogue_metadata().bucket_count
    bucket_num = random.randint(1, bucket_count)

    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_search_with_category_filter',
                'path': 'clothing_type_embed',
                'queryVector': tag_embed.clothing_type_embed,
                'numCandidates': n * FIRST_STAGE_FILTER_RATIO * CANDIDATE_TO_LIMIT_RATIO,
                'limit': n * FIRST_STAGE_FILTER_RATIO,
                'filter': {
                    'bucket_num': bucket_num,
                    'category': category
                }
            }
        },
        # Add color_score, combined_score, etc., same as get_n_closest
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
            "$project": {
                '_id': 1,
                'name': 1,
                'category': 1,
                'price': 1,
                'image_url': 1,
                'product_url': 1,
                'clothing_type': 1,
                'color': 1,
                'material': 1,
                'other_tags': 1
            }
        }
    ]
    return mongodb.catalogue.aggregate(pipeline)


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
        "Bottoms":["Tops","Bottoms","Jackets","Accessories"],

    }
    return category_map[category]

def complementary_wardrobe_item_vectorsearch_pipline(user_id: str, category, embedding: List[float], limit: int = 5):
    """
    Returns MongoDB Atlas Search aggregation pipeline for vector similarity search.
    """
    return [
        {
            "$search": {
                "index": "wardrobe_vector_index",
                "knnBeta": {
                    "vector": embedding,
                    "path": "embedding",
                    "k": limit,
                    "filter": {
                        "compound": {
                            "must": [
                                { "equals": { "path": "user_id", "value": ObjectId(user_id) }},
                                { "text": { "path": "category", "query": category }}
                            ]
                        }
                    }
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
                "score": { "$meta": "searchScore" }
            }
        }
    ]