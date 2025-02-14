from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
from services.mongodb import catalogue
from services.metadata import get_catalogue_metadata
from pydantic import BaseModel
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
            "required": ["top", "bottom", "shoes"],
            "properties": {
                "top": {
                    "type": "object",
                    "required": ["clothing_type", "colour", "material", "other_tags"],
                    "properties": {
                        "clothing_type": {"type": "string", "description": "Type of clothing for the top"},
                        "colour": {"type": "string", "description": "Color of the top garment"},
                        "material": {"type": "string", "description": "Material from which the top is made"},
                        "other_tags": {"type": "string", "description": "Description of the style of the top"}
                    },
                    "additionalProperties": False
                },
                "bottom": {
                    "type": "object",
                    "required": ["clothing_type", "colour", "material", "other_tags"],
                    "properties": {
                        "clothing_type": {"type": "string", "description": "Type of clothing for the bottom"},
                        "colour": {"type": "string", "description": "Color of the bottom garment"},
                        "material": {"type": "string", "description": "Material from which the bottom is made"},
                        "other_tags": {"type": "string", "description": "Description of the style of the bottom"}
                    },
                    "additionalProperties": False
                },
                "shoes": {
                    "type": "object",
                    "required": ["clothing_type", "colour", "material", "other_tags"],
                    "properties": {
                        "clothing_type": {"type": "string", "description": "Type of shoes"},
                        "colour": {"type": "string", "description": "Color of the shoes"},
                        "material": {"type": "string", "description": "Material from which the shoes are made"},
                        "other_tags": {"type": "string", "description": "Description of the style of the shoes"}
                    },
                    "additionalProperties": False
                }
            },
            "additionalProperties": False
        }
    }
}

class ClothingTag(BaseModel):  # For catalogue
    clothing_type: str
    color: str
    material: str
    other_tags: list[str]

#store the recommendations given to the user, to allow for feedback to be given - then store the feedback to make next recommendations better
#I have no idea how to store this for each user and how to load in when logging in
class UserPersona(BaseModel):
    age: int
    gender: str
    height: int
    skin_tone: str
    style: list[str]

    #recommendations: {'item':ClothingTag, recommended: [{'top':clothingTag, 'bottom':clothingTag, 'shoes':clothingTag}]}
    recommendations: dict[str, list[dict[str, ClothingTag]] | ClothingTag]

    #preferences: {'tops': ["casual tee shirts","textual graphics tee shirts"],'bottoms': ["jeans","trousers"],'shoes': ["sneakers","boots"]}}
    #idea is to populate this when the user gives fedback on the recommendations, keep maybe 3 preferences for each category -> to be updated with each feedback
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


def generate_wardrobe_tags(image_url: str) -> WardrobeTag:  # Generate tags from user uploaded image
    prompt = f"Give a name description of this clothing item (5 words or less), choose category from {category_labels}, and tag with other adjectives (eg. color, material, occasion, fit, sleeve, brand). Give tags in all lowercase."
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
                        "text": "You'll be given a description for an outfit. Generate tags for this, including clothing type, color, material, other adjectives (eg. occasion, fit, sleeve, brand). If nothing can be inferred for the type or color or material, use \"NIL\". For others tag, keep it as an empty list if nothing can be inferred. Give tags in all lowercase."
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

#TODO: safety to not prompt inject the additional prompt
def generate_outfit_recommendations(item: ClothingTag, additional_prompt: str) -> ClothingTag:
    age = UserPersona.age
    gender = UserPersona.gender
    skin_tone = UserPersona.skin_tone
    style = UserPersona.style
    preferences = UserPersona.preferences

    #making a nice sentence about user preferences to inclcude in prompt
    parts = []
    if preferences != {}:
        users_preferences = ""
    else:
        if preferences['tops']:
            parts.append(f"{', '.join(preferences['tops'])}")
        if preferences['bottoms']:
            parts.append(f"{', '.join(preferences['bottoms'])}")
        if preferences['shoes']:
            parts.append(f"{', '.join(preferences['shoes'])}")
        user_preferences = f"The user prefers wearing: {', '.join(parts)}."

    #if there is no additional prompt, we don't want to include it in the prompt
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
        tools = [create_outfit_tool],
        tool_choice = {"type": "function", "function": {"name": "create_outfit"}})
    
    parsed_response = json.loads(response.choices[0].message.content)
    top = ClothingTag(**parsed_response['top'])
    bottom = ClothingTag(**parsed_response['bottom'])
    shoes = ClothingTag(**parsed_response['shoes'])


    #recommendations: {'item':ClothingTag, recommended: [{'top':clothingTag, 'bottom':clothingTag, 'shoes':clothingTag}]}
    recommended = UserPersona.recommendations['recommended']

    #we only want to keep information about the last 3 recommendations
    if len(recommended) < 3:
        recommended.append({'top': top, 'bottom': bottom, 'shoes': shoes})
    else:
        recommended.pop(0)
        recommended.append({'top': top, 'bottom': bottom, 'shoes': shoes})

    UserPersona.recommendations = {'item': item, 'recommended': recommended}
    return top, bottom, shoes

def clothing_tag_to_embedding(tag: ClothingTag) -> ClothingTagEmbed:
    # TODO(maybe): Handle case where the tag is 'NIL'. Use [0,0,0....0]? Keep it as is?
    
    clothing_type_embed = openai_client.embeddings.create(
        input=tag.clothing_type, model="text-embedding-3-large").data[0].embedding
    color_embed = openai_client.embeddings.create(
        input=tag.color, model="text-embedding-3-large").data[0].embedding
    material_embed = openai_client.embeddings.create(
        input=tag.material, model="text-embedding-3-large").data[0].embedding
    other_tags_embed = []
    for o in tag.other_tags:
        other_tags_embed.append(openai_client.embeddings.create(input=o, model="text-embedding-3-large").data[0].embedding)
    return ClothingTagEmbed(clothing_type_embed=clothing_type_embed, color_embed=color_embed, material_embed=material_embed, other_tags_embed=other_tags_embed)


def get_n_closest(tag_embed: ClothingTagEmbed, n: int):
    FIRST_STAGE_FILTER_RATIO = 10
    CANDIDATE_TO_LIMIT_RATIO = 10
    CLOTHING_TYPE_WEIGHT = 0.7
    COLOR_WEIGHT = 0.3
    
    bucket_count =  get_catalogue_metadata().bucket_count
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
                "color_score": {
                    "$sum": {
                        "$map": {
                            "input": {"$range": [0, {"$size": "$color_embed"}]},
                            "as": "index",
                            "in": {
                                "$multiply": [
                                    {"$arrayElemAt": ["$color_embed", "$$index"]},
                                    {"$arrayElemAt": [tag_embed.color_embed, "$$index"]}
                                ]
                            }
                        }
                    }
                }
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
