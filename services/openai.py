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


class ClothingTag(BaseModel):  # For catalogue
    clothing_type: str
    color: str
    material: str
    other_tags: list[str]


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


def get_n_closest(tag_embed: ClothingTagEmbed, n: int):
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

from pydantic import BaseModel
from typing import List

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

def get_wardrobe_recommendation(tag: WardrobeTag, additional_prompt: str = "") -> list[ClothingTag]:
    tag_dict = tag.model_dump()
    if additional_prompt != "":
        tag_dict["additional_prompt"] = additional_prompt

    prompt = json.dumps(tag_dict)
    openai_output = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Given a description of one clothing item in JSON, recommend complementary clothes to complete the outfit.\n\nFollow these steps to recommend an outfit:\n1. If the given item is a top (e.g., tee shirt, polo, shirt, dress, tank top), give recommendations for other outfit pieces like bottoms (e.g., pants, shorts, trousers, jeans, skirts, leggings).\n2. If you are confident in layering clothes with the given item, recommend clothes that can be layered on top of the top (e.g., overshirt, sweater, jacket); if additional context provided in the prompt discourages layering (e.g., summer, hot, lightweight), don't suggest a layer.\n\nAdditional constraints or style prompts might be included to tailor outfit recommendations further.\n\nCompile the output into a JSON that contains the description of all the items in the completed outfit. In case there is nothing to layer with, leave the layer tag with an empty string. Do not output the original clothing item. (e.g., if you're given a tee shirt, don't output the tee shirt as part of the outfit).\n Avoid recommending duplicate clothing categories (Tops/Bottoms/Layers). For instance, if shorts are already recommended, do not also recommend skirts or jeans, as both cannot be worn simultaneously."}
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
