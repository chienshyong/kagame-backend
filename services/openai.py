from openai import OpenAI
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID
from PIL import Image
from pydantic import BaseModel
import json
from io import BytesIO
import base64

# Hardcoded available categories
category_labels = ["Tops", "Bottoms", "Dresses", "Shoes", "Jackets", "Accessories"]

# Fixed formats which GPT4 will force to return


class ClothingTags(BaseModel):  # For catalogue
    clothing_type: str
    color: str
    material: str
    other: list[str]


class WardrobeTags(BaseModel):  # For wardrobe
    name: str
    category: str
    tags: list[str]


openai_client = OpenAI(
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJ_ID,
    api_key=OPENAI_API_KEY
)

# Generate tags from user uploaded image


def generate_tags(image_url: str):
    prompt = f"Give a name description of this clothing item (5 words or less), choose category from {category_labels}, and tag with other adjectives (eg. color, material, occasion, fit, sleeve, brand)"
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
        response_format=WardrobeTags,
    )

    return json.loads(output.choices[0].message.content)
