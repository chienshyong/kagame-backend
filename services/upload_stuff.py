from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from services.mongodb import catalogue, CatalogueItem
from secretstuff.secret import OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJ_ID

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJ_ID)

# Customize the prompt. Like force it to always describe if the outfit is slimfit/loosefit/etc, the color, etc.
prompt = '''
You are a fashion stylist specialized in giving detailed descriptions of clothing items.
You will be provided with an image, the title of the item depicted in the image, and the type of clothing item (either 'Tops', 'Bottoms', 'Shoes' or 'Dresses'), and your goal is to provide descriptions for only the item specified. 
Keep your description concise and less than 50 words.
'''


def add_catalogue_item(name: str, category: Literal['Tops', 'Bottoms', 'Shoes', 'Dresses'], price: float, image_url: str, product_url: str, retailer: str):
    description = get_openai_description(name, category, image_url)
    embedding = get_openai_embedding(description)
    catalogue_item = CatalogueItem(name=name, category=category, description=description, embedding=embedding,
                                   price=price, image_url=image_url, product_url=product_url, retailer=retailer)
    catalogue.insert_one(dict(catalogue_item))


def get_openai_description(name: str, category: Literal['Tops', 'Bottoms', 'Shoes', 'Dresses'], image_url: str) -> str:
    output = client.chat.completions.create(model="gpt-4o-mini",
                                            messages=[
                                                {"role": "system", "content": prompt},
                                                {"role": "user", "content": [
                                                    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}}]},
                                                {"role": "user", "content": name},
                                                {"role": "user", "content": category}
                                            ],
                                            stream=False
                                            )
    # Since default n = 1, we'll only always need to first element
    return output.choices[0].message.content


def get_openai_embedding(description: str):
    embedding = client.embeddings.create(input=description, model="text-embedding-3-large").data[0].embedding
    return embedding
