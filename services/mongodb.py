from pymongo import MongoClient
from services import MONGODB_CONNECTION_STRING
from pydantic import BaseModel
from typing import Literal, List


class CatalogueItem(BaseModel):
    name: str
    category: Literal['Tops', 'Bottoms', 'Shoes', 'Dresses']
    description: str
    embedding: List[float]
    price: float
    image_url: str
    product_url: str
    retailer: str

# TODO(aurel): Add other 'schemas'


# If connection fail, ensure IP address is whitelisted on Atlas
client = MongoClient(MONGODB_CONNECTION_STRING)

try:
    server_info = client.server_info()
    print("Connected to MongoDB server. Server info:")
    print(server_info)
except Exception as e:
    print("Unable to connect to the server.")
    print(e)

db = client.kagame
catalogue = db.catalogue
users = db.users
wardrobe = db.wardrobe
