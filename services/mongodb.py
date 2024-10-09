from pymongo import MongoClient
from services import MONGODB_CONNECTION_STRING

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
