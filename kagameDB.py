from pymongo import MongoClient

# Connect to the MongoDB server running on localhost and the default port 27017
client = MongoClient('localhost', 27017)
db = client.KagameDB
catalogue = db.catalogue