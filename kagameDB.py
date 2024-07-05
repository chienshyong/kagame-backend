from pymongo import MongoClient

# Connect to the MongoDB server running on localhost and the default port 27017
client = MongoClient(host='mongo', port=27017) # Match the name of the container defined in docker-compose.yml
db = client.KagameDB
catalogue = db.catalogue