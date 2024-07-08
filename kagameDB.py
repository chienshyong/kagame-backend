from pymongo import MongoClient
import keys.mongodb
from PIL import Image
import base64
import io

# If connection fail, ensure IP address is whitelisted on Atlas
client = MongoClient(keys.mongodb.CONNECTION_STRING)

try:
    server_info = client.server_info()
    print("Connected to MongoDB server. Server info:")
    print(server_info)
except Exception as e:
    print("Unable to connect to the server.")
    print(e)
    
db = client.kagame
catalogue = db.catalogue

#Get an image from the DB
def get_image(image_path):
    document = catalogue.find_one({"image_path": image_path})
    if document and 'image_data' in document:
        # Assuming 'image_data' is the field where the binary data is stored
        image_binary = document['image_data']
        
        # Convert binary data to image
        image = Image.open(io.BytesIO(image_binary))
        # Create an in-memory file object
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)
        return img_io
    else:
        return None