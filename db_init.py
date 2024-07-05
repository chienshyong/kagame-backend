## Script to add placeholder images into the DB

from pymongo import MongoClient
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import numpy as np
from IPython.display import Image, display
import os
import PIL
from tqdm import tqdm
from pymongo.errors import DuplicateKeyError

# Connect to the MongoDB server running on localhost and the default port 27017
client = MongoClient('mongo', 27017)
db = client.KagameDB
collection = db.catalogue

fclip = FashionCLIP('fashion-clip')

#Add all placeholder images to DB

# Specify the directory path
folder_path = './images'

# Get all filenames in the folder
filenames = os.listdir(folder_path)

images = ["./images/" + str(k) for k in filenames]

# we create image embeddings and text embeddings
image_embeddings = fclip.encode_images(images, batch_size=32)

# we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

for i in tqdm(range(len(filenames))):
    document = {
        "image_path": filenames[i],
        "retailer": "placeholder",
        "embedding": image_embeddings[i].tolist()
    }

    try:
        insert_result = collection.insert_one(document)
    except DuplicateKeyError as e:
        print(f"Error: {e.details['errmsg']}")