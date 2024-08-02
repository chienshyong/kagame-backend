
#Takes in a normal image and returns the format it should be stored in the DB
#For now, store images as binary in the DB, so return the binary
#But later, change to storing in a filesystem and return the link

from PIL import Image
from io import BytesIO
import base64

def store_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_image(image_formatted_for_db):
    img_data = base64.b64decode(image_formatted_for_db)
    image = Image.open(BytesIO(img_data))
    return image