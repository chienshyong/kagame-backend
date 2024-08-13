from rembg import remove
from PIL import Image


def remove_bg(image : Image):
    return remove(image)