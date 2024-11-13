import numpy as np
import tempfile
from PIL import Image

# False so server startup isn't as slow when we're doing development.
load_fashionclip = False

if (load_fashionclip):
    from fashion_clip.fashion_clip import FashionCLIP
    fclip = FashionCLIP('fashion-clip')

category_labels = ["Tops", "Bottoms", "Outerwear", "Dresses", "Underwear",
                   "Activewear", "Sleepwear", "Accessories", "Footwear", "Glasses", "Jacket", "Socks"]
color_labels = ["Red", "Blue", "Yellow", "Green", "Orange",
                "Purple", "Black", "White", "Gray", "Brown", "Beige", "Pink"]
adjective_labels = ["stylish", "elegant", "comfortable", "trendy", "casual", "formal", "chic", "vintage", "modern",
                    "sleek", "vibrant", "classic", "sporty", "luxurious", "fitted", "loose", "colorful", "monochrome", "bold", "subtle"]

category_labels_prompt = [f"A product picture of {k}" for k in category_labels]
color_labels_prompt = [f"A {k} piece of clothing" for k in color_labels]
adjective_labels_prompt = [f"A {k} piece of clothing" for k in adjective_labels]

if (load_fashionclip):
    category_label_embeddings = fclip.encode_text(category_labels_prompt, batch_size=32)
    category_label_embeddings = category_label_embeddings / \
        np.linalg.norm(category_label_embeddings, ord=2, axis=-1, keepdims=True)
    color_label_embeddings = fclip.encode_text(color_labels_prompt, batch_size=32)
    color_label_embeddings = color_label_embeddings / \
        np.linalg.norm(color_label_embeddings, ord=2, axis=-1, keepdims=True)
    adjective_label_embeddings = fclip.encode_text(adjective_labels_prompt, batch_size=32)
    adjective_label_embeddings = adjective_label_embeddings / \
        np.linalg.norm(adjective_label_embeddings, ord=2, axis=-1, keepdims=True)


def generate_tags(image: Image):
    if (not load_fashionclip):
        print("Warning: Fashionclip is disabled. fashion_clip.generate_tags is thus disabled.")
        return {"category": "NIL", "color": "NIL", "description": "NIL"}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        image.save(tmp, format='PNG')
        images = [tmp.name]

        # we create image embeddings and text embeddings
        image_embeddings = fclip.encode_images(images, batch_size=32)

        # we normalize the embeddings to unit norm (so that we can use dot product instead of cosine similarity to do comparisons)
        image_embeddings = image_embeddings/np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)

        category_predicted_classes_distribution = category_label_embeddings.dot(image_embeddings.T)
        color_predicted_classes_distribution = color_label_embeddings.dot(image_embeddings.T)
        adjective_predicted_classes_distribution = adjective_label_embeddings.dot(image_embeddings.T)

        weights = category_predicted_classes_distribution[:, 0]
        cat_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:3]

        weights = color_predicted_classes_distribution[:, 0]
        color_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:3]

        weights = adjective_predicted_classes_distribution[:, 0]
        desc_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:8]

        res = {"category": [category_labels[i] for i in cat_indices], "color": [color_labels[i]
                                                                                for i in color_indices], "description": [adjective_labels[i] for i in desc_indices]}
        return res
