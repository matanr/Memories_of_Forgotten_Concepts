import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


def clip_similarity(image_path, text):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Process image and truncated text inputs
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
    #
    # # Process image and text inputs
    # inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)

    # Compute the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    # Compute cosine similarity between image and text embeddings
    similarity = torch.cosine_similarity(image_embeds, text_embeds).item()
    return similarity