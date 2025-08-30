# # utils/clip_utils.py

# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch

# # Load CLIP once (so it doesn't reload every request)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def rate_outfit(image_path: str):
#     """
#     Takes an outfit image and returns vibe scores.
#     """

#     # Define some vibes we want to score against
#     vibes = ["casual", "formal", "trendy", "sporty", "traditional", "party"]

#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(text=vibes, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

#     # Return as dictionary { vibe: probability }
#     return {v: float(round(p, 3)) for v, p in zip(vibes, probs)}
# utils/clip_utils.py

# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image
# import torch

# # Load CLIP once (so it doesn't reload every request)
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# def analyze_outfit(image_path: str):
#     """
#     Takes an outfit image and returns vibe scores.
#     """

#     # Define some vibes we want to score against
#     vibes = ["casual", "formal", "trendy", "sporty", "traditional", "party"]

#     image = Image.open(image_path).convert("RGB")
#     inputs = processor(text=vibes, images=image, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

#     # Return as dictionary { vibe: probability }
#     return {v: float(round(p, 3)) for v, p in zip(vibes, probs)}
# utils/clip_utils.py

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load CLIP once (so it doesn't reload every request)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_outfit(image_input):
    """
    Takes an outfit image (file path or Streamlit uploaded file) and returns vibe scores.
    """

    # Define vibes
    vibes = ["casual", "formal", "trendy", "sporty", "traditional", "party"]

    # Handle both path and file-like object
    if isinstance(image_input, str):  
        image = Image.open(image_input).convert("RGB")
    else:  
        image = Image.open(image_input).convert("RGB")

    # Process with CLIP
    inputs = processor(text=vibes, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # Return dict
    return {v: float(round(p, 3)) for v, p in zip(vibes, probs)}
