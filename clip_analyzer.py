# utils/clip_analyzer.py

import torch
import clip
from PIL import Image

# Load CLIP model once (not every time function runs)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Outfit vibe descriptions
VIBE_TEXTS = [
    "casual outfit",
    "formal outfit",
    "party outfit",
    "sporty outfit",
    "trendy stylish outfit"
]

def analyze_outfit(image_path: str):
    """Returns best vibe and confidence score for the outfit image."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(VIBE_TEXTS).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # similarity
        similarities = (image_features @ text_features.T).squeeze(0)
        probs = similarities.softmax(dim=0)

    # Find top vibe
    best_idx = probs.argmax().item()
    best_vibe = VIBE_TEXTS[best_idx]
    confidence = probs[best_idx].item()

    return best_vibe, round(confidence * 100, 2)
