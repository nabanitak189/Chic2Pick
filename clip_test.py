from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load pre-trained CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load your test image
image = Image.open("test_outfit.jpg")

# Define vibe categories
texts = ["a stylish outfit", "a casual outfit", "an outdated outfit", "a cool outfit"]

# Process
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Get similarity
logits_per_image = outputs.logits_per_image  # image-text similarity
probs = logits_per_image.softmax(dim=1)      # convert to % scores

# Print results
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.2f}")

# Derive vibe score (0-10) using "stylish" & "cool"
vibe_score = (probs[0][0] + probs[0][3]) * 10
print(f"\n✨ Outfit Vibe Score: {vibe_score:.1f}/10")
