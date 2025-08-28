# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:16:19 2025

@author: senat
"""

import os
import json
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor
import time


# Load JinaCLIP v2
processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
model.eval()

# Define paths
json_path = "..." # Dataset path
image_folder = "..." # Image folder path
output_path = "..." # Embeddings file path

# Results list
embedding_data = []

# Load JSON dataset file
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

start = time.time()

for item in data:
    claim = item["claim"]
    claim_id = item["claim_id"]
    label = item["label"]

    # Find an image which name starts by claim_id
    matching_images = [
        f for f in os.listdir(image_folder)
        if f.startswith(str(claim_id)) and f.lower().endswith((".jpeg"))
    ]
    if not matching_images:
        print(f"No image found for the claim {claim_id}")
        continue
    
    # Choose the first image that starts with claim_id
    image_path = os.path.join(image_folder, matching_images[0])

    # Convert the image in RGB format
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Reading error for {image_path}: {e}")
        continue

    # Preprocessing
    inputs = processor(text=[claim], images=[image], return_tensors="pt", padding=True, truncation=True)


    # Embedding computation
    with torch.no_grad():
        outputs = model(**inputs)

    # Storage in the results list
    embedding_data.append({
        "claim_id": claim_id,
        "label": label,
        "text_emb": outputs.text_embeds.squeeze(0).tolist(),
        "image_emb": outputs.image_embeds.squeeze(0).tolist()
    })
    
    print(f"Claim {claim_id} encoded.")
    
end = time.time()

total_time = (start-end)/3600

print(f" {len(embedding_data)} text-image pairs encoded.")
print(f" This took {total_time} hours")

# Sauvegarde dans un fichier JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(embedding_data, f, ensure_ascii=False, indent=2)

print(f"Embeddings saved in : {output_path}")
