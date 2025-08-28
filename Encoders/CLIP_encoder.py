import os
import sys
import torch
from PIL import Image
import json
from tqdm import tqdm
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip
import preprocessing as prep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose the backbone you want to use
parser = argparse.ArgumentParser(description='Extract CLIP features from updated_mmvc_all.json')
parser.add_argument('-c', '--clip', type=str, default='vit16', help='rn504 | rn50 | vit16')
args = parser.parse_args()

clip_nm = args.clip
model_nms = {'vit16': 'ViT-B/16', 'rn50': 'RN50', 'rn504': 'RN50x4'}
model_dim = {'vit16': 512, 'rn50': 1024, 'rn504': 640}

print(f'----------------- Extracting features using: {model_nms[clip_nm]} -----------------')
model, img_preprocess = clip.load(model_nms[clip_nm], device=device)
model.eval()

_tokenizer = _Tokenizer()
text_processor = prep.get_text_processor(word_stats='twitter', keep_hashtags=True)

# Define the path to the dataset you want to encode
dataset_path = '...'

# Load data
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")
print(data[0])  # preview first item

im_names, img_feats, text_feats = [], [], []

image_extensions = ['.jpg', '.jpeg']


# Create a placeholder image if no image is found
def create_placeholder_image():
    return Image.new('RGB', (450, 600), color=(255, 255, 255))

start_time=time.time()
idm=0

for entry in tqdm(data):
    claim = entry['claim']
    image_ids = entry['claim_id']  

    proc_text = prep.process_tweet(claim, text_processor)

    text_tokens = prep.clip_tokenize(proc_text, _tokenizer).to(device)

    idm = idm+1
    if image_ids == -1:
        print(f"Warning: No image evidence found for claim '{claim}'. Using placeholder...")
        img = create_placeholder_image()  
        img_id = f"{idm}_placeholder" 
    else:
        img_path = None
        for fname in os.listdir('...'): # Define the path to the folder with the images of the dataset
            parts = fname.split('_')
            if len(parts) >= 2 and parts[0] == str(image_ids) and fname.endswith(tuple(image_extensions)):
                img_path = os.path.join('...', fname) # Define the path to the folder with the images of the dataset
                break 
        if img_path is None:
            print(f"Warning: No valid image found for ID {image_ids}. Using placeholder...")
            img = create_placeholder_image()
            img_id = f"{idm}_placeholder"
        else:
            try:
                img = Image.open(img_path)
                img = img.convert("RGB") 
                img_id = image_ids
                print(f"Success: Image '{img_id}' found and processed.")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Using placeholder...")
                img = create_placeholder_image()
                img_id = idm

    image_tensor = img_preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

    img_feats.append(image_features.cpu().numpy().flatten().tolist())
    text_feats.append(text_features.cpu().numpy().flatten().tolist())
    im_names.append(img_id) 

feat_dict = {
    'img_feats': {name: feat for name, feat in zip(im_names, img_feats)},
    'text_feats': {name: feat for name, feat in zip(im_names, text_feats)}
}

end_time = time.time()
extraction_time = (end_time-start_time)/3600

# Save the extracted features to a JSON file. Rename the file depending on the model and the dataset
output_path = 'model_dataset_embeddings.json'
with open(output_path, 'w') as outfile:
    json.dump(feat_dict, outfile)

print(f"Extracted features saved to {output_path}")
print(f"Extraction time : {extraction_time} hours")
