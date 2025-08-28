# TIGER_M4FC

Advancing multimodal fact-checking against climate misinformation.

This repository contains code and resources for multimodal fact-checking using textual claims and images.

---

## Datasets

This folder contains two datasets, **ClimateFever** and **TIGER**, in JSON format. These files contain the textual claims for each dataset.  

To download images and build a multimodal dataset, use the `Image_Scraper.py` script. You may need to run it multiple times to pair all claims with an image, as some downloads may fail.  

If necessary, you can manually download images for the claims that don't have an associated image.

---

## Encoders

This folder contains two encoders to transform image-text pairs into embeddings.  

For the **CLIP encoder**, you can choose the backbone directly when running the script:  

```bash
python CLIP_encoder.py -c <backbone>
````
Replace <backbone> with the desired model.

## Models

This folder contains four models, each ready to use on an embedding file.

If you use the Deep Fusion MLP, you must first select the backbone used with CLIP by modifying the CLIP_BACKBONE variable at line 26 of the script. This ensures consistent file and folder naming.
