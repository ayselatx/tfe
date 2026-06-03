import os
import pickle
import torch
import numpy as np
from captum.attr import IntegratedGradients, Saliency, GradientShap, Occlusion
from PIL import Image

from utils.models_registry import VISION_EXPLAIN_MODELS
from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)

import hashlib

def hash_image(pil_img):
    return hashlib.md5(pil_img.tobytes()).hexdigest()



# -----------------------------
# Load + transform image
# -----------------------------
def load_image_for_xai(path, transform):
    try:
        img = Image.open(path).convert("RGB")
    except:
        img = Image.new("RGB", (224, 224), (0, 0, 0))

    tensor = transform(img)
    return tensor, np.array(img)


# -----------------------------
# Attribution → heatmap
# -----------------------------
def attr_to_heatmap(attr):
    a = attr.detach().cpu().numpy()
    a = np.squeeze(a)

    if a.ndim == 3:
        a = a.mean(axis=0)

    a = np.abs(a)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a


# -----------------------------
# Unnormalize image
# -----------------------------
def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


# -----------------------------
# Load Captum record
# -----------------------------
def load_captum_record(dataset, model_name, idx):
    path = os.path.join(
        vision_xai_dir(dataset, model_name),
        "captum_records.pkl"
    )
    with open(path, "rb") as f:
        records = pickle.load(f)

    return next(r for r in records if r["idx"] == idx)


# -----------------------------
# Main function for Streamlit
# -----------------------------
def get_vision_explanations(model_name, img_path, dataset="Flickr8k"):
    img_name = os.path.basename(img_path)

    # CASE 1 — Uploaded image (not in dataset)
    if img_path not in IMAGE_PATHS:
        pil_img = Image.open(img_path).convert("RGB")
        return compute_attributions_on_the_fly(model_name, pil_img)

    # CASE 2 — Dataset image → load precomputed Captum
    if model_name == "ResNet50": model_name = "resnet50"
    elif model_name == "MobileNetV3": model_name = "mobilenet_v3"
    elif model_name == "ViT": model_name = "vit"
    elif model_name == "PVT": model_name = "pvt"
    model, transform = VISION_EXPLAIN_MODELS[model_name]("cpu")

    img_tensor, img_np = load_image_for_xai(img_path, transform)

    idx = next(i for i, p in enumerate(IMAGE_PATHS)
               if os.path.basename(p) == img_name)

    rec = load_captum_record(dataset, model_name, idx)

    methods = ["IG", "Saliency", "GradShap", "Occlusion"]
    heatmaps = {m: attr_to_heatmap(rec[m]) for m in methods}

    return unnormalize(img_tensor), heatmaps


def normalize_attr(a):
    a = np.abs(a)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def to_numpy(attr):
    return attr.detach().cpu().numpy().squeeze()

def compute_attributions_on_the_fly(model_name, pil_image):
    # Hash uploaded image
    img_hash = hash_image(pil_image)
    save_dir = f"streamlit_app/data/uploaded_xai/{model_name}/{img_hash}"
    os.makedirs(save_dir, exist_ok=True)

    # If already computed → load from disk
    cache_file = os.path.join(save_dir, "heatmaps.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Load model + transform
    model, transform = VISION_EXPLAIN_MODELS[model_name]("cpu")
    model.eval()

    # Transform image
    img_t = transform(pil_image).unsqueeze(0)

    # Forward pass to get predicted label
    with torch.no_grad():
        logits = model(img_t)
        label = logits.argmax(dim=1).item()

    # Captum methods
    ig  = IntegratedGradients(model)
    sal = Saliency(model)
    gs  = GradientShap(model)
    occ = Occlusion(model)

    ig_attr  = ig.attribute(img_t, target=label)
    sal_attr = sal.attribute(img_t, target=label)
    gs_attr  = gs.attribute(img_t, baselines=torch.zeros_like(img_t), target=label)
    occ_attr = occ.attribute(
        img_t,
        target=label,
        sliding_window_shapes=(3, 15, 15),
        strides=(3, 8, 8)
    )

    # Convert to heatmaps
    heatmaps = {
        "IG": normalize_attr(to_numpy(ig_attr)),
        "Saliency": normalize_attr(to_numpy(sal_attr)),
        "GradShap": normalize_attr(to_numpy(gs_attr)),
        "Occlusion": normalize_attr(to_numpy(occ_attr)),
    }

    # ORIGINAL IMAGE (this was missing!)
    orig_np = np.array(pil_image)

    # Save to disk
    result = (orig_np, heatmaps)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result
