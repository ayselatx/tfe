#!/usr/bin/env python
# coding: utf-8

# # Imports

# ## Standard Library

# In[ ]:


import os          # file paths, directory management
import sys         # system-level operations (rarely used, but useful for debugging)
import time        # timing execution (performance measurement)
import gc          # garbage collection (manual memory cleanup)
import csv         # CSV file handling (reading/writing tabular data)


# ## Data Handling & Numerical Computation

# In[ ]:


import numpy as np         # numerical operations, arrays, embeddings
import pandas as pd        # dataset loading and manipulation (DataFrames)


# ## System Monitoring & External Communication

# In[ ]:


import psutil              # monitor RAM/CPU usage (GreenAI metrics)
import requests            # send notifications (by NTFY)


# ## Image Processing & Computer Vision

# In[ ]:


from PIL import Image      # image loading
import cv2                 # OpenCV (advanced image processing if needed)


# ## Progress & Visualization

# In[ ]:


from tqdm.notebook import tqdm   # progress bars

import matplotlib.pyplot as plt  # plotting results
import seaborn as sns            # statistical visualizations


# ## Statistics / Evaluation

# In[ ]:


from scipy.stats import spearmanr   # rank correlation (similarity preservation)
import pickle 


# ## PyTorch (Core Deep Learning Framework)

# In[ ]:


import torch
import torch.nn as nn              # neural network layers


# ## TorchVision (Pretrained Vision Models & Transforms)

# In[ ]:


from torchvision import models, transforms


# ## Transformers (HuggingFace Models)

# In[ ]:


from transformers import (
    # Vision Transformers
    ViTModel, ViTImageProcessor,
    DeiTModel, DeiTImageProcessor,

    # Generic auto models (flexible loading)
    AutoImageProcessor, AutoModel,

    # CLIP (multimodal)
    CLIPProcessor, CLIPVisionModel, CLIPTextModel,

    # Text models
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    GPT2Tokenizer, GPT2Model
)


# ## Explainable Libraries

# In[ ]:


import captum
from captum.attr import IntegratedGradients, Saliency, Occlusion, DeepLift, GradientShap

import shap

import quantus


# In[ ]:


import urllib.request
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


# # Configuration

# ## Dataset Selection

# In[ ]:


# Available datasets: Flickr8k, Flickr30k, ConceptualCaptions
CURRENT_DATASET = "Flickr8k" 
ALL_DATASETS = ["Flickr8k", "Flickr30k", "ConceptualCaptions"]

assert CURRENT_DATASET in ALL_DATASETS, f"{CURRENT_DATASET} is not a valid dataset"


# ## Directory Architecture

# In[ ]:


BASE_DIR = os.path.join(os.getcwd(), 'TFE_Data')

DATASETS_DIR = os.path.join(BASE_DIR, 'Datasets')
#RAW_DIR = os.path.join(BASE_DIR, 'Features_RAW', CURRENT_DATASET)
#RESULTS_DIR = os.path.join(BASE_DIR, 'Unimodal_VisionXAI_Results', CURRENT_DATASET)
"""

def get_raw_dir(dataset, modality):
    path = os.path.join(BASE_DIR, "Features", "raw", dataset, modality)
    os.makedirs(path, exist_ok=True)
    return path

def get_indexed_dir(dataset, modality, model):
    path = os.path.join('TFE_Data', "Results_Unimodal", dataset, modality, model)
    os.makedirs(path, exist_ok=True)
    return path

def load_embeddings(dataset, modality, model):
    file_path = os.path.join('TFE_Data', "Results_Unimodal", dataset, modality, model, "image_tensors.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing RAW image_tensors: {file_path}")
    return np.load(file_path)

def get_results_dir(dataset):
    path = os.path.join(BASE_DIR, "Unimodal_VisionXAI_Results", dataset)
    os.makedirs(path, exist_ok=True)
    return path

"""


# In[ ]:


PLOT_DIR = "XAI_Plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# In[ ]:


def get_indexed_dir(dataset, modality, model):
    path = os.path.join('TFE_Data', "Results_Unimodal", dataset, modality, model)
    os.makedirs(path, exist_ok=True)
    return path


# ## Device Configuration

# In[ ]:


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Environment initialized. Using device: {device}")


# ## Notification Setup

# In[ ]:


def send_ntfy_notification(message, title="TFE Pipeline Update"):
    """Sends a push notification via ntfy.sh."""
    NTFY_TOPIC = "aysel_tfe_server_9988"
    try:
        requests.post(
            f"https://ntfy.sh/{'aysel_tfe_server_9988'}",
            data=message.encode(encoding='utf-8'),
            headers={"Title": title}
        )
    except Exception as e:
        print(f"Notification failed: {e}")



# # Data Loading

# In[ ]:


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            try:
                image = self.transform(image)
            except:
                image = torch.zeros((3, 224, 224))

        return image


# In[ ]:


## TFE_Data/Results_Unimodal/Flickr8k/vision/clip_vision/embeddings.npy

def load_embeddings(dataset, modality, model):
    """Load embeddings from the dataset folder structure."""
    file_path = os.path.join(BASE_DIR, "Results_Unimodal", dataset, modality, model, "embeddings.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing RAW embeddings: {file_path}")
    return np.load(file_path)


# # Models

# In[ ]:


def get_resnet50_model(device):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).to(device).eval()
    transform = weights.transforms()
    return model, transform

def get_mobilenet_v3_model(device):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights).to(device).eval()
    transform = weights.transforms()
    return model, transform

class VitWithHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k'
        ).to(device).eval()
        self.head = nn.Linear(self.backbone.config.hidden_size, 1000).to(device)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)

def get_vit_model(device):
    model = VitWithHead(device).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, transform


class DeitWithHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = DeiTModel.from_pretrained(
            'facebook/deit-base-distilled-patch16-224'
        ).to(device).eval()
        self.head = nn.Linear(self.backbone.config.hidden_size, 1000).to(device)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)

def get_deit_model(device):
    model = DeitWithHead(device).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, transform


class CLIPWithHead(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device).eval()
        self.head = nn.Linear(self.backbone.config.hidden_size, 1000).to(device)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        pooled = out.pooler_output
        return self.head(pooled)

def get_clip_vision_model(device):
    model = CLIPWithHead(device).to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return model, transform


# # Explanation 

# ## Captum

# ### Saliency 

# In[ ]:


def explain_saliency(model, image_tensor, label):
    """Compute Saliency attribution for a single image_tensor."""
    sal = Saliency(model)
    attribution = sal.attribute(image_tensor.unsqueeze(0), target=label)
    return attribution


# ### Integrated Gradient

# In[ ]:


def explain_ig(model, image_tensor, label):
    """Compute Integrated Gradients attribution for a single image_tensor."""
    ig = IntegratedGradients(model)
    attribution = ig.attribute(image_tensor.unsqueeze(0), target=label)
    return attribution


# ### GradientSHAP

# In[ ]:


def explain_gradientShap(model, image_tensor, label):
    """Compute GradientShap attribution for a single image_tensor."""
    gs = GradientShap(model)
    attribution = gs.attribute(
        image_tensor.unsqueeze(0),
        baselines=torch.zeros_like(image_tensor).unsqueeze(0),
        target=label
    )
    return attribution


# ### Occlusion

# In[ ]:


def explain_occlusion(model, image_tensor, label):
    occlusion = Occlusion(model)
    attribution = occlusion.attribute(
        image_tensor.unsqueeze(0),
        target=label,
        sliding_window_shapes=((3, 15, 15),),   # (channels, h, w)
        strides=((3, 8, 8),)                    # optional but recommended
    )
    return attribution


# ## SHAP Explanations

# In[ ]:


#SHAP DeepExplainer passes inputs as a list, but your model expects a single tensor.
class SHAPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = x.float()
        return self.model(x)


# In[ ]:


def explain_shap(model, image_tensor, background, model_name):
    shap_model = SHAPWrapper(model)

    # CNNs → GradientExplainer
    if "resnet" in model_name or "mobilenet" in model_name:
        explainer = shap.GradientExplainer(shap_model, background)
    else:
        explainer = shap.DeepExplainer(shap_model, background)

    shap_values = explainer.shap_values(image_tensor.unsqueeze(0))
    return torch.tensor(shap_values)


# # Evaluation

# ## Quantus

# In[ ]:


from quantus.metrics.faithfulness.infidelity import Infidelity
from quantus.metrics.robustness.max_sensitivity import MaxSensitivity
from quantus.metrics.complexity.sparseness import Sparseness
from quantus.metrics.faithfulness.sensitivity_n import SensitivityN



# In[ ]:


def evaluate_quantus(model, img, label, attr):
    metrics = [
        quantus.Infidelity(),
        quantus.MaxSensitivity(),
        quantus.Sparseness(),
        quantus.SensitivityN()
    ]

    results = {}
    for m in metrics:
        try:
            results[m.__class__.__name__] = m(
                model=model,
                x_batch=img.unsqueeze(0),
                a_batch=attr.unsqueeze(0),
                y_batch=torch.tensor([label])
            )
        except TypeError:
            results[m.__class__.__name__] = m(
                model=model,
                x=img.unsqueeze(0),
                a=attr.unsqueeze(0),
                y=torch.tensor([label])
            )
        except Exception as e:
            print(f"[{ts()}] Warning: {m.__class__.__name__} failed — {e}", flush=True)
            results[m.__class__.__name__] = np.nan
    return results


# In[ ]:


quantus.AVAILABLE_METRICS


# # Save Results

# In[ ]:


def save_attribution_tensor(tensor, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(tensor.cpu(), out_path)


# In[ ]:


def save_quantus_csv(all_metrics, out_path="TFE_Data/Unimodal/XAI/vision/quantus_results.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "dataset", "model", "image_index", "method", "metric", "value"])

        for entry in all_metrics:
            dataset = entry["dataset"]
            model = entry["model"]
            results = entry["results"]

            for idx, res in enumerate(results):
                for method, metrics_dict in res.items():
                    for metric_name, metric_value in metrics_dict.items():
                        try:
                            writer.writerow([
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                dataset,
                                model,
                                idx,
                                method,
                                metric_name,
                                float(metric_value)
                            ])
                        except Exception as e:
                            print(f"[{ts()}] Warning: Failed to write {model} | {dataset} | img {idx} | {method} | {metric_name}: {e}", flush=True)
                            continue

    print(f"[{ts()}] Quantus CSV saved successfully → {out_path}", flush=True)


# # Visualisation

# ## Heatmap Visualisation (Captum and SHAP)

# In[ ]:


def visualize_heatmap(attribution, original_img, title, name, ds_name, idx):
    attr = attribution.detach().cpu().numpy()
    if attr.ndim == 3:
        attr = np.mean(attr, axis=0)

    img = original_img.detach().cpu().numpy().transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(attr, cmap='jet', alpha=0.45)
    plt.title(title)
    plt.axis("off")

    out_path = os.path.join(PLOT_DIR, f"{name}_{ds_name}_img{idx}_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[{ts()}] Saved plot → {out_path}", flush=True)


# ## SHAP visualisation

# In[ ]:


def visualize_shap(shap_values, original_img, name, ds_name, idx):
    vals = shap_values.squeeze().detach().cpu().numpy()
    vals = np.mean(vals, axis=0)

    img = original_img.detach().cpu().numpy().transpose(1,2,0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.imshow(vals, cmap='coolwarm', alpha=0.45)
    plt.title("SHAP Attribution")
    plt.axis("off")

    out_path = os.path.join(PLOT_DIR, f"{name}_{ds_name}_img{idx}_shap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[{ts()}] Saved plot → {out_path}", flush=True)


# ## Side by Side for each image

# In[ ]:


def visualize_side_by_side(original_img, ig, sal, gs, occ, shap_vals, title, name, ds_name, idx):
    img_np = original_img.detach().cpu().numpy().transpose(1,2,0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    def prep(attr):
        a = attr.detach().cpu().numpy()
        a = np.squeeze(a)
        if a.ndim == 4 and a.shape[0] > 10: a = a.mean(axis=0)
        if a.ndim == 4 and a.shape[-1] > 10: a = a.mean(axis=-1)
        if a.ndim == 3 and a.shape[0] > 10: a = a.mean(axis=0)
        if a.ndim == 3 and a.shape[0] in [1,3]: a = a.mean(axis=0)
        if a.ndim == 3 and a.shape[-1] > 10: a = a.mean(axis=-1)
        return a

    ig_np   = prep(ig)
    sal_np  = prep(sal)
    gs_np   = prep(gs)
    occ_np  = prep(occ)
    shap_np = prep(shap_vals)

    fig, axs = plt.subplots(2, 3, figsize=(14, 9))
    axs = axs.flatten()

    axs[0].imshow(img_np); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(img_np); axs[1].imshow(ig_np, cmap="jet", alpha=0.45); axs[1].set_title("IG"); axs[1].axis("off")
    axs[2].imshow(img_np); axs[2].imshow(sal_np, cmap="jet", alpha=0.45); axs[2].set_title("Saliency"); axs[2].axis("off")
    axs[3].imshow(img_np); axs[3].imshow(gs_np, cmap="jet", alpha=0.45); axs[3].set_title("GradientShap"); axs[3].axis("off")
    axs[4].imshow(img_np); axs[4].imshow(occ_np, cmap="jet", alpha=0.45); axs[4].set_title("Occlusion"); axs[4].axis("off")
    axs[5].imshow(img_np); axs[5].imshow(shap_np, cmap="coolwarm", alpha=0.45); axs[5].set_title("SHAP"); axs[5].axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    out_path = os.path.join(PLOT_DIR, f"{name}_{ds_name}_img{idx}_side_by_side.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[{ts()}] Saved plot → {out_path}", flush=True)


# ## Per model summary

# In[ ]:


def summarize_model_attributions(model_name, all_attrs):
    """
    all_attrs: list of dicts with keys:
        IG, Saliency, GradientShap, Occlusion, SHAP
    """

    # Average each method across images
    def avg_attr(key):
        tensors = [a[key] for a in all_attrs]
        stacked = torch.stack([t.mean(dim=0) if t.ndim == 3 else t for t in tensors])
        return stacked.mean(dim=0).detach().cpu().numpy()

    ig_avg   = avg_attr("IG")
    sal_avg  = avg_attr("Saliency")
    gs_avg   = avg_attr("GradientShap")
    occ_avg  = avg_attr("Occlusion")
    shap_avg = avg_attr("SHAP")

    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    maps = [ig_avg, sal_avg, gs_avg, occ_avg, shap_avg]
    titles = ["IG", "Saliency", "GradientShap", "Occlusion", "SHAP"]

    for ax, m, t in zip(axs, maps, titles):
        ax.imshow(m, cmap="jet")
        ax.set_title(t)
        ax.axis("off")

    plt.suptitle(f"{model_name} — Average Attribution Summary", fontsize=16)
    plt.tight_layout()
    plt.show()


# # Execution

# ## Debug of Full Mode

# In[ ]:


# ============================
# XAI MODE SWITCHES
# ============================

MODE = "C"   # "A" = full, "B" = balanced, "C" = fast

if MODE == "A":
    MAX_SHAP_IMAGES = None
    MAX_QUANTUS_IMAGES = None
    MAX_VISUAL_IMAGES = None
elif MODE == "B":
    MAX_SHAP_IMAGES = 32
    MAX_QUANTUS_IMAGES = 100
    MAX_VISUAL_IMAGES = 20
elif MODE == "C":
    MAX_SHAP_IMAGES = 8
    MAX_QUANTUS_IMAGES = 20
    MAX_VISUAL_IMAGES = 5
else:
    raise ValueError("Invalid MODE")


# In[ ]:


models = {
    "resnet50": get_resnet50_model(device),
    "mobilenet_v3": get_mobilenet_v3_model(device),
    "vit": get_vit_model(device),
    "deit": get_deit_model(device),
    "clip_vision": get_clip_vision_model(device)
}


# In[ ]:


from datetime import datetime
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============================
# FAST DEBUG MODE
# ============================

FAST_DEBUG = True

if FAST_DEBUG:
    MAX_IMAGES = 10          # total images to explain
    MAX_SHAP_IMAGES = 2      # SHAP is slow → keep tiny
    MAX_QUANTUS_IMAGES = 5   # Quantus is slow → keep small
    MAX_VISUAL_IMAGES = 2    # matplotlib is slow
    SHAP_BACKGROUND = 2      # background images for SHAP
else:
    MAX_IMAGES = None
    MAX_SHAP_IMAGES = None
    MAX_QUANTUS_IMAGES = None
    MAX_VISUAL_IMAGES = None
    SHAP_BACKGROUND = 8

# ============================
# MAIN XAI EXECUTION LOOP
# ============================

all_metrics = []

for ds_name in ALL_DATASETS:

    df_path = os.path.join(DATASETS_DIR, f"df_{ds_name}.pkl")
    if not os.path.exists(df_path):
        print(f"[{ts()}] Skipping {ds_name}: Metadata not found.", flush=True)
        continue

    df = pd.read_pickle(df_path)
    IMAGE_PATHS = df["image_path"].tolist()

    for name, (model, transform) in models.items():

        print(f"\n[{ts()}] === Starting {name} on {ds_name} ===", flush=True)
        send_ntfy_notification(f"XAI: Starting {name} on {ds_name}")

        dataset = ImageDataset(IMAGE_PATHS, transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # ----------------------------
        # SHAP BACKGROUND (FAST)
        # ----------------------------
        print(f"[{ts()}] Building SHAP background ({SHAP_BACKGROUND} images)...", flush=True)
        send_ntfy_notification(f"XAI: Building SHAP background for {name}")

        bckgrd = []
        for i, img in enumerate(loader):
            if i >= SHAP_BACKGROUND:
                break
            bckgrd.append(img.squeeze(0).to(device))
        bckgrd = torch.stack(bckgrd)

        model_results = []
        model_attr_collection = []

        torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------
        # MAIN XAI LOOP
        # ----------------------------
        for idx, img in enumerate(tqdm(loader)):

            if FAST_DEBUG and idx >= MAX_IMAGES:
                print(f"[{ts()}] Reached MAX_IMAGES={MAX_IMAGES}, stopping early.", flush=True)
                break

            print(f"[{ts()}] Processing image {idx}", flush=True)
            send_ntfy_notification(f"XAI: {name} | {ds_name} | image {idx}")

            img = img.squeeze(0).to(device)

            # Forward pass
            logits = model(img.unsqueeze(0))
            label = logits.argmax(dim=1).item()

            # ----------------------------
            # XAI METHODS
            # ----------------------------
            print(f"[{ts()}] IG...", flush=True)
            ig_attr = explain_ig(model, img, label)

            print(f"[{ts()}] Saliency...", flush=True)
            sal_attr = explain_saliency(model, img, label)

            print(f"[{ts()}] GradientShap...", flush=True)
            gs_attr = explain_gradientShap(model, img, label)

            print(f"[{ts()}] Occlusion...", flush=True)
            occ_attr = explain_occlusion(model, img, label)

            # SHAP (limited)
            if idx < MAX_SHAP_IMAGES:
                print(f"[{ts()}] SHAP...", flush=True)
                shap_attr = explain_shap(model, img, bckgrd, name)
            else:
                shap_attr = torch.zeros_like(ig_attr)

            # ----------------------------
            # VISUALIZATION (limited)
            # ----------------------------
            if idx < MAX_VISUAL_IMAGES:
                print(f"[{ts()}] Visualizing...", flush=True)
                visualize_side_by_side(
                    img,
                    ig_attr,
                    sal_attr,
                    gs_attr,
                    occ_attr,
                    shap_attr,
                    title=f"{name} — Image {idx}",
                    name=name,
                    ds_name=ds_name,
                    idx=idx
                )

            # ----------------------------
            # STORE ATTRIBUTIONS
            # ----------------------------
            model_attr_collection.append({
                "IG": ig_attr,
                "Saliency": sal_attr,
                "GradientShap": gs_attr,
                "Occlusion": occ_attr,
                "SHAP": shap_attr
            })


            # ----------------------------
            # QUANTUS (limited)
            # ----------------------------
            if idx < MAX_QUANTUS_IMAGES:
                print(f"[{ts()}] Quantus...", flush=True)
                results = {
                    "IntegratedGradients": evaluate_quantus(model, img, label, ig_attr),
                    "Saliency": evaluate_quantus(model, img, label, sal_attr),
                    "GradientShap": evaluate_quantus(model, img, label, gs_attr),
                    "Occlusion": evaluate_quantus(model, img, label, occ_attr),
                    "SHAP": evaluate_quantus(model, img, label, shap_attr),
                }
            else:
                results = {}

            model_results.append(results)

        # ----------------------------
        # PER-MODEL SUMMARY
        # ----------------------------
        print(f"[{ts()}] Summarizing model attributions...", flush=True)
        summarize_model_attributions(name, model_attr_collection)

        all_metrics.append({
            "dataset": ds_name,
            "model": name,
            "results": model_results
        })

        send_ntfy_notification(f"XAI: Completed {name} on {ds_name}")
        print(f"[{ts()}] === Finished {name} on {ds_name} ===", flush=True)

        del model
        torch.cuda.empty_cache()
        gc.collect()

# ============================
# SAVE QUANTUS RESULTS
# ============================

save_quantus_csv(all_metrics)
send_ntfy_notification("XAI: Quantus CSV saved")
print(f"[{ts()}] Quantus CSV saved.", flush=True)


# In[ ]:


"""ALL_DATASETS = ["Flickr8k", "Flickr30k", "ConceptualCaptions"]
all_metrics = []

for ds_name in ALL_DATASETS:

    df_path = os.path.join(DATASETS_DIR, f"df_{ds_name}.pkl")
    if not os.path.exists(df_path):
        print(f"Skipping {ds_name}: Metadata not found.")
        continue

    df = pd.read_pickle(df_path)
    IMAGE_PATHS = df["image_path"].tolist()

    for name, (model, transform) in models.items():

        print(f"\n=== Running {name} on {ds_name} ===")
        send_ntfy_notification(f"Starting SHAP background for {name} on {ds_name}", "XAI Pipeline")

        dataset = ImageDataset(IMAGE_PATHS, transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1)

        # ----------------------------
        # SHAP BACKGROUND (always 32)
        # ----------------------------
        # SHAP background
        bckgrd = []
        for i, img in enumerate(loader):
            if i >= 8:  # instead of 32 for MODE C
                break
            bckgrd.append(img.squeeze(0).to(device))
        bckgrd = torch.stack(bckgrd)


        model_results = []
        model_attr_collection = []

        torch.cuda.empty_cache()
        gc.collect()

        # ----------------------------
        # MAIN XAI LOOP
        # ----------------------------
        for idx, img in enumerate(tqdm(loader)):

            img = img.squeeze(0).to(device)

            logits = model(img.unsqueeze(0))
            label = logits.argmax(dim=1).item()

            # Core XAI methods
            ig_attr = explain_ig(model, img, label)
            sal_attr = explain_saliency(model, img, label)
            gs_attr = explain_gradientShap(model, img, label)
            occ_attr = explain_occlusion(model, img, label)

            # SHAP (limited)
            if MAX_SHAP_IMAGES is None or idx < MAX_SHAP_IMAGES:
                shap_attr = explain_shap(model, img, bckgrd, name)
            else:
                shap_attr = torch.zeros_like(ig_attr)

            # Visualization (limited)
            if MAX_VISUAL_IMAGES is None or idx < MAX_VISUAL_IMAGES:
                visualize_side_by_side(
                    img,
                    ig_attr,
                    sal_attr,
                    gs_attr,
                    occ_attr,
                    shap_attr,
                    title=f"{name} — Image {idx}"
                )

            # Store attributions
            model_attr_collection.append({
                "IG": ig_attr,
                "Saliency": sal_attr,
                "GradientShap": gs_attr,
                "Occlusion": occ_attr,
                "SHAP": shap_attr
            })

            # Quantus (limited)
            if MAX_QUANTUS_IMAGES is None or idx < MAX_QUANTUS_IMAGES:
                results = {
                    "IntegratedGradients": evaluate_quantus(model, img, label, ig_attr),
                    "Saliency": evaluate_quantus(model, img, label, sal_attr),
                    "GradientShap": evaluate_quantus(model, img, label, gs_attr),
                    "Occlusion": evaluate_quantus(model, img, label, occ_attr),
                    "SHAP": evaluate_quantus(model, img, label, shap_attr),
                }
            else:
                results = {}

            model_results.append(results)

        # ----------------------------
        # PER-MODEL SUMMARY
        # ----------------------------
        summarize_model_attributions(name, model_attr_collection)

        all_metrics.append({
            "dataset": ds_name,
            "model": name,
            "results": model_results
        })

        send_ntfy_notification(f"Completed explanation for {ds_name} for {name}", "XAI Pipeline")

        del model
        torch.cuda.empty_cache()
        gc.collect()


# ============================
# SAVE QUANTUS RESULTS
# ============================

save_quantus_csv(all_metrics)
send_ntfy_notification("Quantus CSV saved", "XAI Pipeline")"""

