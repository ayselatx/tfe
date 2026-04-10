#!/usr/bin/env python
# coding: utf-8

# # Imports

# ## Standard Library

# In[ ]:


import os          # file paths, directory management
import sys         # system-level operations (rarely used, but useful for debugging)
import time        # timing execution (performance measurement)
import gc          # garbage collection (manual memory cleanup)
from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
from thop import profile  # for FLOPs and params
import pynvml


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


# In[ ]:


import urllib.request
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm


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
RAW_DIR = os.path.join(BASE_DIR, 'Features_RAW', CURRENT_DATASET)
RESULTS_DIR = os.path.join(BASE_DIR, 'Unimodal_Results', CURRENT_DATASET)

def get_raw_dir(dataset, modality):
    path = os.path.join(BASE_DIR, "Features", "raw", dataset, modality)
    os.makedirs(path, exist_ok=True)
    return path

def get_results_dir(dataset):
    path = os.path.join(BASE_DIR, "Unimodal", "Unimodal_Results", dataset)
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



# # Utility Functions

# ## GreenAI Metrics

# In[ ]:


def measure_memory():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

def measure_gpu_memory():
    """Return GPU memory used in MB (NVIDIA only)."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # assume single GPU
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used / 1024**2

def measure_gpu_utilization():
    """Return GPU utilization percentage (NVIDIA only)."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu


# In[ ]:


def estimate_energy(flops, gflops_per_s=35000, power_w=300):
    """
    Estimate energy (J) from FLOPs.
    - flops: total FLOPs for the model
    - gflops_per_s: GPU performance in GFLOPs/s
    - power_w: GPU TDP in Watts
    """
    if flops is None:
        return None
    exec_time_s = flops / (gflops_per_s * 1e9)  # seconds
    energy_j = exec_time_s * power_w
    return energy_j


# In[ ]:


def get_size_in_mb(obj):
    """Returns the size of a numpy array in MB."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes / (1024 * 1024)
    else:
        return sys.getsizeof(obj) / (1024 * 1024)


# In[ ]:


def save_metadata(data, dataset, modality, model_name):
    model_dir = os.path.join(BASE_DIR, "Results", dataset, modality, model_name)
    os.makedirs(model_dir, exist_ok=True)
    meta_path = os.path.join(model_dir, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({"samples": data, "indices": list(range(len(data)))}, f)


# In[ ]:


def execute_and_save(dataset, modality, model_name, extract_func, data, device, model=None):
    """
    Runs feature extraction, saves embeddings and metadata, 
    and returns exhaustive GreenAI metrics.
        """
    start_time = time.time()
    mem_before = measure_memory()
    gpu_mem_before = measure_gpu_memory()
    pynvml.nvmlInit()

    try:
        features = extract_func(data, device, dataset)
    except Exception as e:
        print(f"[ERROR] {model_name} extraction failed: {e}")
        return None

    exec_time = time.time() - start_time
    mem_used = max(0, measure_memory() - mem_before)
    gpu_mem_used = None
    gpu_mem_used = max(0, measure_gpu_memory() - gpu_mem_before)
    latency = exec_time / len(data)
    throughput = len(data) / exec_time

    # --- FLOPs and parameters ---
    flops, params = (None, None)
    if model is not None:
        try:
            # Convert data to tensor for FLOPs measurement
            if isinstance(data, list):
                data_tensor = torch.stack([torch.tensor(d) for d in data]).to(device)
            else:
                data_tensor = data.to(device)
            flops, params = profile(model, inputs=(data_tensor,), verbose=False)
        except Exception as e:
            print(f"[WARN] FLOPs calculation failed: {e}")
    energy_j = estimate_energy(flops)

    # --- Save embeddings ---
    model_dir = os.path.join(BASE_DIR, "Results_Unimodal", dataset, modality, model_name)
    os.makedirs(model_dir, exist_ok=True)
    emb_path = os.path.join(model_dir, "embeddings.npy")
    np.save(emb_path, features)

    # --- Save metadata ---
    save_metadata(data, dataset, modality, model_name)

    # --- GPU/CPU utilization snapshot ---
    gpu_util = measure_gpu_utilization()
    cpu_util = psutil.cpu_percent(interval=None)

    print(f"[SAVED] {model_name} | shape={features.shape} | dataset={dataset}")

    return {
        "Dataset": dataset,
        "Modality": modality,
        "Model": model_name,
        "Dim": features.shape[1],
        "Time_s": exec_time,
        "Latency_s": latency,
        "Throughput_samples_per_s": throughput,
        "Memory_MB": mem_used,
        "GPU_Memory_MB": gpu_mem_used,
        "GPU_Util_percent": gpu_util,
        "CPU_Util_percent": cpu_util,
        "FLOPs": flops,
        "Params": params,
        "Energy_J": energy_j, 
        "Disk_IO_MB": (features.nbytes / 1024**2)  # embeddings size as rough I/O
    }


# ## XAI Saliency Maps

# In[ ]:


def get_saliency_dir(dataset_name, model_name):
    path = os.path.join(BASE_DIR, "Results", dataset_name, model_name, "Saliency_Unimodal")
    os.makedirs(path, exist_ok=True)
    return path


# ### Vision XAI

# In[ ]:


def save_vision_saliency_and_overlay(model_name, activations, img_path, dataset_name):
    """
    Saves saliency maps and overlay images for vision models.
    Supports norm-based weighting or Grad-CAM if gradients exist.
    """
    act = activations.detach().cpu()

    # Compute saliency
    if act.ndim == 4:  # CNN: [B(batch),C(channels),H(height),W(width)]
        weights = torch.norm(act, dim=1, keepdim=True) # L2 norm across channels
        heatmap = torch.sum(act * weights, dim=1).squeeze().numpy() # weighted sum across channels
        # Result: 2D heatmap (H x W) showing salient regions for the image

    elif act.ndim == 3:  # Transformer: [B(batch),T(tokens),D(dimensions)]
        tokens = act[:, 1:, :] # skip CLS token

        if tokens.shape[1] in [197, 198]: # some ViT variants have 197 tokens (196 patches + CLS) or 198 (with extra tokens)
            tokens = tokens[:, :196, :]

        weights = torch.norm(tokens, dim=-1, keepdim=True) # L2 norm across dimensions
        weighted_sum = torch.sum(tokens * weights, dim=-1) # weighted sum across dimensions
        side = int(np.sqrt(tokens.shape[1])) # assuming square arrangement of tokens (e.g., 14x14 for 196 tokens)
        heatmap = weighted_sum.view(side, side).numpy() 

        # Result: 2D heatmap (side x side) showing salient regions for the image

    else:
        return

    # Normalize & resize
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    heatmap_res = cv2.resize(heatmap, (224, 224))

    # Save raw saliency
    saliency_dir = get_saliency_dir(dataset_name, model_name)
    os.makedirs(saliency_dir, exist_ok=True)
    image_filename = os.path.splitext(os.path.basename(img_path))[0]
    np.save(os.path.join(saliency_dir, f"Saliency_{image_filename}.npy"), heatmap_res)

    # Save overlay
    overlay_dir = os.path.join(BASE_DIR, "Results", dataset_name, model_name, "Overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"[WARNING] Could not read image for overlay (wrong path?): {img_path}")
        return  # skip this image safely

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[WARNING] Could not read image for overlay (cv2 failed): {img_path}")
        return

    img_bgr = cv2.resize(img_bgr, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET) # convert heatmap to color
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0) # blend original image with heatmap
    cv2.imwrite(os.path.join(overlay_dir, f"Overlay_{image_filename}.jpg"), overlay)


# ### Text XAI

# In[ ]:


# Convert each caption to separate sample and keep mapping to image
def prepare_text_samples(df):
    texts = []
    caption_image_map = []
    for i, caption in enumerate(df['captions']):
        if isinstance(caption, list):
            for cap in caption:
                texts.append(cap)
                caption_image_map.append(df['image_path'][i])
        else:
            texts.append(caption)
            caption_image_map.append(df['image_path'][i])
    return texts, caption_image_map


# In[ ]:


def save_text_saliency(tokens, scores, dataset, model_name, idx, img_path=None):
    saliency_dir = get_saliency_dir(dataset, model_name)
    path = os.path.join(saliency_dir, f"SaliencyText_{idx}.npy")
    np.save(path, {
        "tokens": tokens,
        "scores": scores,
        "image_path": img_path
    })


# # Unimodal Models

# ## CBIR: Vision Feature Extractions

# ### Image Dataset

# In[ ]:


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        # Filter out missing images at init
        self.image_paths = [p for p in image_paths if os.path.exists(p)]
        self.transform = transform
        if len(self.image_paths) == 0:
            raise ValueError("No valid image paths found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Could not read image: {img_path} | {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))  # fallback

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"[WARNING] Transform failed for image: {img_path} | {e}")
                image = torch.zeros((3, 224, 224))  # fallback tensor

        return image


# ### Feature Extraction

# In[ ]:


def vision_extract_features(model, model_name, dataloader, device, target_layer,
                            paths, dataset_name, save_saliency=True, gradcam=False):
    """
    Extract vision features and save saliency maps (CNN or Transformer).
    """
    model.eval()
    features_list = []
    activations = {}
    img_idx = 0

    # Forward hook to capture intermediate activations
    def hook_fn(module, input, output):
        if hasattr(output, "last_hidden_state"):
            activations['value'] = output.last_hidden_state
        elif isinstance(output, tuple):
            activations['value'] = output[0]
        else:
            activations['value'] = output

    hook = target_layer.register_forward_hook(hook_fn) if target_layer else None

    for batch in tqdm(dataloader, desc=f"{model_name}"):
        batch = batch.to(device)
        batch_acts = None

        ## FORWARD PASS
        output = model(batch)
        batch_acts = activations.get('value')

        # --- Feature vector extraction ---
        if hasattr(output, 'last_hidden_state'):
            feat = output.last_hidden_state[:, 0, :] #Transformers: use [CLS] token embedding (output.last_hidden_state[:, 0, :]).
        elif isinstance(output, torch.Tensor):
            if output.ndim == 4:  #CNNs: apply adaptive average pooling on H x W feature maps → flatten to vector.
                feat = torch.nn.functional.adaptive_avg_pool2d(output, 1).view(output.size(0), -1)
            else: #Generic tensor outputs: used directly if already 1D or 2D.
                feat = output
        else:
            feat = output[0]

        features_list.append(feat.detach().cpu().numpy()) #Converts to NumPy for storage/analysis.

        # --- Compute gradients if Grad-CAM ---
        if gradcam and batch_acts is not None:
            batch_acts.requires_grad_(True)
            target_score = feat.sum()
            target_score.backward(retain_graph=True)

        # --- Save saliency maps ---
        if save_saliency and batch_acts is not None:
            for i in range(batch_acts.size(0)):
                img_path = paths[img_idx]

                if not os.path.exists(img_path):
                    print(f"[WARNING] Skipping missing image: {img_path}")
                    img_idx += 1
                    continue

                save_vision_saliency_and_overlay(
                    model_name.lower(),
                    batch_acts[i:i+1],
                    img_path,
                    dataset_name
                )
                img_idx += 1
            activations['value'] = None

        # Reset gradients for next iteration
        model.zero_grad()
        if batch_acts is not None and batch_acts.grad is not None:
            batch_acts.grad.zero_() #Ensures no gradient accumulation between batches.

    if hook:
        hook.remove()

    return np.vstack(features_list)


# #### CNN Embeddings

# In[ ]:


def get_resnet50_embeddings(image_paths, device, dataset_name):
    try:
        model = models.resnet50(weights="DEFAULT").to(device)
        target_layer = model.layer4[-1] #Defines target layer for saliency hooks: ResNet50 → layer4[-1]
        model.fc = nn.Identity()  # Removes classifier head (nn.Identity()) → only feature embeddings remain.

        transform = models.ResNet50_Weights.DEFAULT.transforms()
        dataset = ImageDataset(image_paths, transform=transform) #Wraps images in ImageDataset and DataLoader
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

        return vision_extract_features(model, "resnet50", loader, device,
                                              target_layer, image_paths, dataset_name, save_saliency=True, gradcam=True)
    except Exception as e:
        print(f"[ERROR] ResNet50 embedding failed: {e}")
        return None


# In[ ]:


def get_mobilenet_v3_embeddings(image_paths, device, dataset_name):
    try:
        model = models.mobilenet_v3_large(weights="DEFAULT").to(device)
        target_layer = model.features[-1] #Defines target layer for saliency hooks: MobileNetV3 → last convolutional layer (features[-1])
        model.classifier = nn.Identity()  # Removes classifier head (nn.Identity()) → only feature embeddings remain.

        transform = models.MobileNet_V3_Large_Weights.DEFAULT.transforms()
        dataset = ImageDataset(image_paths, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4)

        return vision_extract_features(model, "mobilenet_v3", loader, device,
                                              target_layer, image_paths, dataset_name, save_saliency=True, gradcam=True)
    except Exception as e:
        print(f"[ERROR] MobileNetV3 embedding failed: {e}")
        return None


# #### Transformer Embeddings

# In[ ]:


def get_vit_embeddings(image_paths, device, dataset_name):
    try:
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device)

        target_layer = model.encoder.layer[-1] #Defines target layer for activations (e.g., last encoder block).

        def collate_fn(batch):
            processed_imgs = []
            for img in batch:
                # Ensure image is a PIL Image
                try:
                    if not isinstance(img, Image.Image):
                        raise ValueError("Invalid image type")
                    img = img.convert('RGB')  # force RGB
                    pixel_values = processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                except Exception as e:
                    # Fallback black RGB image if something fails
                    print(f"[WARNING] Replacing corrupted image with fallback | {e}")
                    fallback = Image.new('RGB', (224, 224), (0, 0, 0))
                    pixel_values = processor(images=fallback, return_tensors="pt")['pixel_values'].squeeze(0)
                processed_imgs.append(pixel_values)
            return torch.stack(processed_imgs)

        dataset = ImageDataset(image_paths)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, num_workers=4, collate_fn=collate_fn
        )

        return vision_extract_features(
            model, "vit", loader, device,
            target_layer, image_paths, dataset_name,
            save_saliency=True, gradcam=True
        )

    except Exception as e:
        print(f"[ERROR] ViT embedding failed: {e}")
        return None


# In[ ]:


def get_deit_embeddings(image_paths, device, dataset_name):
    try:
        processor = DeiTImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224').to(device)
        target_layer = model.encoder.layer[-1].output

        def collate_fn(batch):
            processed_imgs = []
            for img in batch:
                # Ensure image is a PIL Image
                try:
                    if not isinstance(img, Image.Image):
                        raise ValueError("Invalid image type")
                    img = img.convert('RGB')  # force RGB
                    pixel_values = processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                except Exception as e:
                    # Fallback black RGB image if something fails
                    print(f"[WARNING] Replacing corrupted image with fallback | {e}")
                    fallback = Image.new('RGB', (224, 224), (0, 0, 0))
                    pixel_values = processor(images=fallback, return_tensors="pt")['pixel_values'].squeeze(0)
                processed_imgs.append(pixel_values)
            return torch.stack(processed_imgs)

        dataset = ImageDataset(image_paths)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

        return vision_extract_features(model, "deit", loader, device,
                                              target_layer, image_paths, dataset_name, save_saliency=True, gradcam=True)
    except Exception as e:
        print(f"[ERROR] DeiT embedding failed: {e}")
        return None


# In[ ]:


def get_clip_vision_embeddings(image_paths, device, dataset_name):
    try:
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
        target_layer = model.vision_model.encoder.layers[-1]

        def collate_fn(batch):
            return torch.stack([processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0) for img in batch])

        dataset = ImageDataset(image_paths)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

        return vision_extract_features(model, "clip_vision", loader, device,
                                              target_layer, image_paths, dataset_name, save_saliency=True, gradcam=True)
    except Exception as e:
        print(f"[ERROR] CLIP Vision embedding failed: {e}")
        return None


# ## T2T: Text Feature Extractions
# 

# ### Text Dataset

# In[ ]:


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}


# ### Multiple Captions Encoder

# In[ ]:


def encode_multiple_captions(captions, model, tokenizer, device):
    """
    Encodes multiple captions for the same image and averages the embeddings.
    """
    embeddings = []
    for cap in captions:
        inputs = tokenizer(
            cap, return_tensors="pt", truncation=True, padding=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)  # mean pooling
            embeddings.append(emb.detach().cpu().numpy())
    return np.mean(embeddings, axis=0)


# ### Text Feature Extraction

# In[ ]:


def text_extract_features(model, tokenizer, dataloader, device, dataset_name, model_name, 
                          feature_type='mean_pool', save_xai=True, caption_image_map=None):
    """
    Extract embeddings for all captions and optionally save token-level saliency maps.
    Supports linking saliency to both caption and original image.
    """
    model.eval()
    features_list = []
    xai_index = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"{model_name} Text Extraction")):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        # Feature extraction
        if feature_type == 'mean_pool':
            mask = batch['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            batch_features = (sum_emb / sum_mask).detach().cpu().numpy()
        else:
            batch_features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        features_list.append(batch_features)

        # Token-level XAI
        if save_xai:
            for i in range(outputs.last_hidden_state.size(0)):
                tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                scores = outputs.last_hidden_state[i].norm(dim=-1).detach().cpu().numpy()
                img_path = caption_image_map[xai_index] if caption_image_map is not None else None
                save_text_saliency(tokens, scores, dataset_name, model_name, xai_index, img_path=img_path)
                xai_index += 1

    return np.vstack(features_list)


# ### Specialized RNN for text

# In[ ]:


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.rnn(embedded)
        return hidden[-1], output  # last hidden for embedding, full seq for XAI

# ----------------------------
# RNN Embedding + XAI
# ----------------------------
def get_rnn_embeddings(texts, device, dataset_name, caption_image_map=None):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SimpleRNN(tokenizer.vocab_size).to(device)
    dataset = TextDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    features_list = []
    saliency_dir = os.path.join(BASE_DIR, f"TextSaliency_{dataset_name}")
    os.makedirs(saliency_dir, exist_ok=True)

    model.eval()
    for idx, batch in enumerate(tqdm(dataloader, desc="RNN XAI")):
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            last_hidden, full_seq = model(input_ids)
            features_list.append(last_hidden.detach().cpu().numpy())

        # Token-level saliency: magnitude of LSTM output
        for b in range(full_seq.size(0)):
            tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
            scores = full_seq[b].norm(dim=-1).detach().cpu().numpy()
            saliency_path = os.path.join(saliency_dir, f"SaliencyText_rnn_{idx*32 + b}.npy")
            np.save(saliency_path, {'tokens': tokens, 'scores': scores})

    return np.vstack(features_list)


# ### Text Models

# In[ ]:


def get_bert_embeddings(texts, device, dataset_name, caption_image_map=None):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    dataset = TextDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    return text_extract_features(model, tokenizer, dataloader, device, dataset_name, "bert")


# In[ ]:


def get_roberta_embeddings(texts, device, dataset_name, caption_image_map=None):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base').to(device)
    dataset = TextDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    return text_extract_features(model, tokenizer, dataloader, device, dataset_name, "roberta")


# In[ ]:


def get_gpt2_embeddings(texts, device, dataset_name, caption_image_map=None):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained('gpt2').to(device)
    dataset = TextDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    return text_extract_features(model, tokenizer, dataloader, device, dataset_name, "gpt2")


# In[ ]:


def get_clip_text_embeddings(texts, device, dataset_name=None, caption_image_map=None, max_length=32):
    from transformers import CLIPTextModel, CLIPTokenizer
    import torch
    import numpy as np
    from tqdm import tqdm

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    features = []
    model.eval()
    batch_size = 32
    global_index = 0  # Tracks position across all batches

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Extracting CLIP Text"):
        batch_texts = texts[start_idx:start_idx+batch_size]

        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length)
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoding)
            batch_features = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # CLS token
            features.append(batch_features)

            # --- Token-level saliency ---
            for b in range(outputs.last_hidden_state.size(0)):
                if caption_image_map is not None:
                    img_path = caption_image_map[global_index]
                else:
                    img_path = None

                tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][b])
                scores = outputs.last_hidden_state[b].norm(dim=-1).detach().cpu().numpy()
                save_text_saliency(tokens, scores, dataset_name, "clip_text", global_index, img_path=img_path)
                global_index += 1

    return np.vstack(features)


# # Execution

# In[ ]:


"""ALL_DATASETS = ["ConceptualCaptions", "Flickr8k", "Flickr30k"]
all_metrics = []

for ds_name in ALL_DATASETS:
    df_path = os.path.join(DATASETS_DIR, f"df_{ds_name}.pkl")
    if not os.path.exists(df_path):
        print(f"Skipping {ds_name}: Metadata not found.")
        continue

    df = pd.read_pickle(df_path)
    IMAGE_PATHS = df['image_path'].tolist()
    CAPTIONS_LIST = df['captions'].tolist()

    CAPTIONS_LIST_SIMPLIFIED = [caps[0] for caps in CAPTIONS_LIST] # For XAI, we will use the first caption of each image to keep it simple.
    print(f"Ready. Loaded {len(IMAGE_PATHS)} images and {len(CAPTIONS_LIST_SIMPLIFIED)} captions into memory.")
    texts_for_xai, caption_image_map = prepare_text_samples(df)

    send_ntfy_notification(f"Starting Indexation for {ds_name}", "TFE Pipeline")

    # --- Vision Pipeline ---
    vision_pipeline = {
        "resnet50": get_resnet50_embeddings,
        "mobilenet_v3": get_mobilenet_v3_embeddings,
        "vit": get_vit_embeddings,
        "deit": get_deit_embeddings,
        "clip_vision": get_clip_vision_embeddings
    }

    for name, func in vision_pipeline.items():
        metrics = execute_and_save(ds_name, "vision", name, func, IMAGE_PATHS, device)
        if metrics: 
            all_metrics.append(metrics)
        send_ntfy_notification(f"Completed Indexation for {ds_name} for {name}", "TFE Pipeline")


    # --- Text Pipeline ---
    text_pipeline = {
        "bert": get_bert_embeddings,
        "roberta": get_roberta_embeddings,
        "gpt2": get_gpt2_embeddings,
        "clip_text": get_clip_text_embeddings
    }

    for name, func in text_pipeline.items():
        # Flatten captions if model expects single caption per sample
        flattened_captions = [cap for caps in CAPTIONS_LIST for cap in caps]

        metrics = execute_and_save(ds_name, "text", name, func, flattened_captions, device)
        if metrics:
            all_metrics.append(metrics)
        send_ntfy_notification(f"Completed Indexation for {ds_name} for {name}", "TFE Pipeline")


    send_ntfy_notification(f"Completed {ds_name}", "TFE Pipeline Success")

# --- Save Global Metrics ---
os.makedirs(os.path.join(BASE_DIR, "Results"), exist_ok=True)
df_final = pd.DataFrame(all_metrics)
df_final.to_pickle(os.path.join(BASE_DIR, "Results", "global_unimodal_metrics.pkl"))

print("All datasets and models processed successfully.")
send_ntfy_notification(f"Completed Indexation for all datasets", "TFE Pipeline Success")
"""


# In[ ]:


# ============================
# TEST SUBSET INDEXATION BLOCK
# ============================
vision_pipeline = {
        "resnet50": get_resnet50_embeddings,
        "mobilenet_v3": get_mobilenet_v3_embeddings,
        "vit": get_vit_embeddings,
        "deit": get_deit_embeddings,
        "clip_vision": get_clip_vision_embeddings
    }

text_pipeline = {
        "bert": get_bert_embeddings,
        "roberta": get_roberta_embeddings,
        "gpt2": get_gpt2_embeddings,
        "clip_text": get_clip_text_embeddings
    }

TEST_SAMPLE_SIZE = 5
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "TestEmbeddings")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

print(f"\n[{ts()}] Starting TEST subset indexation...")

for ds_name in ALL_DATASETS:

    df_path = os.path.join(DATASETS_DIR, f"df_{ds_name}.pkl")
    if not os.path.exists(df_path):
        print(f"[{ts()}] Skipping {ds_name}: Metadata not found.")
        continue

    df = pd.read_pickle(df_path)
    IMAGE_PATHS = df['image_path'].tolist()
    CAPTIONS_LIST = df['captions'].tolist()

    test_image_paths = IMAGE_PATHS[:TEST_SAMPLE_SIZE]
    test_captions = [caps[0] for caps in CAPTIONS_LIST[:TEST_SAMPLE_SIZE]]

    print(f"[{ts()}] Test subset for {ds_name}: {len(test_image_paths)} images")

    # --- Vision Models ---
    for model_name, func in vision_pipeline.items():
        print(f"[{ts()}] Computing TEST embeddings for {model_name} on {ds_name}...")

        # CORRECT CALL SIGNATURE
        test_embs = func(ds_name, "vision", model_name, test_image_paths, device)

        out_path = os.path.join(TEST_OUTPUT_DIR, ds_name, "vision")
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, f"{model_name}_full.npy"), test_embs)

    # --- Text Models ---
    for model_name, func in text_pipeline.items():
        print(f"[{ts()}] Computing TEST text embeddings for {model_name} on {ds_name}...")

        test_text_embs = func(ds_name, "text", model_name, test_captions, device)

        out_path = os.path.join(TEST_OUTPUT_DIR, ds_name, "text")
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, f"{model_name}_full.npy"), test_text_embs)

print(f"[{ts()}] TEST subset indexation completed.")
send_ntfy_notification("Test subset indexation completed", "TFE Pipeline")


# In[ ]:




