import numpy as np
import streamlit as st

import sys
sys.path.insert(0, "/home/aysel/tfe")

from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import open_clip, clip
from streamlit_app.utils.loaders import DATASETS_DIR
import pandas as pd
import torch
from torchvision import transforms
from transformers import (
    ViTModel, ViTImageProcessor,
    AutoModel, AutoImageProcessor,
    CLIPVisionModel, CLIPProcessor
)

from streamlit_app.utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache

# Load CLIP
@st.cache_resource
def get_clip():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    try:
        model = model.cuda().eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, preprocess

clip_model, clip_preprocess = get_clip()


def clip_encode_image(img):
    with torch.no_grad():
        return clip_model.encode_image(
            clip_preprocess(img).unsqueeze(0).cuda()
        ).cpu().numpy()

def clip_encode_text(text):
    tok = clip.tokenize([text]).cuda()
    with torch.no_grad():
        return clip_model.encode_text(tok).cpu().numpy()

# Load OpenCLIP
@st.cache_resource
def get_openclip():
    import open_clip
    model, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    try:
        model = model.cuda().eval()
    except RuntimeError:
        model = model.cpu().eval()
    return model, preprocess

oc_model, oc_preprocess = get_openclip()
oc_model = oc_model.cuda().eval()

def openclip_encode_image(img):
    with torch.no_grad():
        return oc_model.encode_image(
            oc_preprocess(img).unsqueeze(0).cuda()
        ).cpu().numpy()

def openclip_encode_text(text):
    tok = open_clip.tokenize([text]).cuda()
    with torch.no_grad():
        return oc_model.encode_text(tok).cpu().numpy()

# Unimodal query embedding
from streamlit_app.utils.loaders import VISION_CACHE, TEXT_CACHE

# ---------------------------------------------------------
# IMAGE EMBEDDING (mirrors unimodal notebooks)
# ---------------------------------------------------------

def embed_image_query(path, model_name, device=device):
    img = Image.open(path).convert("RGB")

    # -----------------------------
    # ResNet‑50
    # -----------------------------
    if model_name == "ResNet50" or model_name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        model.eval()

        # Remove classifier → keep 2048‑dim features
        model = torch.nn.Sequential(*list(model.children())[:-1])

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(x).squeeze()  # shape (2048,)
        return feat.cpu().numpy().reshape(1, -1)


    # -----------------------------
    # MobileNetV3‑Large
    # -----------------------------
    if model_name == "MobileNetV3" or model_name =="mobilenet_v3":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights).to(device)
        model.eval()

        # Remove classifier → keep 960-d features
        feature_extractor = torch.nn.Sequential(
            model.features,
            model.avgpool
        ).to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = feature_extractor(x).squeeze()  # shape (960,)

        return feat.cpu().numpy().reshape(1, -1)


    # -----------------------------
    # ViT‑Base
    # -----------------------------
    if model_name == "ViT" or model_name == "vit":
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        model.eval()

        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.pooler_output.cpu().numpy()

    # -----------------------------
    # PVT‑Tiny
    # -----------------------------
    if model_name == "PVT" or model_name == "pvt":
        processor = AutoImageProcessor.from_pretrained("Zetatech/pvt-tiny-224")
        model = AutoModel.from_pretrained("Zetatech/pvt-tiny-224").to(device)
        model.eval()

        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # PVT has no pooler_output → use mean pooling
            feat = outputs.last_hidden_state.mean(dim=1)  # shape (1, 512)

        return feat.cpu().numpy()


    raise ValueError(f"Unsupported vision model: {model_name}")


from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer

# ---------------------------------------------------------
# TEXT EMBEDDING (mirrors unimodal notebooks)
# ---------------------------------------------------------
def embed_text_query(text, model_name, device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- BERT ---
    if model_name == "BERT" or model_name == "bert":
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased").to(device)
        model.eval()

        enc = tok(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1)
        return pooled.cpu().numpy()

    # --- RoBERTa ---
    if model_name == "RoBERTa" or model_name == "roberta":
        tok = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModel.from_pretrained("roberta-base").to(device)
        model.eval()

        enc = tok(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1)
        return pooled.cpu().numpy()

    # --- GPT‑2 ---
    if model_name in ["GPT2", "gpt2", "gpt-2"]:
        tok = AutoTokenizer.from_pretrained("gpt2")

        # GPT‑2 has no pad token → assign eos_token as pad_token
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModel.from_pretrained("gpt2").to(device)
        model.eval()

        enc = tok(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)

        with torch.no_grad():
            out = model(**enc).last_hidden_state
            pooled = out.mean(dim=1)

        return pooled.cpu().numpy()


    raise ValueError(f"Unsupported text model: {model_name}")



def fusion_retrieve(query_emb, fusion_row, index_type="image"):
    Wv = np.load(fusion_row["Wv_path"])
    Wt = np.load(fusion_row["Wt_path"])

    if index_type == "image":
        F = np.load(fusion_row["image_index_path"])
    else:
        F = np.load(fusion_row["caption_index_path"])

    # Project query
    if index_type == "image":
        Xv = query_emb @ Wv
        Xt = np.zeros_like(Xv)
    else:
        Xt = query_emb @ Wt
        Xv = np.zeros_like(Xt)

    # Fuse
    if fusion_row["fusion"] == "concat":
        Xq = np.concatenate([Xv, Xt], axis=1)
    elif fusion_row["fusion"] == "add":
        Xq = Xv + Xt
    else:
        Xq = 0.5 * Xv + 0.5 * Xt

    sims = cosine_similarity(Xq, F).flatten()
    idx = np.argsort(sims)[::-1][:50]
    return idx, sims[idx]
