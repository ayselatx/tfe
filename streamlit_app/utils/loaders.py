import os
import numpy as np
import pandas as pd
import json


import sys
sys.path.insert(0, "/home/aysel/tfe")

from streamlit_app.utils.paths import PROJECTIONS_DIR, vision_emb_path, text_emb_path
# -----------------------------
# Correct base directory for the Streamlit app
# -----------------------------
BASE_DIR = "data"
DATASETS_DIR = os.path.join(BASE_DIR, "dataset")
UNIMODAL_DIR = os.path.join(BASE_DIR, "unimodal")
MULTIMODAL_DIR = os.path.join(BASE_DIR, "multimodal")

# Global caches for embeddings (used by retrieval_multimodal)
VISION_CACHE = {}
TEXT_CACHE = {}


def find_model_column(df):
    candidates = ["Model", "model", "model_name", "architecture", "backbone"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No model column found in df. Columns = {df.columns.tolist()}")

def normalize(df):
    col = find_model_column(df)
    df[col] = (
        df[col]
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )
    df = df.rename(columns={col: "Model"})
    return df

# -----------------------------
# Model lists
# -----------------------------
def load_text_models():
    return {
        "BERT": "BERT Base Uncased",
        "RoBERTa": "RoBERTa Base",
        "GPT2": "GPT‑2",
    }

def load_vision_models():
    return {
        "resnet50": "ResNet‑50",
        "mobilenet_v3": "MobileNetV3‑Large",
        "vit": "ViT‑Base",
        "pvt": "PVT‑Tiny",
    }

# -----------------------------
# Unimodal metrics
# -----------------------------
def load_text_metrics():
    perf = pd.read_csv(f"{UNIMODAL_DIR}/metrics/text_retrieval_results.csv")
    exp = pd.read_csv(f"{UNIMODAL_DIR}/metrics/Explainability_Text.csv")
    eff = pd.read_pickle(f"{UNIMODAL_DIR}/metrics/global_unimodal_metrics.pkl")
    eff = eff[eff["Modality"] == "text"]

    df = pd.read_pickle(f"{DATASETS_DIR}/df_Flickr8k.pkl")
    stress_caps = [cap for caps in df["captions"].tolist() for cap in caps][:50]

    return {
        "performance": perf,
        "explainability": exp,
        "efficiency": eff,
        "stress_captions": stress_caps
    }

def load_vision_metrics():
    perf = pd.read_csv(f"{UNIMODAL_DIR}/metrics/vision_retrieval_results.csv")
    exp = pd.read_csv(f"{UNIMODAL_DIR}/metrics/Explainability_Vision.csv")
    eff = pd.read_pickle(f"{UNIMODAL_DIR}/metrics/global_unimodal_metrics.pkl")
    eff = eff[eff["Modality"] == "vision"]

    df = pd.read_pickle(f"{DATASETS_DIR}/df_Flickr8k.pkl")
    stress_imgs = df["image_path"].tolist()[:50]

    return {
        "performance": perf,
        "explainability": exp,
        "efficiency": eff,
        "stress_images": stress_imgs
    }

# -----------------------------
# Global unimodal results
# -----------------------------
def load_global_unimodal_results():

    # Load all metric sources
    perf_v = pd.read_csv(f"{UNIMODAL_DIR}/metrics/vision_retrieval_results.csv")
    exp_v  = pd.read_csv(f"{UNIMODAL_DIR}/metrics/Explainability_Vision.csv")
    eff    = pd.read_pickle(f"{UNIMODAL_DIR}/metrics/global_unimodal_metrics.pkl")

    # Filter efficiency for each modality
    eff_v = eff[eff["Modality"] == "vision"]
    eff_t = eff[eff["Modality"] == "text"]

    # Normalize BEFORE merging
    perf_v = normalize(perf_v)
    exp_v  = normalize(exp_v)
    eff_v  = normalize(eff_v)

    # Rename BEFORE merging
    eff_v = eff_v.rename(columns={
        "Time_s": "inference_time",
        "Latency_s": "embedding_time",
        "Memory_MB": "memory_mb"
    })

    # Merge all vision metrics
    df_vision = (
        perf_v
        .merge(exp_v, on="Model", how="inner")
        .merge(eff_v, on="Model", how="inner")
    )

    # Same for TEXT
    perf_t = pd.read_csv(f"{UNIMODAL_DIR}/metrics/text_retrieval_results.csv")
    exp_t  = pd.read_csv(f"{UNIMODAL_DIR}/metrics/Explainability_Text.csv")

    perf_t = normalize(perf_t)
    exp_t  = normalize(exp_t)
    eff_t  = normalize(eff_t)

    eff_t = eff_t.rename(columns={
        "Time_s": "inference_time",
        "Latency_s": "embedding_time",
        "Memory_MB": "memory_mb"
    })

    df_text = (
        perf_t
        .merge(exp_t, on="Model", how="inner")
        .merge(eff_t, on="Model", how="inner")
    )

    # Sort by efficiency
    vision_sorted = df_vision.sort_values("inference_time")
    text_sorted   = df_text.sort_values("inference_time")

    return vision_sorted.iloc[:2], text_sorted.iloc[:2], df_vision, df_text

import pandas as pd
import os

BASE = "/home/aysel/tfe/streamlit_app/data/unimodal/metrics"
def normalize_model_name(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def load_unimodal_metrics():
    # RAW (non-normalized)
    vision_perf_raw = pd.read_csv(f"{BASE}/vision_performance.csv")
    vision_expl_raw = pd.read_csv(f"{BASE}/vision_explainability.csv")
    vision_eff_raw  = pd.read_csv(f"{BASE}/vision_efficiency.csv")

    text_perf_raw = pd.read_csv(f"{BASE}/text_performance.csv")
    text_expl_raw = pd.read_csv(f"{BASE}/text_explainability.csv")
    text_eff_raw  = pd.read_csv(f"{BASE}/text_efficiency.csv")

    # NORMALIZED
    vision_perf_norm = pd.read_csv(f"{BASE}/vision_performance_normalized.csv")
    vision_expl_norm = pd.read_csv(f"{BASE}/vision_explainability_normalized.csv")
    vision_eff_norm  = pd.read_csv(f"{BASE}/vision_efficiency_normalized.csv")

    text_perf_norm = pd.read_csv(f"{BASE}/text_performance_normalized.csv")
    text_expl_norm = pd.read_csv(f"{BASE}/text_explainability_normalized.csv")
    text_eff_norm  = pd.read_csv(f"{BASE}/text_efficiency_normalized.csv")
    
    for df in [vision_perf_raw, vision_expl_raw, vision_eff_raw,
            text_perf_raw, text_expl_raw, text_eff_raw,
            vision_perf_norm, vision_expl_norm, vision_eff_norm,
            text_perf_norm, text_expl_norm, text_eff_norm]:
        df["Model"] = df["Model"].apply(normalize_model_name)

    # MERGE RAW
    df_vision_raw = (
        vision_perf_raw
        .merge(vision_expl_raw, on="Model")
        .merge(vision_eff_raw,  on="Model")
    )
    df_vision_raw["Modality"] = "vision"

    df_text_raw = (
        text_perf_raw
        .merge(text_expl_raw, on="Model")
        .merge(text_eff_raw,  on="Model")
    )
    df_text_raw["Modality"] = "text"

    # MERGE NORMALIZED
    df_vision_norm = (
        vision_perf_norm
        .merge(vision_expl_norm, on="Model")
        .merge(vision_eff_norm,  on="Model")
    )
    df_vision_norm["Modality"] = "vision"

    df_text_norm = (
        text_perf_norm
        .merge(text_expl_norm, on="Model")
        .merge(text_eff_norm,  on="Model")
    )
    df_text_norm["Modality"] = "text"

    return df_vision_raw, df_text_raw, df_vision_norm, df_text_norm

# -----------------------------
# Fusion metadata
# -----------------------------
def load_fusion_metadata():
    return pd.read_csv(f"{PROJECTIONS_DIR}/Flickr8k/fusion_index_metadata.csv")

# -----------------------------
# SOTA embeddings
# -----------------------------
def load_sota_embeddings(model_name):
    if model_name == "CLIP ViT‑B/32" or model_name == "CLIP":
        vision_embs = np.load(vision_emb_path("Flickr8k", "CLIP"))
        text_embs   = np.load(text_emb_path("Flickr8k", "CLIP"))
    elif model_name == "OpenCLIP ViT‑L/14" or model_name == "OpenCLIP":
        vision_embs = np.load(vision_emb_path("Flickr8k", "OpenCLIPL14"))
        text_embs   = np.load(text_emb_path("Flickr8k", "OpenCLIPL14"))
    else:
        raise ValueError(f"Unknown SOTA model: {model_name}")
    return vision_embs, text_embs
