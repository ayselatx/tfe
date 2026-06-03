import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
# Canonical paths
from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
)

# Query embedding functions
from utils.retrieval_multimodal import (
    embed_text_query,
    embed_image_query,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1e-9, norms)


# ---------------------------------------------------------
# TEXT → TEXT RETRIEVAL (unimodal)
# ---------------------------------------------------------
def retrieve_text(query_caption, model_name, top_k=20, dataset="Flickr8k"):
    """
    Caption → Caption retrieval using precomputed text embeddings.
    Mirrors the logic of retrieve_vision().
    """

    emb_path = text_emb_path(dataset, model_name)
    X = np.load(emb_path)  # shape (N, D)
    X_norm = normalize_matrix(X)

    with open(FLICKR8K_CAPTIONS) as f:
        captions = [line.strip() for line in f.readlines()]

    try:
        idx = captions.index(query_caption)
    except ValueError:
        raise ValueError(f"Query caption not found: {query_caption}")

    q = X[idx:idx+1]  # shape (1, D)
    q_norm = normalize_matrix(q)

    sims = (X_norm @ q_norm.T).flatten()

    top_idx = np.argsort(sims)[::-1][:top_k]

    return [(captions[i], float(sims[i])) for i in top_idx]


def retrieve_text_custom(query_text, model_name, top_k=20, dataset="Flickr8k"):
    # Load dataset embeddings
    emb_path = text_emb_path(dataset, model_name)
    X = np.load(emb_path)
    X_norm = normalize_matrix(X)

    # Load captions
    with open(FLICKR8K_CAPTIONS) as f:
        captions = [line.strip() for line in f.readlines()]

    # Embed custom query
    q = embed_text_query(query_text, model_name)  # shape (1, D)
    q_norm = normalize_matrix(q)

    # Similarity
    sims = (X_norm @ q_norm.T).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]

    return [(captions[i], float(sims[i])) for i in top_idx]

# ---------------------------------------------------------
# IMAGE → IMAGE RETRIEVAL (unimodal)
# ---------------------------------------------------------

def retrieve_vision(image_path, model_name, dataset="Flickr8k"):
    """
    - Load image embeddings from canonical path
    - Normalize
    - Embed query image using same model
    - Compute cosine similarity
    - Return top-20 images
    """
    emb_path = vision_emb_path(dataset, model_name)
    X = np.load(emb_path)
    X_norm = normalize_matrix(X)

    # Debug dataset embedding shape

    # Embed query image
    q = embed_image_query(image_path, model_name)

    q_norm = normalize_matrix(q)

    sims = cosine_similarity(q_norm, X_norm).flatten()
    top_idx = np.argsort(sims)[::-1][:20]

    imgs = IMAGE_PATHS
    return [(imgs[i], float(sims[i])) for i in top_idx]
