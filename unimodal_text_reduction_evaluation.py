#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

import quantus
import torch

from mpl_toolkits.mplot3d import Axes3D

import requests


# # Configuration

# In[ ]:


DATA_DIR = "TFE_Data"
#DATASETS = ["Flickr8k", "Flickr30k", "ConceptualCaptions"]
DATASETS = ["Flickr8k"]

TEXT_MODELS = ["bert", "roberta", "gpt2", "clip_text"]
REDUCTION_METHODS = ["pca", "svd", "grp", "umap"]
DIMENSIONS = [512, 256, 128, 64, 32, 16]

TOP_K_BASE = 5
TOP_K_REDUCED = 50


# In[ ]:


NTFY_TOPIC = "aysel_tfe_server_9988"

def notify(msg, title="Text Retrieval Eval"):
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=msg.encode("utf-8"),
            headers={"Title": title}
        )
    except:
        pass


# # Data Loading

# In[ ]:


def load_full_embeddings(dataset, model):
    path = os.path.join(
        DATA_DIR, "Results_Unimodal", dataset, "text", model, "embeddings.npy"
    )
    return np.load(path)


def load_reduced_embeddings(dataset, model, method, dim):
    path = os.path.join(
        DATA_DIR,
        "Results_Unimodal",
        dataset,
        "text",
        model,
        "Features_Reduced",
        method,
        f"X_text_{model}_{method}_{dim}_{dataset}.npy"
    )
    return np.load(path)


# # Retrieval Metrics

# ## Self‑Retrieval Accuracy

# In[ ]:


def self_retrieval_accuracy(emb1, emb2):
    sims = cosine_similarity(emb1, emb2)
    top1 = sims.argmax(axis=1)
    correct = np.arange(len(emb1))
    return (top1 == correct).mean()


# ## Top‑k Neighborhood Overlap

# In[ ]:


def topk_overlap(emb1, emb2, k_base=5, k_reduced=50):
    sims_full = cosine_similarity(emb1, emb1)
    sims_reduced = cosine_similarity(emb2, emb2)

    top_full = np.argsort(-sims_full, axis=1)[:, 1:k_base+1]
    top_reduced = np.argsort(-sims_reduced, axis=1)[:, 1:k_reduced+1]

    overlaps = []
    for i in range(len(emb1)):
        overlap = len(set(top_full[i]).intersection(set(top_reduced[i]))) / k_base
        overlaps.append(overlap)

    return np.mean(overlaps)


# ## Spearman Rank Correlation

# In[ ]:


def rank_correlation(emb1, emb2):
    sims_full = cosine_similarity(emb1, emb1)
    sims_reduced = cosine_similarity(emb2, emb2)

    correlations = []
    for i in range(len(emb1)):
        rho, _ = spearmanr(sims_full[i], sims_reduced[i])
        correlations.append(rho)

    return np.nanmean(correlations)


# ## Trustworthiness

# In[ ]:


def trustworthiness_score(emb1, emb2, n_neighbors=10):
    return trustworthiness(emb1, emb2, n_neighbors=n_neighbors)


# ## Explainability Preservation (Quantus)

# For text, explainability is token‑level, not spatial.
# 
# You will compare:
# 
# IG (full) vs IG (reduced)
# 
# Saliency (full) vs Saliency (reduced)
# 
# SHAP (full) vs SHAP (reduced)

# In[ ]:


def explainability_difference(full_attr, reduced_attr):
    full_attr = full_attr.cpu().numpy()
    reduced_attr = reduced_attr.cpu().numpy()
    return np.mean(np.abs(full_attr - reduced_attr))


# # Execution

# In[ ]:


from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

results = []

try:
    for dataset in tqdm(DATASETS, desc="Datasets"):

        notify(f"T: Starting dataset: {dataset}")
        print(f"[{ts()}] Starting dataset: {dataset}", flush=True)

        for model in tqdm(TEXT_MODELS, desc=f"{dataset} models", leave=False):

            notify(f"T: Starting model: {model} on {dataset}")
            print(f"[{ts()}] Starting model: {model} on {dataset}", flush=True)

            full_emb = load_full_embeddings(dataset, model)

            # Baseline metrics (computed once)
            print(f"[{ts()}] Computing baseline metrics for {model} on {dataset}", flush=True)
            baseline_self = self_retrieval_accuracy(full_emb, full_emb)
            baseline_topk = topk_overlap(full_emb, full_emb)
            baseline_rank = rank_correlation(full_emb, full_emb)
            baseline_trust = trustworthiness_score(full_emb, full_emb)

            results.append({
                "dataset": dataset,
                "model": model,
                "method": "full",
                "dim": full_emb.shape[1],
                "scenario": "full",
                "self_retrieval": baseline_self,
                "topk_overlap": baseline_topk,
                "rank_corr": baseline_rank,
                "trustworthiness": baseline_trust
            })

            for method in tqdm(REDUCTION_METHODS, desc=f"{model} methods", leave=False):
                print(f"[{ts()}] Starting method: {method} for {model} on {dataset}", flush=True)

                for dim in DIMENSIONS:

                    try:
                        print(f"[{ts()}] Loading reduced embeddings: {dataset} | {model} | {method} | {dim}", flush=True)
                        reduced_emb = load_reduced_embeddings(dataset, model, method, dim)
                    except Exception as e:
                        print(f"[{ts()}] Failed loading reduced embeddings for {dataset} | {model} | {method} | {dim}: {e}", flush=True)
                        continue

                    print(f"[{ts()}] Running metrics: {dataset} | {model} | {method} | {dim}", flush=True)

                    # Reduced vs reduced
                    res = {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "dim": dim,
                        "scenario": "reduced",
                        "self_retrieval": self_retrieval_accuracy(reduced_emb, reduced_emb),
                        "topk_overlap": topk_overlap(reduced_emb, reduced_emb),
                        "rank_corr": rank_correlation(reduced_emb, reduced_emb),
                        "trustworthiness": trustworthiness_score(reduced_emb, reduced_emb)
                    }

                    results.append(res)

                print(f"[{ts()}] Completed method: {method} for {model} on {dataset}", flush=True)

            notify(f"T: Completed model: {model} on {dataset}")
            print(f"[{ts()}] Completed model: {model} on {dataset}", flush=True)

        notify(f"T: Completed dataset: {dataset}")
        print(f"[{ts()}] Completed dataset: {dataset}", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv("text_retrieval_full_vs_reduced.csv", index=False)

    notify("Text retrieval evaluation complete!", title="Finished")
    print(f"[{ts()}] Text retrieval evaluation complete!", flush=True)

except Exception as e:
    notify(f"T: ERROR: {str(e)}", title="Text Eval FAILED")
    print(f"[{ts()}] ERROR: {str(e)}", flush=True)
    raise e


# # 3D Plotting

# In[ ]:


def plot_3d(df, metric):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["dim"], df["self_retrieval"], df[metric], c=df["dim"], cmap="viridis")

    ax.set_xlabel("Dimension")
    ax.set_ylabel("Self Retrieval")
    ax.set_zlabel(metric)
    plt.title(f"Dimension vs Self Retrieval vs {metric}")
    plt.show()

