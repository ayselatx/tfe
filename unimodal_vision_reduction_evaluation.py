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

VISION_MODELS = ["resnet50", "mobilenet_v3", "vit", "deit", "clip_vision"]
REDUCTION_METHODS = ["pca", "svd", "grp", "umap"]
DIMENSIONS = [512, 256, 128, 64, 32, 16]

TOP_K_BASE = 5
TOP_K_REDUCED = 50


# In[ ]:


NTFY_TOPIC = "aysel_tfe_server_9988"

def notify(msg, title="Vision Retrieval Eval"):
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
    path = os.path.join(DATA_DIR, "Results_Unimodal", dataset, "vision", model, "embeddings.npy")
    return np.load(path)

def load_reduced_embeddings(dataset, model, method, dim):
    path = os.path.join(
        DATA_DIR,
        "Results_Unimodal",
        dataset,
        "vision",
        model,
        "Features_Reduced",
        method,
        f"X_vision_{model}_{method}_{dim}_{dataset}.npy"
    )
    return np.load(path)



# # Retrieval Metrics

# ## Self‑Retrieval Accuracy

# In[ ]:


def self_retrieval_accuracy(full_emb, reduced_emb):
    sims = cosine_similarity(reduced_emb, full_emb)
    top1 = sims.argmax(axis=1)
    correct = np.arange(len(full_emb))
    return (top1 == correct).mean()


# ## Top‑k Neighborhood Overlap

# In[ ]:


def topk_overlap(full_emb, reduced_emb, k_base=5, k_reduced=50):
    sims_full = cosine_similarity(full_emb, full_emb)
    sims_reduced = cosine_similarity(reduced_emb, reduced_emb)

    top_full = np.argsort(-sims_full, axis=1)[:, 1:k_base+1]
    top_reduced = np.argsort(-sims_reduced, axis=1)[:, 1:k_reduced+1]

    overlaps = []
    for i in range(len(full_emb)):
        overlap = len(set(top_full[i]).intersection(set(top_reduced[i]))) / k_base
        overlaps.append(overlap)

    return np.mean(overlaps)


# ## Spearman Rank Correlation

# In[ ]:


def rank_correlation(full_emb, reduced_emb):
    sims_full = cosine_similarity(full_emb, full_emb)
    sims_reduced = cosine_similarity(reduced_emb, reduced_emb)

    correlations = []
    for i in range(len(full_emb)):
        rho, _ = spearmanr(sims_full[i], sims_reduced[i])
        correlations.append(rho)

    return np.nanmean(correlations)


# ## Trustworthiness

# In[ ]:


def trustworthiness_score(full_emb, reduced_emb, n_neighbors=10):
    return trustworthiness(full_emb, reduced_emb, n_neighbors=n_neighbors)


# ## Explainability Preservation (Quantus)

# In[ ]:


def explainability_difference(full_attr, reduced_attr):
    return np.mean(np.abs(full_attr - reduced_attr))


# # Execution

# In[ ]:


from datetime import datetime

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

results = []

try:
    for dataset in tqdm(DATASETS, desc="Datasets"):

        notify(f"Starting dataset: {dataset}")
        print(f"[{ts()}] Starting dataset: {dataset}", flush=True)

        for model in tqdm(VISION_MODELS, desc=f"{dataset} models", leave=False):

            notify(f"V: Starting model: {model} on {dataset}")
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

            notify(f"V: Completed model: {model} on {dataset}")
            print(f"[{ts()}] Completed model: {model} on {dataset}", flush=True)

        notify(f"V: Completed dataset: {dataset}")
        print(f"[{ts()}] Completed dataset: {dataset}", flush=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv("vision_retrieval_full_vs_reduced.csv", index=False)

    notify("V: Vision retrieval evaluation complete!", title="Finished")
    print(f"[{ts()}] Vision retrieval evaluation complete!", flush=True)

except Exception as e:
    notify(f"V: ERROR: {str(e)}", title="Vision Eval FAILED")
    print(f"[{ts()}] ERROR: {str(e)}", flush=True)
    raise e


# In[ ]:


def plot_3d(df, metric):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    df_reduced = df[df["scenario"] == "reduced"]

    ax.scatter(
        df_reduced["dim"],
        df_reduced["self_retrieval"],
        df_reduced[metric],
        c=df_reduced["dim"],
        cmap="viridis"
    )

    ax.set_xlabel("Reduced Dimension")
    ax.set_ylabel("Self Retrieval")
    ax.set_zlabel(metric)
    plt.title(f"Dimension vs Self Retrieval vs {metric}")
    plt.show()


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

