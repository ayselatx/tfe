#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import time
import requests
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
import umap
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import normalize


# # Configuration

# In[ ]:


NTFY_TOPIC = "aysel_tfe_server_9988" 
# "Flickr8k"  or "Flickr30k" or "ConceptualCaptions"
CURRENT_DATASET = "Flickr8k"  
DATASETS = ["Flickr8k", "Flickr30k", "ConceptualCaptions"]  

BASE_DIR = os.path.join(os.getcwd(), 'TFE_Data')



def send_ntfy(message, title="TFE Production"):
    try:
        requests.post(f"https://ntfy.sh/{NTFY_TOPIC}", 
                      data=message.encode('utf-8'),
                      headers={"Title": title, "Priority": "3"})
    except: pass


# # Data Loading

# In[ ]:


def load_raw_matrix(modality, model_name):
    """Load raw embeddings from the dataset folder structure."""
    file_path = os.path.join(
        BASE_DIR,
        "Results_Unimodal",
        CURRENT_DATASET,
        modality,
        model_name,
        "embeddings.npy"
    )
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing RAW embeddings: {file_path}")
    return np.load(file_path)


# # Dimension Reduction

# In[ ]:


def run_production_batch(modality, model_name):
    print(f"\nProcessing: {model_name.upper()} ({modality})")

    try:
        X_raw = load_raw_matrix(modality, model_name)
        X_raw = normalize(X_raw)  # improves PCA/UMAP stability
    except Exception as e:
        print(f"[SKIP] {model_name}: {e}")
        return

    input_dim = X_raw.shape[1]

    for method_name, reduction_func in REDUCTION_PIPELINE.items():

        # Create directory ONCE per method
        save_dir = os.path.join(
            BASE_DIR,
            "Results_Unimodal",
            CURRENT_DATASET,
            modality,
            model_name,
            "Features_Reduced",
            method_name
        )
        os.makedirs(save_dir, exist_ok=True)

        for dim in DIMENSIONS_TO_TEST:
            if dim >= input_dim:
                continue

            # Build filename + path
            save_filename = f"X_{modality}_{model_name}_{method_name}_{dim}_{CURRENT_DATASET}.npy"
            save_path = os.path.join(save_dir, save_filename)

            # Skip if already computed
            if os.path.exists(save_path):
                print(f"[SKIP] {save_filename}")
                continue

            # Compute reduction
            start_red = time.time()
            X_reduced = reduction_func(X_raw, dim)
            print(f"{method_name} {dim}D took {time.time() - start_red:.2f}s")

            # Save result
            np.save(save_path, X_reduced)

    print(f"Finished all reductions for {model_name}")
    send_ntfy(f"Reductions for {model_name} ({CURRENT_DATASET}) are saved to disk.")


# # Execution

# ## Parameters

# In[ ]:


from sklearn.decomposition import TruncatedSVD


DIMENSIONS_TO_TEST = [512, 256, 128, 64, 32, 16]

REDUCTION_PIPELINE = {
    "pca": lambda X, d: PCA(n_components=d).fit_transform(X),
    "svd": lambda X, d: TruncatedSVD(n_components=d, random_state=42).fit_transform(X),
    "grp": lambda X, d: GaussianRandomProjection(n_components=d, random_state=42).fit_transform(X),
    "umap": lambda X, d: umap.UMAP(
        n_components=d,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine", 
        random_state=42
    ).fit_transform(X)
}

VISION_MODELS = ["resnet50", "mobilenet_v3", "vit", "deit", "clip_vision"]
TEXT_MODELS = ["rnn", "bert", "roberta", "gpt2", "clip_text"]


# ## Run Experiment

# In[ ]:


def run_full_pipeline():

    start_global = time.time()

    print(f"\n===== STARTING REDUCTIONS: {CURRENT_DATASET} =====\n")

    # --- Vision ---
    for model in VISION_MODELS:
        try:
            run_production_batch("vision", model)
        except Exception as e:
            print(f"[ERROR] Vision {model}: {e}")

    # --- Text ---
    for model in TEXT_MODELS:
        try:
            run_production_batch("text", model)
        except Exception as e:
            print(f"[ERROR] Text {model}: {e}")

    total_time = time.time() - start_global

    print(f"\n===== DONE: {CURRENT_DATASET} in {total_time/60:.2f} min =====")

    send_ntfy(f"{CURRENT_DATASET} reductions COMPLETE in {total_time/60:.2f} min")


# In[ ]:


"""for CURRENT_DATASET in DATASETS:
    print(f"\n=== Starting Batch for Dataset: {CURRENT_DATASET} ===")
    run_full_pipeline()"""


# In[ ]:


# ============================
# TEST SUBSET INDEXATION BLOCK
# ============================

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

