import streamlit as st
from utils.auth import check_login_status, init_db
from utils.eval_db import init_eval_db, insert_human_eval, get_human_eval
init_eval_db()
init_db()
from utils.rag_db import init_rag_db
init_rag_db()
import numpy as np
from utils.ui import fixed_image, section_title
from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)
from pathlib import Path
import time
from utils.rag import (evaluate_rag, get_top50)
from utils.loaders import load_fusion_metadata, load_sota_embeddings, DATASETS_DIR
from utils.rag_db import get_rag_cache, insert_rag_cache
from utils.retrieval_multimodal import embed_image_query, clip_encode_image, openclip_encode_image
from sklearn.metrics.pairwise import cosine_similarity

if not check_login_status():
    st.switch_page("pages/login.py")
    
st.session_state["logged_in"] = True
username=st.session_state["username"]

from utils.sql_retrievals import save_rag_history
from utils.layout import app_header, app_footer
app_header()

# =========================
# Global Styling
# =========================
st.markdown("""
<style>

/* Global background */
body, .stApp {
    background-color: #004783 !important;
}

/* Main white container */
.block-container {
    background-color: white !important;
    padding: 2rem 3rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-top: 2rem;
    max-width: 1400px;        /* keeps content centered */
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Typography */
h1, h2, h3, h4, h5, h6, p, label {
    color: #00345c !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #004783 !important;
    padding-top: 30px;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] a:hover {
    color: #8DADD4 !important;
}

/* Navigation buttons */
.nav-button {
    display: block;
    background-color: #004783;
    color: white !important;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 600;
    transition: 0.2s ease;
}
.nav-button:hover {
    background-color: #005fa3;
    transform: translateY(-2px);
}

/* Feature cards */
.feature-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    text-align: center;
    border: 1px solid #eaeaea;
}
.feature-card h4 {
    color: #004783;
}
.feature-card p {
    color: #333;
}

/* Blue side strips */
.side-strip {
    background-color: #00345c;
    position: fixed;
    top: 0;
    bottom: 0;
    width: 70px;              /* same as login page */
    z-index: -1;
}
.left-strip { left: 0; }
.right-strip { right: 0; }

/* Push content inward (correct selector for Streamlit 1.32+) */
[data-testid="stAppViewContainer"] > .main {
    padding-left: 90px !important;
    padding-right: 90px !important;
}

</style>

<div class="side-strip left-strip"></div>
<div class="side-strip right-strip"></div>

""", unsafe_allow_html=True)

# =========================
# Navigation
# =========================

pages = [
    ("Home", "app.py", "🏠"),
    ("Unimodal", "pages/unimodal.py", "🖼️"),
    ("Alignment", "pages/fusion_results.py", "🔀"),
    ("Multimodal", "pages/multimodal_evaluation.py", "📊"),
    ("RAG", "pages/rag.py", "✨"),
    ("My Account", "pages/my_account.py", "👤")
]

cols = st.columns(len(pages))

for col, (label, path, icon) in zip(cols, pages):
    with col:
        st.page_link(
            path,
            label=label,
            icon=icon
        )


st.markdown("<br>", unsafe_allow_html=True)


import json
from PIL import Image

import pandas as pd
import os

# ---------------------------------------------------------
# Load Flickr8k captions
# ---------------------------------------------------------
def load_flickr8k_captions(path):
    caps = []
    with open(path, "r") as f:
        for line in f:
            _, caption = line.strip().split("\t")
            caps.append(caption)
    return caps

captions_list = load_flickr8k_captions(FLICKR8K_CAPTIONS)

def reset_rag_state():
    for key in ["rag_result", "compare_results", "cache_key"]:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title="RAG Ameliorations", page_icon="✨", layout="wide")
st.title("✨ RAG — Retrieval Augmented Generation")
st.markdown("""
### 🔍 What This Page Does

This page evaluates how **Retrieval‑Augmented Generation (RAG)** improves multimodal image‑to‑text retrieval.

The workflow is:

1. Retrieve the **top‑50 captions** for an image using any of the supported multimodal retrieval models  
   (aligned unimodal models or SOTA models like CLIP / OpenCLIP).

2. Feed these captions into a **Large Language Model (LLM)** to refine them into a single, coherent caption.  
   You can also **edit the prompt** used for refinement.

3. Evaluate the refined caption using:
   - **Quantitative metrics**  
     (cosine similarity, centroid distance, entropy, BLEU)
   - **Optional human‑level evaluation**  
     (relevance, fluency, descriptiveness, correctness)

4. Compare the refined caption across **all models**, using cached results when available.

---

### 🤖 LLM Used for Caption Refinement

This page uses the following model for RAG refinement:

**FLAN‑T5‑Base**  
HuggingFace link: https://huggingface.co/google/flan-t5-base

This model is used to:
- summarize the retrieved captions  
- remove noise and redundancy  
- produce a more coherent, human‑like description  
- adapt to your **custom prompt**  

---

### 💾 Smart Caching

All refined captions, metrics, and semantic differences are **cached in SQLite**.  
This means:

- If you revisit the same **image + model + prompt**, results load instantly  
- Model comparison becomes extremely fast  
- No repeated LLM inference is needed  

""")


def star_rating(label, key):
    st.markdown(f"**{label}**")
    return st.radio(
        "",
        options=[1,2,3,4,5],
        format_func=lambda x: "⭐" * x,
        key=key,
        horizontal=True
    )


# ---------------------------------------------------------
# Model Type Selection
# ---------------------------------------------------------
st.subheader("🤖 Choose Model Type")

model_type = st.radio(
    "Model Type",
    ["Aligned Unimodal Models", "SOTA Models"],
    
)
meta = load_fusion_metadata()

# ---------------------------------------------------------
# Aligned Model Selection
# ---------------------------------------------------------
if model_type == "Aligned Unimodal Models":

    col_v, col_t, col_p = st.columns(3)

    with col_v:
        vision_model = st.selectbox("Vision Model", sorted(meta["vision_model"].unique()))

    with col_t:
        text_model = st.selectbox(
            "Text Model",
            sorted(meta[meta["vision_model"] == vision_model]["text_model"].unique()))

    with col_p:
        projection_name = st.selectbox(
            "Projection",
            sorted(meta[
                (meta["vision_model"] == vision_model) &
                (meta["text_model"] == text_model)
            ]["projection"].unique())
        )

    # Retrieve row
    row = meta[
        (meta["vision_model"] == vision_model) &
        (meta["text_model"] == text_model) &
        (meta["projection"] == projection_name)
    ].iloc[0]

    # Load unimodal embeddings
    Xv = np.load(vision_emb_path("Flickr8k", vision_model))
    Xt = np.load(text_emb_path("Flickr8k", text_model))

    # Load projection matrices
    Wv = np.load(row["Wv_path"])
    Wt = np.load(row["Wt_path"])

    # Project
    Xv = Xv @ Wv
    Xt = Xt @ Wt


# ---------------------------------------------------------
# SOTA Model Selection
# ---------------------------------------------------------
else:
    sota_choice = st.selectbox("SOTA Model", ["CLIP ViT‑B/32", "OpenCLIP ViT‑L/14"])
    Xv, Xt = load_sota_embeddings(sota_choice)

result = None
# ---------------------------------------------------------
# Image Selection
# ---------------------------------------------------------
st.subheader("🖼️ Choose an Image")

# ---------------------------------------------------------
# IMAGE SOURCE SELECTION
# ---------------------------------------------------------
image_source = st.radio(
    "Image Source",
    ["Dataset Image", "Upload Image"],
    horizontal=True,
    key="rag_image_source",
    
)


uploaded_image_path = None

# ---------------------------------------------------------
# OPTION 1 — DATASET IMAGE
# ---------------------------------------------------------
if image_source == "Dataset Image":
    image_mode = "dataset"
    selected_idx = st.session_state.get("selected_idx", 0)
    image_path = IMAGE_PATHS[selected_idx]
    gt_caps = captions_list[selected_idx*5 : selected_idx*5 + 5]
    
    # Ensure selected_idx exists
    if "selected_idx" not in st.session_state:
        st.session_state["selected_idx"] = 0

    scroll_container = st.container(height=350)

    with scroll_container:
        cols = st.columns(10)
        for i, img_path in enumerate(IMAGE_PATHS[:100]):
            with cols[i % 10]:
                if st.button(f"{i}", key=f"img_{i}"):
                    st.session_state["selected_idx"] = i
                    reset_rag_state()

                st.image(img_path, use_container_width=True)

    selected_idx = st.session_state["selected_idx"]
    img_path = IMAGE_PATHS[selected_idx]

    # Ground‑truth captions exist only for dataset images
    gt_caps = captions_list[selected_idx*5 : selected_idx*5 + 5]
    
# ---------------------------------------------------------
# OPTION 2 — UPLOAD IMAGE
# ---------------------------------------------------------
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # CASE 1 — User uploads a new file
    if uploaded_file is not None:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        save_path = upload_dir / f"{username}_{int(time.time())}.jpg"
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        uploaded_image_path = str(save_path)

        # Save permanently in session_state
        st.session_state["uploaded_image_path"] = uploaded_image_path

        # Reset RAG state because this is a NEW image
        reset_rag_state()

    # CASE 2 — Rerun after upload (file_uploader returns None)
    elif "uploaded_image_path" in st.session_state:
        uploaded_image_path = st.session_state["uploaded_image_path"]

    # CASE 3 — No upload yet (first time)
    else:
        st.info("Please upload an image to continue.")
        st.stop()

    # Now we ALWAYS have a valid uploaded_image_path
    image_mode = "upload"
    image_path = st.session_state.get("uploaded_image_path")

    # Display image + GT captions UI
    l, r = st.columns([1,2])
    with l:
        st.image(image_path, caption="Uploaded Image", use_container_width=True)

    with r:
        st.markdown("### ✍️ Provide 5 Ground‑Truth Captions")

        if "user_gt_caps" not in st.session_state:
            st.session_state["user_gt_caps"] = [""] * 5

        user_gt_caps = []
        for i in range(5):
            cap = st.text_input(
                f"Ground‑Truth Caption {i+1}",
                value=st.session_state["user_gt_caps"][i],
                key=f"user_gt_cap_{i}"
            )
            user_gt_caps.append(cap)

        st.session_state["user_gt_caps"] = user_gt_caps
        gt_caps = [c for c in user_gt_caps if c.strip() != ""]

    
st.subheader("🧠 RAG Prompt")

default_prompt = (
    "Rewrite the following captions into a single, coherent, human-like description."
)

user_prompt = st.text_area("Prompt", default_prompt, height=120)


# --- DISPLAY SELECTED IMAGE + GT CAPTIONS ---
left, right = st.columns([1, 5])

with left:
    st.image(image_path, caption="Selected Image", use_container_width=True)

with right:
    st.markdown("### 📝 Ground‑Truth Captions")

    if image_mode == "dataset":
        for cap in gt_caps:
            st.markdown(f"- {cap}")
    else:
        if gt_caps:
            for cap in gt_caps:
                st.markdown(f"- {cap}")
        else:
            st.info("Please provide ground‑truth captions above.")

# ---------------------------------------------------------
# Run RAG
# ---------------------------------------------------------
    
if model_type == "Aligned Unimodal Models":
    model_name = f"{vision_model}__{text_model}__{projection_name}"
else:
    model_name = sota_choice
    
# ---------------------------------------------------------
# RAG EXECUTION (only when button is clicked)
# ---------------------------------------------------------
if "rag_result" not in st.session_state:
    st.session_state["rag_result"] = None

st.subheader("✨ RAG‑Refined Caption")

run_rag = st.button("🚀 Generate Refined Caption")

if run_rag:

    # Determine model name
    if model_type == "Aligned Unimodal Models":
        model_name = f"{vision_model}__{text_model}__{projection_name}"
    else:
        model_name = sota_choice

    # -----------------------------
    # CACHE KEY
    # -----------------------------
    if image_mode == "dataset":
        cache_key = selected_idx
    else:
        cache_key = image_path   # uploaded image path
    
    st.session_state["cache_key"] = cache_key

    # -----------------------------
    # Try cache
    # -----------------------------
    cached = get_rag_cache(cache_key, model_name, user_prompt)

    if cached:
        st.info("Loaded from cache.")
        st.session_state["rag_result"] = cached
        result = cached

    else:
        st.warning("Running RAG refinement… this may take a few seconds.")

        progress = st.progress(0)
        for pct in range(0, 100, 10):
            time.sleep(0.05)
            progress.progress(pct + 10)

        # -----------------------------
        # RETRIEVAL
        # -----------------------------
        if image_mode == "dataset":
            # Use dataset index
            top50_caps, top50_embs, img_emb = get_top50(
                Xv, Xt, selected_idx, captions_list
            )
        else:
            # Uploaded image → compute embedding
            raw_img = Image.open(image_path).convert("RGB")

            if model_type == "Aligned Unimodal Models":
                # Use aligned vision encoder
                xv = embed_image_query(image_path, vision_model).reshape(-1)
                img_emb = xv @ Wv
            else:
                # Use SOTA encoder
                if sota_choice == "CLIP ViT‑B/32":
                    img_emb = clip_encode_image(raw_img).reshape(-1)
                else:
                    img_emb = openclip_encode_image(raw_img).reshape(-1)

            # Compute similarity to all text embeddings
            sims = cosine_similarity(img_emb.reshape(1, -1), Xt).flatten()
            idx = np.argsort(sims)[::-1][:50]

            top50_caps = [captions_list[j] for j in idx]
            top50_embs = Xt[idx]

        # -----------------------------
        # LLM refinement
        # -----------------------------
        with st.spinner("Refining caption with LLM..."):
            result = evaluate_rag(
                Xv=Xv,
                Xt=Xt,
                image_index=selected_idx if image_mode=="dataset" else None,
                captions=captions_list,
                gt_caps=gt_caps,
                prompt=user_prompt,
                img_emb=img_emb,
                top50_caps=top50_caps,
                top50_embs=top50_embs
            )
            st.session_state["rag_result"] = result

        progress.progress(100)
        st.success("Refinement complete!")

        insert_rag_cache(cache_key, model_name, user_prompt, result)

    # Display refined caption
    st.success(result["refined"])

    save_rag_history(
        user=username,
        dataset="Flickr8k",
        image_id=cache_key,
        model=model_name,
        prompt=user_prompt,
        refined_caption=result["refined"]
    )

result = st.session_state["rag_result"]

if result is not None:
    if image_mode == "dataset":
        top50_caps, top50_embs, img_emb = get_top50(Xv, Xt, selected_idx, captions_list)
    else:
        raw_img = Image.open(image_path).convert("RGB")

        if model_type == "Aligned Unimodal Models":
            xv = embed_image_query(image_path, vision_model).reshape(-1)
            img_emb = xv @ Wv
        else:
            if sota_choice == "CLIP ViT‑B/32":
                img_emb = clip_encode_image(raw_img).reshape(-1)
            else:
                img_emb = openclip_encode_image(raw_img).reshape(-1)

        sims = cosine_similarity(img_emb.reshape(1, -1), Xt).flatten()
        idx = np.argsort(sims)[::-1][:50]

        top50_caps = [captions_list[j] for j in idx]
        top50_embs = Xt[idx]


    # Show top‑50 captions
    st.subheader("🔝 Top‑50 Retrieved Captions")
    with st.expander("Show Top‑50"):
        for i, cap in enumerate(top50_caps):
            st.markdown(f"**{i+1}.** {cap}")

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    st.subheader("📊 Evaluation Summary")

    metrics = result["metrics"]

    # Compute diffs once
    cos_diff  = metrics["cosine_refined"]   - metrics["cosine_original"]
    cent_diff = metrics["centroid_refined"] - metrics["centroid_original"]
    ent_diff  = metrics["entropy_refined"]  - metrics["entropy_original"]

    # Create 4 columns: Cosine | Centroid | Entropy | BLEU
    col_cos, col_cent, col_ent, col_bleu = st.columns(4)

    # ---------------- COSINE ----------------
    with col_cos:
        st.markdown("### 🔵 Cosine Similarity")
        st.markdown("""**Range:** -1 → 1  
        Measures alignment between the image embedding and caption embedding.  
        - **1** = perfect alignment  
        - **0** = unrelated  
        - **-1** = opposite meaning  
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Original", f"{metrics['cosine_original']:.4f}")
        with c2:
            st.metric("Refined", f"{metrics['cosine_refined']:.4f}")

        # Interpretation
        if cos_diff > 0:
            st.markdown(f"**Improved** by **+{cos_diff:.4f}** → better image–caption alignment.")
        else:
            st.markdown(f"**Dropped** by **{cos_diff:.4f}** → weaker alignment after refinement.")

    # ---------------- CENTROID ----------------
    with col_cent:
        st.markdown("### 🟣 Centroid Distance")
        st.markdown(""" 
            **Range:** ~0 → 2  
            Measures how close the refined caption is to the “center” of the top‑50 retrieved captions.  
            - **Lower = more representative**  
            - **Higher = more divergent**  
            """)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Original", f"{metrics['centroid_original']:.4f}")
        with c2:
            st.metric("Refined", f"{metrics['centroid_refined']:.4f}")

        if cent_diff < 0:
            st.markdown(f"**Closer** by **{cent_diff:.4f}** → refined caption is more representative.")
        else:
            st.markdown(f"**Further** by **+{cent_diff:.4f}** → refined caption diverges from cluster.")

    # ---------------- ENTROPY ----------------
    with col_ent:
        st.markdown("### 🟢 Entropy")
        st.markdown("""
        **Range:** 0 → ~8  
        Measures lexical diversity.  
        - **Low entropy** = concise, focused  
        - **High entropy** = varied vocabulary  
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Original", f"{metrics['entropy_original']:.4f}")
        with c2:
            st.metric("Refined", f"{metrics['entropy_refined']:.4f}")

        if ent_diff < 0:
            st.markdown(f"**Lower** by **{ent_diff:.4f}** → more concise and focused.")
        else:
            st.markdown(f"**Higher** by **+{ent_diff:.4f}** → more varied vocabulary.")

    # ---------------- BLEU ----------------
    with col_bleu:
        st.markdown("### 🟠 BLEU Score")
        st.markdown("""  
        **Range:** 0 → 1  
        Measures overlap with ground‑truth captions.  
        - **1** = perfect match  
        - **0** = no overlap  
        """)
        _, m, _ = st.columns([1, 3, 1])

        with m: st.metric("BLEU", f"{metrics['bleu']:.4f}")

        st.markdown(
            f"Measures similarity to human captions. "
            f"Score of **{metrics['bleu']:.4f}** indicates overall overlap."
        )

    # ---------------- SEMANTIC DIFFERENCES ----------------
    st.markdown("---")
    st.subheader("🔎 Semantic Differences")

    diff = result["semantic_diff"]

    # Helper to format lists
    def fmt_list(items):
        return ", ".join(items) if items else "None"

    # Create 3 columns: Nouns | Verbs | Adjectives
    col_noun, col_verb, col_adj = st.columns(3)

    with col_noun:
        st.markdown("### 🟦 Nouns")
        st.markdown(f"**Missing:** {fmt_list(diff['missing_nouns'])}")
        st.markdown(f"**Added:** {fmt_list(diff['added_nouns'])}")

    with col_verb:
        st.markdown("### 🟩 Verbs")
        st.markdown(f"**Missing:** {fmt_list(diff['missing_verbs'])}")
        st.markdown(f"**Added:** {fmt_list(diff['added_verbs'])}")

    with col_adj:
        st.markdown("### 🟧 Adjectives")
        st.markdown(f"**Missing:** {fmt_list(diff['missing_adjs'])}")
        st.markdown(f"**Added:** {fmt_list(diff['added_adjs'])}")


    st.markdown("### 🧠 Did RAG Improve the Caption?")

    improved = []

    if metrics["cosine_refined"] > metrics["cosine_original"]:
        improved.append("cosine similarity (better visual alignment)")
    if metrics["centroid_refined"] < metrics["centroid_original"]:
        improved.append("centroid distance (more representative)")
    if metrics["entropy_refined"] < metrics["entropy_original"]:
        improved.append("entropy (more concise)")
    if metrics["bleu"] > 0.0:
        improved.append("BLEU score (closer to ground truth)")

    if improved:
        st.success("RAG improved the caption in: " + ", ".join(improved))
    else:
        st.warning("RAG did not significantly improve the caption based on the metrics.")


    st.markdown("---")
    st.markdown("### 🧑‍🏫 Relevance Feedback")
    cache_key = st.session_state.get("cache_key")
    st.session_state["cache_key"] = cache_key
    st.session_state["model_name"] = model_name

    use_human_eval = st.toggle("Enable human‑level evaluation")

    if use_human_eval:

        # --- FETCH ALL EVALUATIONS FOR THIS IMAGE + MODEL ---
        rows = get_human_eval(cache_key, model_name)

        st.markdown("### 📊 Human‑Level Scores")

        if rows:
            df_human = pd.DataFrame(rows, columns=["relevance", "fluency", "descriptiveness", "correctness"])

            avg_global = df_human.mean().mean()
            avg_per_metric = df_human.mean()
            num_evals = len(df_human)

            st.markdown(f"**Overall Human‑Level Score:** {avg_global:.2f} / 5")
            st.markdown(f"**Number of evaluations:** {num_evals}")

            st.markdown("#### Breakdown by Metric")
            st.write(pd.DataFrame({
                "Metric": ["Relevance", "Fluency", "Descriptiveness", "Correctness"],
                "Average Score": [
                    avg_per_metric["relevance"],
                    avg_per_metric["fluency"],
                    avg_per_metric["descriptiveness"],
                    avg_per_metric["correctness"]
                ]
            }).style.format({"Average Score": "{:.2f}"}))

        else:
            st.info("No human evaluations yet for this image and model.")

        # --- ASK IF USER WANTS TO ADD THEIR OWN EVALUATION ---
        st.markdown("### ⭐ Add Your Evaluation")
        add_eval = st.toggle("Would you like to add your own evaluation?")

        if add_eval:
            relevance = star_rating("Relevance", f"rel_{cache_key}_{model_name}")
            fluency = star_rating("Fluency", f"flu_{cache_key}_{model_name}")
            descriptiveness = star_rating("Descriptiveness", f"desc_{cache_key}_{model_name}")
            correctness = star_rating("Correctness", f"corr_{cache_key}_{model_name}")

            if st.button("Save Evaluation"):
                insert_human_eval(
                    username=username,
                    image_id=st.session_state["cache_key"],
                    model=st.session_state["model_name"],
                    relevance=relevance,
                    fluency=fluency,
                    descriptiveness=descriptiveness,
                    correctness=correctness
                )
                st.success("Your evaluation has been saved!")

st.markdown("---")
st.markdown("### ⚖️ Model Comparison")

cache_key = st.session_state.get("cache_key")

if "compare_results" not in st.session_state:
    st.session_state["compare_results"] = None


run_compare = st.button("📊 Compare models")

if run_compare:
    st.session_state["compare_results"] = "running"

if st.session_state["compare_results"] == "running":

    st.info("Running model comparison… cached results will load instantly.")

    rag_data = {}
    rows = []

    total_models = 2 + (len(meta) if meta is not None else 0)
    progress = st.progress(0)
    count = 0

    # 1. SOTA models
    for sota in ["CLIP ViT‑B/32", "OpenCLIP ViT‑L/14"]:

        cached = get_rag_cache(cache_key, sota, user_prompt)
        if cached:
            rag_data[sota] = cached
        else:
            Xv_s, Xt_s = load_sota_embeddings(sota)
            if image_mode == "dataset":
                rag_data[sota] = evaluate_rag(
                    Xv_s, Xt_s, selected_idx, captions_list, gt_caps, user_prompt
                )
            else:
                # Recompute image embedding for THIS model
                raw_img = Image.open(image_path).convert("RGB")

                if sota == "CLIP ViT‑B/32":
                    img_emb_s = clip_encode_image(raw_img).reshape(-1)
                else:
                    img_emb_s = openclip_encode_image(raw_img).reshape(-1)

                # Compute top‑50 for THIS model
                sims = cosine_similarity(img_emb_s.reshape(1, -1), Xt_s).flatten()
                idx = np.argsort(sims)[::-1][:50]
                top50_caps_s = [captions_list[j] for j in idx]
                top50_embs_s = Xt_s[idx]

                # Now evaluate
                rag_data[sota] = evaluate_rag(
                    Xv_s, Xt_s, None, captions_list, gt_caps, user_prompt,
                    img_emb=img_emb_s,
                    top50_caps=top50_caps_s,
                    top50_embs=top50_embs_s
                )

            insert_rag_cache(cache_key, sota, user_prompt, rag_data[sota])

        count += 1
        progress.progress(int((count / total_models) * 100))

    # 2. Aligned models
    if meta is not None:
        for _, row in meta.iterrows():

            vm = row["vision_model"]
            tm = row["text_model"]
            proj = row["projection"]

            if tm.lower() in ["gpt-2", "gpt2"]:
                tm = "GPT2"

            key = f"{vm}__{tm}__{proj}"

            cached = get_rag_cache(cache_key, key, user_prompt)
            if cached:
                rag_data[key] = cached
            else:
                Xv_u = np.load(vision_emb_path("Flickr8k", vm))
                Xt_u = np.load(text_emb_path("Flickr8k", tm))
                Wv = np.load(row["Wv_path"])
                Wt = np.load(row["Wt_path"])
                Xv_u = Xv_u @ Wv
                Xt_u = Xt_u @ Wt

                if image_mode == "dataset":
                    rag_data[key] = evaluate_rag(
                        Xv_u, Xt_u, selected_idx, captions_list, gt_caps, user_prompt
                    )
                else:
                    # Recompute aligned image embedding
                    xv_u = embed_image_query(image_path, vm).reshape(-1)
                    img_emb_u = xv_u @ Wv

                    # Compute top‑50 for THIS aligned model
                    sims = cosine_similarity(img_emb_u.reshape(1, -1), Xt_u).flatten()
                    idx = np.argsort(sims)[::-1][:50]
                    top50_caps_u = [captions_list[j] for j in idx]
                    top50_embs_u = Xt_u[idx]

                    rag_data[key] = evaluate_rag(
                        Xv_u, Xt_u, None, captions_list, gt_caps, user_prompt,
                        img_emb=img_emb_u,
                        top50_caps=top50_caps_u,
                        top50_embs=top50_embs_u
                    )


                insert_rag_cache(cache_key, key, user_prompt, rag_data[key])

            count += 1
            progress.progress(int((count / total_models) * 100))

    st.success("Model comparison complete!")

    # Save results so they persist
    st.session_state["compare_results"] = rag_data

if st.session_state["compare_results"] not in (None, "running"):

    rag_data = st.session_state["compare_results"]
    rows = []

    for model_key, data in rag_data.items():
        met = data["metrics"]
        rows.append({
            "Model": model_key,
            "Cosine": met["cosine_refined"],
            "Centroid": met["centroid_refined"],
            "Entropy": met["entropy_refined"],
            "BLEU": met["bleu"]
        })

    comparison_df = pd.DataFrame(rows)

    sort_option = st.selectbox(
        "Sort models by:",
        ["Cosine", "Centroid", "Entropy", "BLEU"],
        index=0
    )

    ascending = sort_option in ["Centroid", "Entropy"]  # lower is better
    comparison_df = comparison_df.sort_values(sort_option, ascending=ascending)
    
    styled = comparison_df.style.format({col: "{:.4f}" for col in ["Cosine", "Centroid", "Entropy", "BLEU"]})
    st.dataframe(styled)

    st.markdown("### 🧠 Model Performance Summary")

    best_model = comparison_df.iloc[0]
    selected_row = comparison_df[comparison_df["Model"] == model_name].iloc[0]

    summary = []

    # Cosine
    if selected_row["Cosine"] >= best_model["Cosine"] * 0.98:
        summary.append("excellent visual alignment (cosine similarity)")
    elif selected_row["Cosine"] >= best_model["Cosine"] * 0.90:
        summary.append("strong visual alignment")
    else:
        summary.append("weaker visual alignment")

    # Centroid
    if selected_row["Centroid"] <= best_model["Centroid"] * 1.05:
        summary.append("close to the semantic cluster")
    else:
        summary.append("more distant from the semantic cluster")

    # Entropy
    if selected_row["Entropy"] < best_model["Entropy"]:
        summary.append("more concise wording")
    else:
        summary.append("more diverse wording")

    # BLEU
    if selected_row["BLEU"] >= best_model["BLEU"] * 0.95:
        summary.append("high overlap with ground‑truth captions")
    elif selected_row["BLEU"] > 0:
        summary.append("some overlap with ground‑truth captions")
    else:
        summary.append("little to no overlap with ground‑truth captions")

    st.info(
        f"**{model_name}** shows: " +
        ", ".join(summary) +
        "."
    )

    st.markdown("### 🏆 Refined Captions Overview")

    # Sort by the user's chosen metric (you already implemented sorting)
    # comparison_df is already sorted at this point

    # -----------------------------
    # Identify top 3 models
    # -----------------------------
    top3 = comparison_df.head(3)

    # -----------------------------
    # Build list of all other models
    # -----------------------------
    other_models = comparison_df["Model"].tolist()[3:]  # everything except top 3

    # Build carousel options
    carousel_options = []
    for m in other_models:
        carousel_options.append(f"{m}: {rag_data[m]['refined'][:60]}...")

    # Default selection = current model
    default_index = 0
    for i, m in enumerate(other_models):
        if m == model_name:
            default_index = i

    # -----------------------------
    # 4‑column layout
    # -----------------------------
    col_carousel, col_top1, col_top2, col_top3 = st.columns(4)

    # -----------------------------
    # COLUMN 1 — CAROUSEL
    # -----------------------------
    with col_carousel:
        st.markdown("### 🔄 Other Models")

        if other_models:
            selected_option = st.selectbox(
                "Browse refined captions",
                options=other_models,
                index=default_index
            )

            st.markdown(f"**{selected_option}**")
            st.markdown(f"> {rag_data[selected_option]['refined']}")
        else:
            st.info("Not enough models to show a carousel.")

    # -----------------------------
    # COLUMN 2–4 — TOP 3 MODELS
    # -----------------------------
    top_cols = [col_top1, col_top2, col_top3]

    for (idx, (_, row)), col in zip(enumerate(top3.iterrows()), top_cols):
        m = row["Model"]
        refined = rag_data[m]["refined"]

        with col:
            st.markdown(f"### 🥇 Top {idx+1}")
            st.markdown(f"**{m}**")
            st.markdown(f"> {refined}")



app_footer()
