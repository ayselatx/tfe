import streamlit as st
from PIL import Image
import pandas as pd
import io
import os
import cv2
import numpy as np
import matplotlib.cm as cm
import time
from pathlib import Path
from utils.auth import check_login_status, init_db
from utils.layout import app_header, app_footer
from utils.ui import fixed_image, section_title
from utils.loaders import (
    load_vision_metrics,
    load_text_metrics,
    load_unimodal_metrics,
    load_vision_models,
    load_text_models,
)
from utils.retrieval_unimodal import retrieve_vision, retrieve_text, retrieve_text_custom
from utils.plots import plot_recall_bars, plot_radar
from utils.sql_retrievals import (
    init_unimodal_tables,
    save_unimodal_retrieval,
    load_cached_retrieval,
    save_custom_model
)
from utils.paths import (
    device,
    IMAGE_PATHS,
    FLICKR8K_IMG_DIR,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)
from utils.models_registry import TEXT_EXPLAIN_MODELS
from utils.vision_explainability import get_vision_explanations, compute_attributions_on_the_fly
from utils.text_explainability import load_text_explanation_record, visualize_tokens, TEXT_MODEL_DIRNAMES, compute_text_attributions_on_the_fly
from utils.names import normalize_model_name

def overlay_heatmap(img, heatmap, alpha):
    # Ensure single channel
    if heatmap.ndim == 3:
        heatmap = heatmap.mean(axis=2)

    # Convert to uint8 and resize to match original image
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_uint8 = cv2.resize(heatmap_uint8, (img.shape[1], img.shape[0]))

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Normalize both
    img_norm = img.astype(float) / 255.0 if img.max() > 1.0 else img.astype(float)
    heatmap_norm = heatmap_color.astype(float) / 255.0

    # Blend
    overlay = (1 - alpha) * img_norm + alpha * heatmap_norm
    return overlay.clip(0, 1)


vision_info = {
    "resnet50": {
        "type": "Convolutional Neural Network (CNN)",
        "desc": "ResNet‑50 pretrained on ImageNet‑1k at 224×224 resolution.",
        "hf": "https://huggingface.co/microsoft/resnet-50"
    },
    "mobilenet_v3": {
        "type": "Lightweight CNN",
        "desc": "MobileNetV3‑Large pretrained on ImageNet‑1k at 224×224.",
        "hf": "https://huggingface.co/litert-community/MobileNet-v3-large"
    },
    "vit": {
        "type": "Vision Transformer (ViT)",
        "desc": "ViT‑Base pretrained on ImageNet‑21k at 224×224.",
        "hf": "https://huggingface.co/google/vit-base-patch16-224"
    },
    "pvt": {
        "type": "Pyramid Vision Transformer",
        "desc": "PVT‑Tiny pretrained on ImageNet‑1k at 224×224.",
        "hf": "https://huggingface.co/Zetatech/pvt-tiny-224"
    }
}

text_info = {
    "BERT": {
        "type": "Masked Language Model (MLM)",
        "desc": "BERT pretrained on English corpora using masked‑token prediction.",
        "hf": "https://huggingface.co/bert-base-uncased"
    },
    "RoBERTa": {
        "type": "Masked Language Model (MLM)",
        "desc": "RoBERTa is a robustly optimized BERT variant.",
        "hf": "https://huggingface.co/roberta-base"
    },
    "GPT2": {
        "type": "Autoregressive Transformer",
        "desc": "GPT‑2 pretrained to predict the next token in large English corpora.",
        "hf": "https://huggingface.co/gpt2"
    }
}

# ---------------------------------------------------------
# Init + access control
# ---------------------------------------------------------
init_db()
init_unimodal_tables()
st.set_page_config(page_title="Unimodal Benchmarking", page_icon="📚", layout="wide")

if not check_login_status():
    st.switch_page("pages/login.py")

username = st.session_state.get("username", "anonymous")

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


st.title("📚 Unimodal Benchmarking")

st.info("""
This module evaluates **vision-only** and **text-only** retrieval models.

You can:
- run real-time unimodal retrieval  
- inspect performance, explainability, and efficiency  
- compare models globally  
- export metrics as CSV  
- visualize graphs with explanations  

All interactions will later appear in **My Account**.
""")
# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_vision, tab_text, tab_global = st.tabs(
    ["🖼️ Vision Retrieval", "📘 Text Retrieval", "🌍 Global Summary"]
)


# =========================================================
# Helpers
# =========================================================

def normalize_metric_frames(perf, exp, eff):
    # -----------------------------------------
    # 1. Normalize column names (lowercase)
    # -----------------------------------------
    for df in [perf, exp, eff]:
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure model column exists
        if "model" not in df.columns:
            for c in df.columns:
                if "model" in c or "arch" in c or "backbone" in c:
                    df.rename(columns={c: "model"}, inplace=True)
                    break

    # -----------------------------------------
    # 2. Standardize model names
    # -----------------------------------------
    for df in [perf, exp, eff]:
        df["model"] = df["model"].apply(normalize_model_name)

    # -----------------------------------------
    # 3. Rename sparsity → compactness
    # -----------------------------------------
    if "sparsity" in exp.columns:
        exp.rename(columns={"sparsity": "compactness"}, inplace=True)

    # -----------------------------------------
    # 4. Normalize efficiency metric names
    # -----------------------------------------
    rename_map = {
        "time_s": "inference_time",
        "latency_s": "embedding_time",
        "memory_mb": "memory_mb",
        "memory": "memory_mb",
    }

    eff.rename(
        columns={old: new for old, new in rename_map.items() if old in eff.columns},
        inplace=True,
    )

    # Ensure required columns exist
    for col in ["inference_time", "embedding_time", "memory_mb"]:
        if col not in eff.columns:
            eff[col] = float("nan")

    return perf, exp, eff

# --- Normalize column names for ALL global metrics ---
def normalize_global_df(df):
    # lowercase all columns
    df.columns = [c.strip().lower() for c in df.columns]
    
    df.rename(columns = {"model": "Model"}, inplace=True)

    # rename recall metrics
    df.rename(columns={
        "recall@1": "Recall@1",
        "recall@5": "Recall@5",
        "recall@10": "Recall@10",
    }, inplace=True)

    # rename explainability metrics
    if "sparsity" in df.columns:
        df.rename(columns={"sparsity": "Compactness"}, inplace=True)
        
    df.rename(columns={
        "faithfulness": "Faithfulness",
        "rank_corr": "Rank_corr",
        "complexity": "Complexity",
    }, inplace=True)

    # rename efficiency metrics
    df.rename(columns={
        "inference_time": "Inference_time",
        "embedding_time": "Embedding_time",
        "memory_mb": "Memory_mb",
    }, inplace=True)

    return df

def build_full_metrics_df(perf, exp, eff):
    full_df = perf.merge(exp, on="model", how="outer").merge(eff, on="model", how="outer")
    return full_df


def get_metric_lists(perf, exp, eff):
    perf_metrics = [c for c in perf.columns if c.startswith("recall")]
    exp_candidates = ["faithfulness", "compactness", "rank_corr", "complexity"]
    exp_metrics = [c for c in exp_candidates if c in exp.columns]
    eff_candidates = ["inference_time", "embedding_time", "memory_mb"]
    eff_metrics = [c for c in eff_candidates if c in eff.columns]
    return perf_metrics, exp_metrics, eff_metrics


def metric_label(name: str) -> str:
    return name.replace("_", " ").title()

# =========================================================
# TAB 1 — VISION RETRIEVAL
# =========================================================
with tab_vision:
    section_title("🖼️ Vision Retrieval")
    
    if "vision_results" not in st.session_state:
        st.session_state["vision_results"] = []
    if "vision_has_results" not in st.session_state:
        st.session_state["vision_has_results"] = False
    if "show_heatmaps" not in st.session_state:
        st.session_state["show_heatmaps"] = False


    st.info("""
    **Image → Image retrieval** using unimodal vision encoders.

    Steps:
    1. Select query mode (dataset image or upload image)  
    2. Choose your image  
    3. Choose a model  
    4. Retrieve top‑K similar images  
    """)


    metrics_v = load_vision_metrics()
    models_v = load_vision_models()

    # ---------------------------------------------------------
    # QUERY SOURCE SELECTION
    # ---------------------------------------------------------
    query_source = st.radio(
        "Choose query source:",
        ["Dataset Image", "Upload Image"],
        horizontal=True,
        key="vision_query_source"
    )

    query_path_v = None

    d, m = st.columns([2,1])
    # ---------------------------------------------------------
    # OPTION 1 — DATASET IMAGE
    # ---------------------------------------------------------
    dataset_v = "Flickr8k"
    with d:
        if query_source == "Dataset Image":
            dataset_v = st.selectbox("Dataset", ["Flickr8k"], key="vision_dataset")
            st.write("### Choose a query image")

            l, r = st.columns([1, 7])

            # Initialize selected index
            if "vision_selected_idx" not in st.session_state:
                st.session_state["vision_selected_idx"] = 0

            selected_idx = st.session_state["vision_selected_idx"]

            # Scrollable container
            with r:
                scroll_container = st.container(height=350)

                with scroll_container:
                    cols = st.columns(10)
                    for i, img_path in enumerate(IMAGE_PATHS[:200]):   # adjust number as needed
                        with cols[i % 10]:
                            if st.button(f"{i}", key=f"vision_img_{i}"):
                                st.session_state["vision_selected_idx"] = i
                                st.rerun()

                            st.image(img_path, use_container_width=True)

            # After selection
            selected_idx = st.session_state["vision_selected_idx"]
            query_path_v = IMAGE_PATHS[selected_idx]

            with l:
                st.write("Selected Image")
                img = Image.open(query_path_v).convert("RGB")
                fixed_image(img, size=250)


        # ---------------------------------------------------------
        # OPTION 2 — UPLOAD IMAGE
        # ---------------------------------------------------------
        else:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)

                save_path = upload_dir / f"{username}_{int(time.time())}.jpg"

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                query_path_v = str(save_path)

                # Show preview
                img = Image.open(query_path_v).convert("RGB")
                fixed_image(img, size=250)

    # ---------------------------------------------------------
    # MODEL SELECTION
    # ---------------------------------------------------------
    with m:
        model_options_v = {normalize_model_name(name): name for name in models_v.keys()}
        model_choice_v = st.selectbox(
            "Choose a vision model",
            list(model_options_v.keys()),
            key="vision_model",
        )
        
        
        model_name_v = model_options_v[model_choice_v]
        
        if model_name_v in vision_info:
            info = vision_info[model_name_v]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(info["desc"])
            st.markdown(f"[HuggingFace page]({info['hf']})")

        # Detect change BEFORE updating
        if st.session_state.get("last_vision_model") != model_name_v:
            st.session_state.pop("vision_results", None)

        # Now update the stored model
        st.session_state["last_vision_model"] = model_name_v

    k = st.slider(
        "Select Top‑K",
        min_value=5,
        max_value=50,
        step=5,
        value=20,
        key="topk_slider_i"
    )

    # ---------------------------------------------------------
    # RETRIEVAL
    # ---------------------------------------------------------
    if query_path_v and st.button("Retrieve (Vision)", key="vision_retrieve_button"):

        cached = load_cached_retrieval(
            modality="vision",
            dataset=dataset_v,
            query=query_path_v,
            model=st.session_state["last_vision_model"],
        )

        if cached is not None:
            st.session_state.vision_results = cached
            st.session_state["vision_has_results"] = True
        else:

            results = retrieve_vision(query_path_v, model_name_v)
            st.session_state.vision_results = results
            st.session_state["vision_has_results"] = True

            save_unimodal_retrieval(
                user=username,
                modality="vision",
                dataset=dataset_v,
                query=query_path_v,
                model=model_name_v,
                results=results,
            )

    # ---------------------------------------------------------
    # DISPLAY RETRIEVAL RESULTS
    # ---------------------------------------------------------
    if st.session_state.get("vision_results"):
        st.subheader(f"Top‑{k} Retrieved Images")

        results_container = st.container(height=350)

        with results_container:
            cols = st.columns(5)
            for idx, (img_path, score) in enumerate(st.session_state.vision_results[:k]):
                with cols[idx % 5]:
                    im = Image.open(img_path).convert("RGB")
                    fixed_image(im, size=150)
                    st.caption(f"{score:.4f}")

        df_results_v = pd.DataFrame(
            st.session_state.vision_results,
            columns=["path", "score"],
        )

        filename_v = (
            f"{dataset_v}_"
            f"{os.path.basename(query_path_v)}_"
            f"{st.session_state['last_vision_model']}_results.csv"
        )

        st.download_button(
            "Download Retrieval Results (Vision)",
            df_results_v.to_csv(index=False),
            filename_v,
            "text/csv",
            key="vision_results_download",
        )


    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    st.markdown("### 📐 Metrics")

    perf_v = metrics_v["performance"].copy()
    exp_v = metrics_v["explainability"].copy()
    eff_v = metrics_v["efficiency"].copy()
    perf_v, exp_v, eff_v = normalize_metric_frames(perf_v, exp_v, eff_v)
    

    mode_v = st.radio(
        "View metrics for:",
        ["Selected model", "All models"],
        horizontal=True,
        key="vision_metrics_mode",
    )

    perf_metrics_v, exp_metrics_v, eff_metrics_v = get_metric_lists(perf_v, exp_v, eff_v)

    if mode_v == "Selected model":
        st.subheader(f"Metrics for **{model_choice_v}**")
        
        st.session_state["last_vision_model"] = normalize_model_name(st.session_state["last_vision_model"])

        perf_row_v = perf_v[perf_v["model"] == st.session_state["last_vision_model"]].iloc[0]
        exp_row_v = exp_v[exp_v["model"] == st.session_state["last_vision_model"]].iloc[0]
        eff_row_v = eff_v[eff_v["model"] == st.session_state["last_vision_model"]].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📈 Performance")
            for m in perf_metrics_v:
                st.metric(metric_label(m), f"{perf_row_v[m]:.3f}")

        with col2:
            st.markdown("#### 🧠 Explainability")
            for m in exp_metrics_v:
                st.metric(metric_label(m), f"{exp_row_v[m]:.3f}")

        with col3:
            st.markdown("#### ⚡ Efficiency")
            for m in eff_metrics_v:
                val = eff_row_v[m]
                if "time" in m:
                    st.metric(metric_label(m), f"{val:.3f}s")
                elif "memory" in m:
                    st.metric(metric_label(m), f"{val:.1f} MB")
                else:
                    st.metric(metric_label(m), f"{val:.3f}")

    else:
        st.subheader("Metrics for **all vision models**")
        full_v = build_full_metrics_df(perf_v, exp_v, eff_v)
        st.dataframe(full_v, use_container_width=True)

        st.download_button(
            "Download Metrics (All Vision Models)",
            full_v.to_csv(index=False),
            "vision_all_models_metrics.csv",
            "text/csv",
            key="vision_all_models_metrics_download",
        )

        st.markdown("### Quick Summary")

        # Initialize session state indices
        if "vision_perf_idx" not in st.session_state:
            st.session_state.vision_perf_idx = 0
        if "vision_exp_idx" not in st.session_state:
            st.session_state.vision_exp_idx = 0
        if "vision_eff_idx" not in st.session_state:
            st.session_state.vision_eff_idx = 0

        col1, col2, col3 = st.columns(3)

        # -------------------------
        # PERFORMANCE
        # -------------------------
        with col1:
            st.markdown("#### 📈 Performance")
            c1, _, c3 = st.columns([1, 3, 1])

            if perf_metrics_v:
                if c1.button("◀", key="vision_perf_left"):
                    st.session_state.vision_perf_idx = (st.session_state.vision_perf_idx - 1) % len(perf_metrics_v)
                if c3.button("▶", key="vision_perf_right"):
                    st.session_state.vision_perf_idx = (st.session_state.vision_perf_idx + 1) % len(perf_metrics_v)

                m = perf_metrics_v[st.session_state.vision_perf_idx]
                st.markdown(f"**{metric_label(m)}**")

                for _, row in perf_v.iterrows():
                    st.metric(row["model"], f"{row[m]:.3f}")
            else:
                st.caption("No performance metrics available.")

        # -------------------------
        # EXPLAINABILITY
        # -------------------------
        with col2:
            st.markdown("#### 🧠 Explainability")
            c1, _, c3 = st.columns([1, 3, 1])

            if exp_metrics_v:
                if c1.button("◀", key="vision_exp_left"):
                    st.session_state.vision_exp_idx = (st.session_state.vision_exp_idx - 1) % len(exp_metrics_v)
                if c3.button("▶", key="vision_exp_right"):
                    st.session_state.vision_exp_idx = (st.session_state.vision_exp_idx + 1) % len(exp_metrics_v)

                m = exp_metrics_v[st.session_state.vision_exp_idx]
                st.markdown(f"**{metric_label(m)}**")

                for _, row in exp_v.iterrows():
                    st.metric(row["model"], f"{row[m]:.3f}")
            else:
                st.caption("No explainability metrics available.")

        # -------------------------
        # EFFICIENCY
        # -------------------------
        with col3:
            st.markdown("#### ⚡ Efficiency")
            c1, _, c3 = st.columns([1, 3, 1])

            if eff_metrics_v:
                if c1.button("◀", key="vision_eff_left"):
                    st.session_state.vision_eff_idx = (st.session_state.vision_eff_idx - 1) % len(eff_metrics_v)
                if c3.button("▶", key="vision_eff_right"):
                    st.session_state.vision_eff_idx = (st.session_state.vision_eff_idx + 1) % len(eff_metrics_v)

                m = eff_metrics_v[st.session_state.vision_eff_idx]
                st.markdown(f"**{metric_label(m)}**")

                for _, row in eff_v.iterrows():
                    val = row[m]
                    if "time" in m:
                        st.metric(row["model"], f"{val:.3f}s")
                    elif "memory" in m:
                        st.metric(row["model"], f"{val:.1f} MB")
                    else:
                        st.metric(row["model"], f"{val:.3f}")
            else:
                st.caption("No efficiency metrics available.")

    # ---------------------------------------------------------
    # IMAGE EXPLAINABILITY
    # ---------------------------------------------------------
    if st.session_state.get("vision_has_results", False):

        st.markdown("### 🔥 Image Explainability")

        selected_img = query_path_v

        st.markdown("""
        **What are attribution maps?**

        Attribution methods highlight **which pixels most influenced the model’s embedding**.
        Brighter regions indicate areas the model relied on more heavily.

        These maps are generated using **[Captum](https://captum.ai/)**, a PyTorch interpretability library.
        """)

        # Initialize flag once
        if "show_heatmaps" not in st.session_state:
            st.session_state.show_heatmaps = False

        # Always visible button
        if st.button("Show Heatmaps", key="vision_xai_button"):
            st.session_state.show_heatmaps = True

        # Render heatmaps if flag is set
        if st.session_state.show_heatmaps:

            cmap_choice = st.selectbox(
                "Choose heatmap colormap",
                ["inferno", "magma", "plasma", "viridis", "Spectral", "coolwarm"],
                key="vision_cmap"
            )

            opacity = st.slider(
                "Overlay opacity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="vision_opacity"
            )

            # Progress bar
            progress = st.progress(0)

            with st.spinner("Computing Captum attributions..."):
                # Step 1: start
                progress.progress(10)

                # Step 2: compute explanations (dataset or uploaded)
                orig, heatmaps = get_vision_explanations(
                    st.session_state["last_vision_model"],
                    selected_img,
                    dataset=dataset_v
                )
                progress.progress(80)

            # Step 3: finish
            progress.progress(100)
            st.success("Attributions ready!")


            left, right = st.columns([1, 3])

            with left:
                st.image(orig, caption="Original Image", width=300)

            with right:
                cols = st.columns(4)

                explanations = {
                    "IG": "Integrated Gradients — measures how the output changes as pixels move from a baseline to the actual image.",
                    "Saliency": "Saliency Maps — highlights pixels where small changes most affect the model output.",
                    "GradShap": "GradientShap — combines gradients with noise sampling to estimate pixel importance.",
                    "Occlusion": "Occlusion — masks parts of the image to see which regions reduce model confidence."
                }

                for col, (method, hmap) in zip(cols, heatmaps.items()):

                    if hmap.ndim == 3:
                        hmap = hmap.mean(axis=0)

                    # Apply colormap safely
                    colored = cm.get_cmap(cmap_choice)(hmap)[..., :3]
                    colored_uint8 = (colored * 255).astype(np.uint8)
                    colored_uint8 = np.ascontiguousarray(colored_uint8)

                    col.image(colored_uint8, caption=method)
                    col.caption(explanations[method])

                    # Overlay
                    overlay = overlay_heatmap(orig, hmap, alpha=opacity)
                    overlay_uint8 = (overlay * 255).astype(np.uint8)
                    overlay_uint8 = np.ascontiguousarray(overlay_uint8)

                    col.image(overlay_uint8, caption=f"{method} Overlay (opacity={opacity:.2f})")





    # ---------------------------------------------------------
    # GRAPHS
    # ---------------------------------------------------------
    st.markdown("### 📊 Graphs (Vision)")
    graph_choice_v = st.selectbox(
        "Choose a graph",
        ["Recall@1 Bar Chart", "Explainability Radar"],
        key="vision_graph_choice",
    )

    perf_plot_v = perf_v.copy()
    buf_v = io.BytesIO()

    if graph_choice_v == "Recall@1 Bar Chart" and "recall@1" in perf_plot_v.columns:
        left, _, right = st.columns([2, 1, 2])

        with left:
            fig_v = plot_recall_bars(perf_plot_v, return_fig=True)
            fig_v.savefig(buf_v, format="png")
            st.pyplot(fig_v)

        with right:
            st.markdown("""
            ### 📘 What is Recall@1?

            **Recall@1** measures how often the model retrieves the **correct matching image** as the **top‑1 result**.

            Retrieval is based on **cosine similarity** between embeddings:
            - Higher similarity → images are closer in embedding space  
            - The model succeeds if the correct match is ranked #1  

            This is a measure of **self‑retrieval performance**.
            """)

    elif graph_choice_v == "Explainability Radar" and not exp_v.empty:
        left, _, right = st.columns([2, 1, 3])

        with left:
            fig_v = plot_radar(exp_v, return_fig=True)
            fig_v.savefig(buf_v, format="png")
            st.pyplot(fig_v)

        with right:
            st.markdown("""
            ### 🧠 What is this radar plot?

            These metrics come from **[Quantus](https://quantus.readthedocs.io/)**, a library for evaluating explainability.

            Each axis measures a different property of an attribution map:
            - **Faithfulness** — does removing important pixels reduce model confidence?  
            - **Robustness** — do explanations change under noise?  
            - **Complexity** — how noisy or fragmented is the explanation?  
            - **Localization** — does the explanation focus on the object?  

            Higher values indicate **better explainability quality**.
            """)

# =========================================================
# TAB 2 — TEXT RETRIEVAL
# =========================================================
with tab_text:
    

    section_title("📘 Text Retrieval")

    st.info("""
    **Caption → Caption retrieval** using unimodal text encoders.

    Steps:
    1. Select query mode (dataset caption or custom text)  
    2. Provide your caption  
    3. Choose a model  
    4. Retrieve top‑K similar captions  
    """)


    metrics_t = load_text_metrics()
    models_t = load_text_models()
    d, m = st.columns([2,1])


    query_mode = st.radio(
        "Choose query input mode:",
        ["Select from dataset", "Enter custom text"],
        horizontal=True,
        key="text_query_mode"
    )

    with d:
        #caption_list = metrics_t["stress_captions"]
        with open(FLICKR8K_CAPTIONS) as f:
            caption_list = [line.strip() for line in f.readlines()]
    
        if query_mode == "Enter custom text":
            query_caption = st.text_area(
                "Enter your custom query caption",
                placeholder="Type any caption you want...",
                key="custom_text_query"
            )
        else:
            st.write("### Choose a query caption")
            dataset_t = st.selectbox("Dataset (Text)", ["Flickr8k"], key="text_dataset")


            if "text_selected_idx" not in st.session_state:
                st.session_state["text_selected_idx"] = 0

            scroll_container = st.container(height=350)

            with scroll_container:
                cols = st.columns(1)
                for i, cap in enumerate(caption_list[:500]):   # adjust limit as needed
                    with cols[i % 1]:
                        if st.button(f"{i}: {cap}", key=f"text_cap_{i}"):
                            st.session_state["text_selected_idx"] = i
                            st.rerun()

            selected_idx = st.session_state["text_selected_idx"]
            query_caption = caption_list[selected_idx]

    st.info(f"**Selected caption:** {query_caption}")

    with m:

        model_options_t = {normalize_model_name(name): name for name in models_t.keys()}
        model_choice_t = st.selectbox(
            "Choose a text model",
            list(model_options_t.keys()),
            key="text_model",
        )
        model_name_t = model_options_t[model_choice_t]
        
        if model_name_t in text_info:
            info = text_info[model_name_t]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(info["desc"])
            st.markdown(f"[HuggingFace page]({info['hf']})")
        
    k = st.slider(
        "Select Top‑K",
        min_value=5,
        max_value=50,
        step=5,
        value=20,
        key="topk_slider_t"
    )

    if st.button("Retrieve (Text)", key="text_retrieve_button"):

        if query_mode == "Enter custom text":
            results_t = retrieve_text_custom(query_caption, model_name_t)
            st.session_state.text_results = results_t

        else:
            # Existing cached retrieval for dataset captions
            cached_t = load_cached_retrieval(
                modality="text",
                dataset=dataset_t,
                query=query_caption,
                model=model_name_t,
            )
            if cached_t is not None:
                st.session_state.text_results = cached_t
            else:
                results_t = retrieve_text(query_caption, model_name_t)
                st.session_state.text_results = results_t
                save_unimodal_retrieval(
                    user=username,
                    modality="text",
                    dataset=dataset_t,
                    query=query_caption,
                    model=model_name_t,
                    results=results_t,
                )

        # ---------------------------------------------------------
        # TOP‑K RETRIEVED CAPTIONS WITH IMAGES
        # ---------------------------------------------------------
        if "text_results" in st.session_state:
            st.markdown("### 🖼️ Top‑K Retrieved Captions")

            results_container = st.container(height=350)

            with results_container:
                top_results = st.session_state.text_results[:k]

                for cap, score in top_results:
                    img_name = cap.split("#")[0]
                    col_img, col_text = st.columns([1, 5])

                    with col_img:
                        st.image(os.path.join(FLICKR8K_IMG_DIR, img_name), width=150)

                    with col_text:
                        img_name, rest = cap.split("#", 1)
                        cap_index, caption_text = rest.split(" ", 1)
                        formatted = f"{img_name} - #{cap_index} {caption_text} — {score:.4f}"
                        st.markdown(f"**{formatted}**")


    # ---------------- Metrics ----------------
    st.markdown("### 📐 Metrics")

    perf_t = metrics_t["performance"].copy()
    exp_t = metrics_t["explainability"].copy()
    eff_t = metrics_t["efficiency"].copy()
    perf_t, exp_t, eff_t = normalize_metric_frames(perf_t, exp_t, eff_t)
    
    # Normalize model names for consistency
    for df in [perf_t, exp_t, eff_t]:
        df["model"] = df["model"].apply(normalize_model_name)
    model_name_t = model_name_t


    mode_t = st.radio(
        "View metrics for:",
        ["Selected model", "All models"],
        horizontal=True,
        key="text_metrics_mode",
    )

    perf_metrics_t, exp_metrics_t, eff_metrics_t = get_metric_lists(perf_t, exp_t, eff_t)

    if mode_t == "Selected model":
        st.subheader(f"Metrics for **{model_choice_t}**")
        
        model_choice_t = normalize_model_name(model_choice_t)


        perf_row_t = perf_t[perf_t["model"] == model_name_t].iloc[0]
        exp_row_t = exp_t[exp_t["model"] == model_name_t].iloc[0]
        eff_row_t = eff_t[eff_t["model"] == model_name_t].iloc[0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📈 Performance")
            for m in perf_metrics_t:
                st.metric(metric_label(m), f"{perf_row_t[m]:.3f}")

        with col2:
            st.markdown("#### 🧠 Explainability")
            for m in exp_metrics_t:
                st.metric(metric_label(m), f"{exp_row_t[m]:.3f}")

        with col3:
            st.markdown("#### ⚡ Efficiency")
            for m in eff_metrics_t:
                val = eff_row_t[m]
                if "time" in m:
                    st.metric(metric_label(m), f"{val:.3f}s")
                elif "memory" in m:
                    st.metric(metric_label(m), f"{val:.1f} MB")
                else:
                    st.metric(metric_label(m), f"{val:.3f}")

        selected_metrics_t = {
            "metric": perf_metrics_t + exp_metrics_t + eff_metrics_t,
            "value": (
                [perf_row_t[m] for m in perf_metrics_t]
                + [exp_row_t[m] for m in exp_metrics_t]
                + [eff_row_t[m] for m in eff_metrics_t]
            ),
        }
        selected_df_t = pd.DataFrame(selected_metrics_t)
        st.download_button(
            "Download Metrics (Selected Text Model)",
            selected_df_t.to_csv(index=False),
            f"{model_name_t}_text_metrics.csv",
            "text/csv",
            key="text_selected_model_metrics_download",
        )

    else:
        st.subheader("Metrics for **all text models**")
        full_t = build_full_metrics_df(perf_t, exp_t, eff_t)
        st.dataframe(full_t, use_container_width=True)
        st.download_button(
            "Download Metrics (All Text Models)",
            full_t.to_csv(index=False),
            "text_all_models_metrics.csv",
            "text/csv",
            key="text_all_models_metrics_download",
        )

        st.markdown("### Quick Summary")

        if "text_perf_idx" not in st.session_state:
            st.session_state.text_perf_idx = 0
        if "text_exp_idx" not in st.session_state:
            st.session_state.text_exp_idx = 0
        if "text_eff_idx" not in st.session_state:
            st.session_state.text_eff_idx = 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📈 Performance")
            c1, _, c3 = st.columns([1, 3, 1])
            if perf_metrics_t:
                if c1.button("◀", key="text_perf_left"):
                    st.session_state.text_perf_idx = (st.session_state.text_perf_idx - 1) % len(perf_metrics_t)
                if c3.button("▶", key="text_perf_right"):
                    st.session_state.text_perf_idx = (st.session_state.text_perf_idx + 1) % len(perf_metrics_t)
                m = perf_metrics_t[st.session_state.text_perf_idx]
                st.markdown(f"**{metric_label(m)}**")
                for _, row in perf_t.iterrows():
                    st.metric(row["model"], f"{row[m]:.3f}")
            else:
                st.caption("No performance metrics available.")

        with col2:
            st.markdown("#### 🧠 Explainability")
            c1, _, c3 = st.columns([1, 3, 1])
            if exp_metrics_t:
                if c1.button("◀", key="text_exp_left"):
                    st.session_state.text_exp_idx = (st.session_state.text_exp_idx - 1) % len(exp_metrics_t)
                if c3.button("▶", key="text_exp_right"):
                    st.session_state.text_exp_idx = (st.session_state.text_exp_idx + 1) % len(exp_metrics_t)
                m = exp_metrics_t[st.session_state.text_exp_idx]
                st.markdown(f"**{metric_label(m)}**")
                for _, row in exp_t.iterrows():
                    st.metric(row["model"], f"{row[m]:.3f}")
            else:
                st.caption("No explainability metrics available.")

        with col3:
            st.markdown("#### ⚡ Efficiency")
            c1, _, c3 = st.columns([1, 3, 1])
            if eff_metrics_t:
                if c1.button("◀", key="text_eff_left"):
                    st.session_state.text_eff_idx = (st.session_state.text_eff_idx - 1) % len(eff_metrics_t)
                if c3.button("▶", key="text_eff_right"):
                    st.session_state.text_eff_idx = (st.session_state.text_eff_idx + 1) % len(eff_metrics_t)
                m = eff_metrics_t[st.session_state.text_eff_idx]
                st.markdown(f"**{metric_label(m)}**")
                for _, row in eff_t.iterrows():
                    val = row[m]
                    if "time" in m:
                        st.metric(row["model"], f"{val:.3f}s")
                    elif "memory" in m:
                        st.metric(row["model"], f"{val:.1f} MB")
                    else:
                        st.metric(row["model"], f"{val:.3f}")
            else:
                st.caption("No efficiency metrics available.")
                
                
    # ---------------------------------------------------------
    # TEXT EXPLAINABILITY
    # ---------------------------------------------------------
    if "text_results" in st.session_state:
        st.markdown("### 🔥 Text Explainability")

        st.markdown("""
        **What are token attributions?**

        Token attribution methods highlight **which words contributed most** to the model’s embedding.
        Higher attribution → the model relied more on that word when computing similarity.

        These attributions are generated using **Captum**.
        """)

        # Persistent flag
        if "show_text_attrib" not in st.session_state:
            st.session_state.show_text_attrib = False

        if st.button("Show Token Attributions", key="text_xai_button"):
            st.session_state.show_text_attrib = True

        if st.session_state.show_text_attrib:

            # Load model + tokenizer
            model, tok, backbone = TEXT_EXPLAIN_MODELS[model_name_t](device)

            # ---------------------------------------------------------
            # Detect dataset caption vs user-typed caption
            # ---------------------------------------------------------
            is_dataset_caption = query_caption in caption_list

            if is_dataset_caption:
                # Dataset → load precomputed
                caption_index = caption_list.index(query_caption)
                image_index = caption_index // 5

                entry = load_text_explanation_record(model_name_t, image_index)
                input_ids = entry["input_ids"]
                attributions = entry["IG"]

            else:
                # User typed → compute IG live
                with st.spinner("Computing token attributions..."):
                    input_ids, attributions = compute_text_attributions_on_the_fly(
                        model, tok, query_caption
                    )
                st.success("Attributions ready!")

            # ---------------------------------------------------------
            # Visualize
            # ---------------------------------------------------------
            l,m, r = st.columns([1,1,1])
            fig = visualize_tokens(
                tok,
                input_ids,
                attributions,
                title=f"{model_name_t} — IG Token Attribution"
            )
            with m:st.pyplot(fig)

            # ---------------------------------------------------------
            # COMPARISON MODE 
            # ---------------------------------------------------------
            if st.button("Compare Across Models", key="text_xai_compare"):
                st.markdown("### 🔍 Model Comparison: Token Attributions")

                model_keys = list(TEXT_EXPLAIN_MODELS.keys())
                cols = st.columns(len(model_keys))

                for col, model_key in zip(cols, model_keys):
                    dirname = TEXT_MODEL_DIRNAMES.get(model_key.lower(), model_key)

                    with col:
                        st.markdown(f"#### **{dirname}**")

                        # Load model + tokenizer
                        model_cmp, tok_cmp, _ = TEXT_EXPLAIN_MODELS[model_key](device)

                        # ---------------------------------------------------------
                        # Dataset caption → load precomputed
                        # ---------------------------------------------------------
                        if is_dataset_caption:
                            caption_index = caption_list.index(query_caption)
                            image_index = caption_index // 5
                            entry = load_text_explanation_record(model_key, image_index)
                            input_ids_cmp = entry["input_ids"]
                            attributions_cmp = entry["IG"]

                        # ---------------------------------------------------------
                        # Custom caption → compute IG live for each model
                        # ---------------------------------------------------------
                        else:
                            with st.spinner(f"Computing IG for {dirname}..."):
                                input_ids_cmp, attributions_cmp = compute_text_attributions_on_the_fly(
                                    model_cmp, tok_cmp, query_caption
                                )

                        # ---------------------------------------------------------
                        # Visualize
                        # ---------------------------------------------------------
                        l, m, r = st.columns([1, 1, 1])
                        fig_cmp = visualize_tokens(
                            tok_cmp,
                            input_ids_cmp,
                            attributions_cmp,
                            title=f"{dirname} — IG Token Attribution"
                        )
                        with m:st.pyplot(fig_cmp)


    # ---------------------------------------------------------
    # GRAPHS (TEXT)
    # ---------------------------------------------------------
    st.markdown("### 📊 Graphs (Text)")

    graph_choice_t = st.selectbox(
        "Choose a graph",
        ["Recall@1 Bar Chart", "Explainability Radar"],
        key="text_graph_choice",
    )

    perf_plot_t = perf_t.copy()
    buf_t = io.BytesIO()

    # -----------------------------
    # Recall@1 Bar Chart (TEXT)
    # -----------------------------
    if graph_choice_t == "Recall@1 Bar Chart" and "recall@1" in perf_plot_t.columns:
        left, _, right = st.columns([2, 1, 2])

        with left:
            fig_t = plot_recall_bars(perf_plot_t, return_fig=True)
            fig_t.savefig(buf_t, format="png")
            st.pyplot(fig_t)

        with right:
            st.markdown("""
            ### 📘 What is Recall@1?

            **Recall@1** measures how often the model retrieves the  
            **correct matching caption** as the **top‑1 result**.

            Retrieval is based on **cosine similarity** between text embeddings:
            - Higher similarity → captions are closer in embedding space  
            - The model succeeds if the correct caption is ranked #1  

            This is a measure of **text‑to‑text retrieval performance**.
            """)

    # -----------------------------
    # Explainability Radar (TEXT)
    # -----------------------------
    elif graph_choice_t == "Explainability Radar" and not exp_t.empty:
        left, _, right = st.columns([2, 1, 3])

        with left:
            fig_t = plot_radar(exp_t, return_fig=True)
            fig_t.savefig(buf_t, format="png")
            st.pyplot(fig_t)

        with right:
            st.markdown("""
            ### 🧠 What is this radar plot?

            These metrics come from **Quantus**, a library for evaluating  
            the quality of **text attribution explanations**.

            Each axis measures a different property of token‑level attributions:
            - **Faithfulness** — do important tokens truly affect predictions  
            - **Robustness** — do explanations remain stable under noise  
            - **Complexity** — how fragmented or noisy the explanation is  
            - **Localization** — does the explanation focus on meaningful tokens  

            Higher values indicate **better explainability quality**.
            """)

        
# =========================================================
# TAB 3 — GLOBAL SUMMARY
# =========================================================
import matplotlib.pyplot as plt
import numpy as np
import io

def export_all_radars(figs):
    buf = io.BytesIO()
    big_fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, (fig, title) in zip(axes, figs):
        img = fig.canvas.renderer.buffer_rgba()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=10)

    big_fig.tight_layout()
    big_fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    return buf

RADAR_COLORS = {
    "main": "#1f77b4",          # blue
    "performance": "#2ca02c",   # green
    "explainability": "#ff7f0e",# orange
    "efficiency": "#d62728"     # red
}

def plot_radar(criteria, values, title, color):
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=8)
    ax.set_yticklabels([])

    # Remove title from inside the plot
    ax.set_title("")

    return fig

def plot_radar_subplot(ax, criteria, values, title):
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)

def normalize_main_weights(changed):
    """Automatically renormalize α, β, γ when one slider changes."""
    a = st.session_state.alpha
    b = st.session_state.beta
    g = st.session_state.gamma

    if changed == "alpha":
        remaining = 1 - a
        st.session_state.beta = remaining * (b / (b + g))
        st.session_state.gamma = remaining * (g / (b + g))

    elif changed == "beta":
        remaining = 1 - b
        st.session_state.alpha = remaining * (a / (a + g))
        st.session_state.gamma = remaining * (g / (a + g))

    elif changed == "gamma":
        remaining = 1 - g
        st.session_state.alpha = remaining * (a / (a + b))
        st.session_state.beta  = remaining * (b / (a + b))
            
def normalize_perf(changed):
    r1 = st.session_state.w_r1
    r5 = st.session_state.w_r5
    r10 = st.session_state.w_r10

    if changed == "w_r1":
        remaining = 1 - r1
        st.session_state.w_r5  = remaining * (r5 / (r5 + r10))
        st.session_state.w_r10 = remaining * (r10 / (r5 + r10))

    elif changed == "w_r5":
        remaining = 1 - r5
        st.session_state.w_r1  = remaining * (r1 / (r1 + r10))
        st.session_state.w_r10 = remaining * (r10 / (r1 + r10))

    elif changed == "w_r10":
        remaining = 1 - r10
        st.session_state.w_r1 = remaining * (r1 / (r1 + r5))
        st.session_state.w_r5 = remaining * (r5 / (r1 + r5))
        
def normalize_exp(changed):
    f = st.session_state.w_faith
    s = st.session_state.w_spar
    r = st.session_state.w_rank
    c = st.session_state.w_comp

    others = {"w_faith": (s, r, c), "w_spar": (f, r, c),
            "w_rank": (f, s, c), "w_comp": (f, s, r)}

    changed_val = st.session_state[changed]
    remaining = 1 - changed_val

    # Redistribute proportionally
    o1, o2, o3 = others[changed]
    total = o1 + o2 + o3

    if changed == "w_faith":
        st.session_state.w_spar = remaining * (s / total)
        st.session_state.w_rank = remaining * (r / total)
        st.session_state.w_comp = remaining * (c / total)

    elif changed == "w_spar":
        st.session_state.w_faith = remaining * (f / total)
        st.session_state.w_rank  = remaining * (r / total)
        st.session_state.w_comp  = remaining * (c / total)

    elif changed == "w_rank":
        st.session_state.w_faith = remaining * (f / total)
        st.session_state.w_spar  = remaining * (s / total)
        st.session_state.w_comp  = remaining * (c / total)

    elif changed == "w_comp":
        st.session_state.w_faith = remaining * (f / total)
        st.session_state.w_spar  = remaining * (s / total)
        st.session_state.w_rank  = remaining * (r / total)
               
def normalize_eff(changed):
    inf = st.session_state.w_inf
    emb = st.session_state.w_emb
    mem = st.session_state.w_mem

    if changed == "w_inf":
        remaining = 1 - inf
        st.session_state.w_emb = remaining * (emb / (emb + mem))
        st.session_state.w_mem = remaining * (mem / (emb + mem))

    elif changed == "w_emb":
        remaining = 1 - emb
        st.session_state.w_inf = remaining * (inf / (inf + mem))
        st.session_state.w_mem = remaining * (mem / (inf + mem))

    elif changed == "w_mem":
        remaining = 1 - mem
        st.session_state.w_inf = remaining * (inf / (inf + emb))
        st.session_state.w_emb = remaining * (emb / (inf + emb))        
                        
with tab_global:
    section_title("🌍 Global Summary")

    st.info("""
    This section aggregates **all unimodal metrics** and ranks models
    based on normalized performance, explainability, and efficiency.
    """)

    # Load raw + normalized metrics
    df_vision_raw, df_text_raw, df_vision_norm, df_text_norm = load_unimodal_metrics()
    df_vision_raw  = normalize_global_df(df_vision_raw)
    df_text_raw    = normalize_global_df(df_text_raw)
    df_vision_norm = normalize_global_df(df_vision_norm)
    df_text_norm   = normalize_global_df(df_text_norm)
    
    for df in [df_vision_raw, df_text_raw, df_vision_norm, df_text_norm]:
        df["Model"] = df["Model"].apply(normalize_model_name)
    

    # ============================================================
    # WEIGHTING UI
    # ============================================================

    st.header("⚖️ Customize Model Selection Weights")

    # -------------------------
    # MAIN WEIGHTS (α, β, γ)
    # -------------------------
    
    st.subheader("Main Criteria Weights (auto-normalized)")
    
    main_left, main_right = st.columns([1, 1])
    
    with main_left:

        # Initialize session state
        if "alpha" not in st.session_state:
            st.session_state.alpha = 0.33
        if "beta" not in st.session_state:
            st.session_state.beta = 0.33
        if "gamma" not in st.session_state:
            st.session_state.gamma = 0.34

        # Sliders with callbacks
        st.slider(
            "Performance Weight (α)",
            0.0, 1.0, st.session_state.alpha, 0.01,
            key="alpha",
            on_change=normalize_main_weights,
            args=("alpha",)
        )

        st.slider(
            "Explainability Weight (β)",
            0.0, 1.0, st.session_state.beta, 0.01,
            key="beta",
            on_change=normalize_main_weights,
            args=("beta",)
        )

        st.slider(
            "Efficiency Weight (γ)",
            0.0, 1.0, st.session_state.gamma, 0.01,
            key="gamma",
            on_change=normalize_main_weights,
            args=("gamma",)
        )

    alpha = st.session_state.alpha
    beta  = st.session_state.beta
    gamma = st.session_state.gamma

    criteria = ["Performance (α)", "Explainability (β)", "Efficiency (γ)"]
    values = [alpha, beta, gamma]

    with main_right:
        fig = plot_radar(criteria, values, "Main Criteria Weights", RADAR_COLORS["main"])
        left, mid, right = st.columns([1, 2, 1])
        with mid:st.pyplot(fig)
        
    st.success(f"Current Weights → α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

    # -------------------------
    # ADVANCED SUB-WEIGHTS
    # -------------------------
    st.subheader("Advanced Metric Weighting")
    advanced = st.checkbox("Enable advanced metric weighting", key="global_advanced")

    if advanced:
        st.markdown("### Performance Sub-Weights (auto-normalized)")
        perf_left, perf_right = st.columns([1, 1])
        
        with perf_left:
            for key, default in [("w_r1", 0.33), ("w_r5", 0.33), ("w_r10", 0.34)]:
                if key not in st.session_state:
                    st.session_state[key] = default

            st.slider("Recall@1 Weight", 0.0, 1.0, st.session_state.w_r1, 0.01,
                    key="w_r1", on_change=normalize_perf, args=("w_r1",))

            st.slider("Recall@5 Weight", 0.0, 1.0, st.session_state.w_r5, 0.01,
                    key="w_r5", on_change=normalize_perf, args=("w_r5",))

            st.slider("Recall@10 Weight", 0.0, 1.0, st.session_state.w_r10, 0.01,
                    key="w_r10", on_change=normalize_perf, args=("w_r10",))

            w_r1  = st.session_state.w_r1
            w_r5  = st.session_state.w_r5
            w_r10 = st.session_state.w_r10
            
        with perf_right:
            perf_criteria = ["R@1", "R@5", "R@10"]
            perf_values = [w_r1, w_r5, w_r10]
            fig_perf = plot_radar(perf_criteria, perf_values, "Performance", RADAR_COLORS["performance"])
            left, mid, right = st.columns([1, 2, 1])
            with mid:st.pyplot(fig_perf)

        # Explainability sub-weights
        st.markdown("### Explainability Sub-Weights (auto-normalized)")

        exp_left, exp_right = st.columns([1, 1])
        
        with exp_left:
            for key, default in [("w_faith", 0.25), ("w_spar", 0.25), ("w_rank", 0.25), ("w_comp", 0.25)]:
                if key not in st.session_state:
                    st.session_state[key] = default

            # Sliders
            st.slider("Faithfulness Weight", 0.0, 1.0, st.session_state.w_faith, 0.01,
                    key="w_faith", on_change=normalize_exp, args=("w_faith",))

            st.slider("Compactness Weight", 0.0, 1.0, st.session_state.w_spar, 0.01,
                    key="w_spar", on_change=normalize_exp, args=("w_spar",))

            st.slider("Rank Correlation Weight", 0.0, 1.0, st.session_state.w_rank, 0.01,
                    key="w_rank", on_change=normalize_exp, args=("w_rank",))

            st.slider("Complexity Weight", 0.0, 1.0, st.session_state.w_comp, 0.01,
                    key="w_comp", on_change=normalize_exp, args=("w_comp",))

            w_faith = st.session_state.w_faith
            w_spar  = st.session_state.w_spar
            w_rank  = st.session_state.w_rank
            w_comp  = st.session_state.w_comp

        with exp_right:
            exp_criteria = ["Faithfulness", "Compactness", "Rank Corr", "Complexity"]
            exp_values = [w_faith, w_spar, w_rank, w_comp]
            fig_exp = plot_radar(exp_criteria, exp_values, "Explainability", RADAR_COLORS["explainability"])
            left, mid, right = st.columns([1, 2, 1])
            with mid:st.pyplot(fig_exp)
            
        # Efficiency sub-weights
        st.markdown("### Efficiency Sub-Weights (auto-normalized)")
        eff_left, eff_right = st.columns([1, 1])
        
        with eff_left:

            for key, default in [("w_inf", 0.33), ("w_emb", 0.33), ("w_mem", 0.34)]:
                if key not in st.session_state:
                    st.session_state[key] = default

            # Sliders
            st.slider("Inference Time Weight", 0.0, 1.0, st.session_state.w_inf, 0.01,
                    key="w_inf", on_change=normalize_eff, args=("w_inf",))

            st.slider("Embedding Time Weight", 0.0, 1.0, st.session_state.w_emb, 0.01,
                    key="w_emb", on_change=normalize_eff, args=("w_emb",))

            st.slider("Memory Weight", 0.0, 1.0, st.session_state.w_mem, 0.01,
                    key="w_mem", on_change=normalize_eff, args=("w_mem",))

            w_inf = st.session_state.w_inf
            w_emb = st.session_state.w_emb
            w_mem = st.session_state.w_mem
        with eff_right:
            eff_criteria = ["Inference", "Embedding", "Memory"]
            eff_values = [w_inf, w_emb, w_mem]
            fig_eff = plot_radar(eff_criteria, eff_values, "Efficiency", RADAR_COLORS["efficiency"])
            left, mid, right = st.columns([1, 2, 1])
            with mid:st.pyplot(fig_eff)

        #st.header("📊 Criteria Weight Profiles")
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), subplot_kw=dict(polar=True))

        # PERFORMANCE
        perf_criteria = ["R@1", "R@5", "R@10"]
        perf_values = [w_r1, w_r5, w_r10]
        #plot_radar_subplot(axes[0], perf_criteria, perf_values, "Performance")

        # EXPLAINABILITY
        exp_criteria = ["Faithfulness", "Compactness", "Rank Corr", "Complexity"]
        exp_values = [w_faith, w_spar, w_rank, w_comp]
        #plot_radar_subplot(axes[1], exp_criteria, exp_values, "Explainability")

        # EFFICIENCY
        eff_criteria = ["Inference", "Embedding", "Memory"]
        eff_values = [w_inf, w_emb, w_mem]
        #plot_radar_subplot(axes[2], eff_criteria, eff_values, "Efficiency")

        #plt.tight_layout()
        #left, mid, right = st.columns([1, 10, 1])
        #with mid:st.pyplot(fig)
        
        st.subheader("📊 Full Criteria Overview")

        col1, col2, col3, col4 = st.columns(4)

        # MAIN
        with col1:
            fig_main = plot_radar(
                ["Performance (α)", "Explainability (β)", "Efficiency (γ)"],
                [alpha, beta, gamma],
                "Main",
                RADAR_COLORS["main"]
            )
            st.pyplot(fig_main)
            st.markdown("<center>Main</center>", unsafe_allow_html=True)

        # PERFORMANCE
        with col2:
            fig_perf = plot_radar(
                ["R@1", "R@5", "R@10"],
                [w_r1, w_r5, w_r10],
                "Performance",
                RADAR_COLORS["performance"]
            )
            st.pyplot(fig_perf)
            st.markdown("<center>Performance</center>", unsafe_allow_html=True)

        # EXPLAINABILITY
        with col3:
            fig_exp = plot_radar(
                ["Faithfulness", "Compactness", "Rank Corr", "Complexity"],
                [w_faith, w_spar, w_rank, w_comp],
                "Explainability",
                RADAR_COLORS["explainability"]
            )
            st.pyplot(fig_exp)
            st.markdown("<center>Explainability</center>", unsafe_allow_html=True)

        # EFFICIENCY
        with col4:
            fig_eff = plot_radar(
                ["Inference", "Embedding", "Memory"],
                [w_inf, w_emb, w_mem],
                "Efficiency",
                RADAR_COLORS["efficiency"]
            )
            st.pyplot(fig_eff)
            st.markdown("<center>Efficiency</center>", unsafe_allow_html=True)

        all_figs = [
            (fig_main, "Main"),
            (fig_perf, "Performance"),
            (fig_exp, "Explainability"),
            (fig_eff, "Efficiency")
        ]

        buf = export_all_radars(all_figs)

        st.download_button(
            label="📥 Download All Radar Plots",
            data=buf,
            file_name="criteria_radar_summary.png",
            mime="image/png"
        )


    else:
        w_r1 = w_r5 = w_r10 = 1/3
        w_faith = w_spar = w_rank = w_comp = 1/4
        w_inf = w_emb = w_mem = 1/3

        # Simple mode → single radar
        criteria = ["Performance (α)", "Explainability (β)", "Efficiency (γ)"]
        values = [alpha, beta, gamma]

    st.header("📊 Criteria Weight Profiles")

    # ============================================================
    # APPLY WEIGHTS
    # ============================================================

    def compute_scores(df):
        df["Performance_score"] = (
            w_r1  * df["Recall@1"] +
            w_r5  * df["Recall@5"] +
            w_r10 * df["Recall@10"]
        )

        df["Explainability_score"] = (
            w_faith * df["Faithfulness"] +
            w_spar  * df["Compactness"] +
            w_rank  * df["Rank_corr"] +
            w_comp  * df["Complexity"]
        )

        df["Efficiency_score"] = (
            -w_inf * df["Inference_time"] +
            -w_emb * df["Embedding_time"] +
            -w_mem * df["Memory_mb"]
        )

        df["Final_score"] = (
            alpha * df["Performance_score"] +
            beta  * df["Explainability_score"] +
            gamma * df["Efficiency_score"]
        )

    # ============================================================
    # RANK BUTTON
    # ============================================================
    if st.button("Rank Models", key="rank_models_button"):

        compute_scores(df_vision_norm)
        compute_scores(df_text_norm)

        st.session_state.ranked = True

        st.session_state.vision_best = df_vision_norm.sort_values("Final_score", ascending=False)
        st.session_state.text_best   = df_text_norm.sort_values("Final_score", ascending=False)

        # Save weights in session_state
        st.session_state.saved_weights = {
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "w_r1": w_r1, "w_r5": w_r5, "w_r10": w_r10,
            "w_faith": w_faith, "w_spar": w_spar,
            "w_rank": w_rank, "w_comp": w_comp,
            "w_inf": w_inf, "w_emb": w_emb, "w_mem": w_mem
        }

        # Build explanation
        dominant = max(
            [("performance", alpha), ("explainability", beta), ("efficiency", gamma)],
            key=lambda x: x[1]
        )[0]

        explanation = ""

        if dominant == "performance":
            explanation += "You prioritize **performance**, meaning you want a model that retrieves the correct item as accurately as possible.\n\n"
            explanation += f"- You emphasize **Recall@1 ({w_r1:.2f})**, so you want the top result to be correct.\n" if w_r1 > 0.5 else ""
            explanation += f"- You value **Recall@5 ({w_r5:.2f})** and **Recall@10 ({w_r10:.2f})**, meaning broader retrieval quality matters.\n"

        elif dominant == "explainability":
            explanation += "You prioritize **explainability**, meaning you want a model whose decisions are transparent and trustworthy.\n\n"
            explanation += f"- High weight on **Faithfulness ({w_faith:.2f})** means you want explanations that truly reflect model reasoning.\n" if w_faith > 0.4 else ""
            explanation += f"- High **Compactness ({w_spar:.2f})** means you prefer concise, focused explanations.\n" if w_spar > 0.4 else ""
            explanation += f"- High **Rank Correlation ({w_rank:.2f})** means you want stable, consistent explanations.\n" if w_rank > 0.4 else ""
            explanation += f"- High **Complexity ({w_comp:.2f})** means you tolerate richer, more detailed explanations.\n" if w_comp > 0.4 else ""

        else:
            explanation += "You prioritize **efficiency**, meaning you want a lightweight, fast model.\n\n"
            explanation += f"- High weight on **Inference Time ({w_inf:.2f})** means you want fast predictions.\n" if w_inf > 0.4 else ""
            explanation += f"- High **Embedding Time ({w_emb:.2f})** means you care about preprocessing speed.\n" if w_emb > 0.4 else ""
            explanation += f"- High **Memory ({w_mem:.2f})** means you want low GPU/CPU usage.\n" if w_mem > 0.4 else ""

        st.session_state.explanation = explanation


    # ============================================================
    # DISPLAY RESULTS (PERSISTENT)
    # ============================================================
    if st.session_state.get("ranked", False):

        st.header("🧠 What Your Weighting Says About You")
        st.write(st.session_state.explanation)

        st.header("🏆 Best Vision Model")
        st.dataframe(st.session_state.vision_best.head(1))

        st.header("🥈 Second Best Vision Model")
        st.dataframe(st.session_state.vision_best.iloc[1:2])

        st.header("🏆 Best Text Model")
        st.dataframe(st.session_state.text_best.head(1))

        st.header("🥈 Second Best Text Model")
        st.dataframe(st.session_state.text_best.iloc[1:2])

        st.header("📊 Full Vision Ranking")
        st.dataframe(st.session_state.vision_best)

        st.header("📊 Full Text Ranking")
        st.dataframe(st.session_state.text_best)

        # ============================================================
        # SAVE UI (PERSISTENT)
        # ============================================================
        st.header(f"💾 Save This Custom System")
        
        st.write("Vision Model: " + st.session_state.vision_best.iloc[0]["Model"])
        st.write("Text Model: " + st.session_state.text_best.iloc[0]["Model"])

        model_name = st.text_input("System Name", placeholder="e.g. High‑Explainability System")
        model_comment = st.text_area("Comment / Rationale", placeholder="Why did you choose these weights?")

        if st.button("Save Model Configuration", key="save_model_button"):
            if not model_name.strip():
                st.error("Please provide a model name.")
            else:
                save_custom_model(
                    user=username,
                    name=model_name,
                    comment=model_comment,
                    weights=st.session_state.saved_weights,
                    best_vision=st.session_state.vision_best.iloc[0]["Model"],
                    best_text=st.session_state.text_best.iloc[0]["Model"]
                )
                st.success("Your custom system configuration has been saved!")

app_footer()