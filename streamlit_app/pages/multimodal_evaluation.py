import streamlit as st
st.set_page_config(page_title="Multimodal Evaluation", page_icon="📊", layout="wide")

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
from utils.ui import fixed_image, section_title
import time
from pathlib import Path
from utils.auth import check_login_status, init_db
from utils.loaders import load_fusion_metadata, load_sota_embeddings, DATASETS_DIR

from utils.sql_retrievals import save_retrieval
from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)

from utils.retrieval_multimodal import embed_image_query, clip_encode_image, openclip_encode_image, embed_text_query, clip_encode_text, openclip_encode_text

init_db()
username = st.session_state.get("username", "anonymous")

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


if not check_login_status():
    st.switch_page("pages/login.py")

st.title("📊 Multimodal Evaluation")

st.markdown("""
### 🔎 What this page does

This page evaluates **multimodal retrieval** using either:

- **Aligned Models** — your own aligned vision/text encoders projected into a shared space  
- **SOTA Models** — pretrained contrastive models such as CLIP and OpenCLIP  

You can run four types of retrieval:

- **Image → Text**  
- **Text → Image**  
- **Multimodal → Text**  
- **Multimodal → Image**

This page provides a **benchmark‑faithful evaluation**, using the same Flickr8k ground truth as your unimodal and aligned unimodal models benchmarks.
""")

def load_flickr8k_captions(caption_file):
    captions = []
    with open(caption_file, "r") as f:
        for line in f:
            _, caption = line.strip().split("\t")
            captions.append(caption)
    return captions

# Load Flickr8k dataset
#df = pd.read_pickle(f"{DATASETS_DIR}/df_Flickr8k.pkl")
#images = df["image_path"].tolist()
#captions = [cap for caps in df["captions"].tolist() for cap in caps]
images = IMAGE_PATHS
captions_list = load_flickr8k_captions(FLICKR8K_CAPTIONS)

# Ground truth (same as benchmark)
#gt_i2t = {i: list(range(i*5, i*5 + 5)) for i in range(8091)}
#gt_t2i = {j: j // 5 for j in range(40455)}
gt_i2t = {i: list(range(i*5, i*5 + 5)) for i in range(len(images))}
gt_t2i = {j: j // 5 for j in range(len(captions_list))}

# Directory for projected embeddings
PROJECTED_DIR = "data/projected_embeddings"

# Load fusion metadata
meta = load_fusion_metadata()

# Remove random projection rows
meta = meta[meta["projection"] != "random"]

# Keep only rows where projection matrices exist
valid_rows = []
for idx, row in meta.iterrows():
    if os.path.exists(row["Wv_path"]) and os.path.exists(row["Wt_path"]):
        valid_rows.append(idx)

meta = meta.loc[valid_rows].reset_index(drop=True)


# Session state
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_caption" not in st.session_state:
    st.session_state.selected_caption = None
if "results" not in st.session_state:
    st.session_state.results = None


def fuse_embeddings(xv, xt, method, alpha=0.5):
    if method == "add":
        return xv + xt
    elif method == "gated":
        return alpha * xv + (1 - alpha) * xt
    elif method == "mul":
        return xv * xt
    else:
        raise ValueError("Unknown alignment method")


# UI
st.header("Multimodal Retrieval")
model_type = st.selectbox("Choose Model Type", ["Aligned Unimodal Models", "SOTA Model"])


fusion_info = {
    "add": {
        "name": "Additive Fusion",
        "desc": (
            "Adds the vision and text embeddings element‑wise. "
            "This assumes both modalities contribute equally and preserves their shared structure. "
            "Simple, stable, and often surprisingly strong."
        )
    },
    "gated": {
        "name": "Gated Fusion",
        "desc": (
            "Learns a weighted combination of vision and text embeddings using a gating parameter α. "
            "This allows the model to emphasize one modality more than the other depending on the query."
        )
    },
    "mul": {
        "name": "Multiplicative Fusion",
        "desc": (
            "Multiplies the embeddings element‑wise. "
            "This highlights dimensions where both modalities strongly agree and suppresses mismatched features. "
            "Often produces sharper, more selective retrieval."
        )
    }
}
# ---------------------------------------------------------
# FUSION MODEL RETRIEVAL (Benchmark‑Faithful)
# ---------------------------------------------------------
if model_type == "Aligned Unimodal Models":

    st.header("🔀 Aligned Unimodal Models Retrieval")

    col_v, col_t, col_p = st.columns(3)

    # -----------------------------
    # Column 1 — Vision model
    # -----------------------------
    with col_v:
        vision_options = sorted(meta["vision_model"].unique())
        vision_model = st.selectbox("Vision model", vision_options)

        vision_info = {
            "ResNet50": {
                "type": "Convolutional Neural Network (CNN)",
                "desc": "ResNet‑50 pretrained on ImageNet‑1k at 224×224 resolution.",
                "hf": "https://huggingface.co/microsoft/resnet-50"
            },
            "MobileNetV3": {
                "type": "Lightweight CNN",
                "desc": "MobileNetV3‑Large pretrained on ImageNet‑1k at 224×224.",
                "hf": "https://huggingface.co/litert-community/MobileNet-v3-large"
            },
            "ViT": {
                "type": "Vision Transformer (ViT)",
                "desc": "ViT‑Base pretrained on ImageNet‑21k at 224×224.",
                "hf": "https://huggingface.co/google/vit-base-patch16-224"
            },
            "PVT": {
                "type": "Pyramid Vision Transformer",
                "desc": "PVT‑Tiny pretrained on ImageNet‑1k at 224×224.",
                "hf": "https://huggingface.co/Zetatech/pvt-tiny-224"
            }
        }

        if vision_model in vision_info:
            info = vision_info[vision_model]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(info["desc"])
            st.markdown(f"[HuggingFace page]({info['hf']})")

    # -----------------------------
    # Column 2 — Text model
    # -----------------------------
    with col_t:
        text_options = sorted(
            meta[meta["vision_model"] == vision_model]["text_model"].unique()
        )
        text_model = st.selectbox("Text model", text_options)

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

        if text_model in text_info:
            info = text_info[text_model]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(info["desc"])
            st.markdown(f"[HuggingFace page]({info['hf']})")

    # -----------------------------
    # Column 3 — Projection
    # -----------------------------
    with col_p:
        proj_options = sorted(
            meta[
                (meta["vision_model"] == vision_model) &
                (meta["text_model"] == text_model)
            ]["projection"].unique()
        )
        projection_name = st.selectbox("Projection", proj_options)

        proj_info = {
            "rp": "Random Gaussian projection used as a baseline.",
            "wcca": "Weighted CCA: maximizes correlation between modalities.",
            "cpca": "Cluster-Aware PCA: finds shared principal components."
        }

        if projection_name in proj_info:
            st.markdown(f"**Projection description:** {proj_info[projection_name]}")
    
    st.markdown(
        f"**Selected alignment:** `{vision_model}` × `{text_model}` — **{projection_name}**"
    )
        
    # Retrieve the matching row
    fusion_row = meta[
        (meta["vision_model"] == vision_model) &
        (meta["text_model"] == text_model) &
        (meta["projection"] == projection_name)
    ]

    if fusion_row.empty:
        st.error("❌ No matching alignment configuration found.")
        st.stop()

    fusion_row = fusion_row.iloc[0]

    # Load projection matrices
    Wv = np.load(fusion_row["Wv_path"])
    Wt = np.load(fusion_row["Wt_path"])

    # Load unimodal embeddings
    Xv = np.load(vision_emb_path("Flickr8k", vision_model))
    Xt = np.load(text_emb_path("Flickr8k", text_model))

    # Project into shared space
    Xv = Xv @ Wv
    Xt = Xt @ Wt

    query_type = st.radio(
        "Query Type",
        ["Image → Text", "Text → Image", "Multimodal → Text", "Multimodal → Image"]
    )


    # Reset state when query type changes
    if "last_query_type" not in st.session_state:
        st.session_state.last_query_type = query_type

    if st.session_state.last_query_type != query_type:
        st.session_state.selected_image = None
        st.session_state.selected_caption = None
        st.session_state.results = None
        st.session_state.last_query_type = query_type

    # ---------------- IMAGE → TEXT ----------------
    if query_type == "Image → Text":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)
        
        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            horizontal=True,
            key="image_mode"
        )
        # -----------------------------
        # NO IMAGE SELECTED YET
        # -----------------------------
        if st.session_state.selected_image is None:

            # ---------- UPLOAD MODE ----------
            if image_mode == "Upload image":
                uploaded_mm = st.file_uploader(
                    "Upload an image",
                    type=["jpg", "jpeg", "png"],
                    key="mm_upload"
                )

                if uploaded_mm is not None:
                    upload_dir = Path("data/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                    with open(save_path_mm, "wb") as f:
                        f.write(uploaded_mm.getbuffer())

                    st.session_state.uploaded_mm_image = str(save_path_mm)

                    img = Image.open(save_path_mm).convert("RGB")
                    st.image(img, caption="Uploaded Query Image", use_container_width=True)

                    if st.button("Use this image", key="use_uploaded_image"):
                        st.session_state.selected_image = st.session_state.uploaded_mm_image
                        st.session_state.results = None
                        st.rerun()

            # ---------- DATASET MODE ----------
            else:
                st.write("### Choose a query image from the dataset")

                # Scrollable container
                scroll_container = st.container(height=350)
                with scroll_container:
                    cols = st.columns(5)

                    for idx, path in enumerate(images[:50]):
                        img = Image.open(path).convert("RGB")
                        with cols[idx % 5]:
                            st.image(img, width=150)
                            if st.button(f"Select Image {idx+1}", key=f"{query_type}_img_{idx}"):
                                st.session_state.selected_image = path
                                st.session_state.results = None
                                st.rerun()

        # -----------------------------
        # IMAGE ALREADY SELECTED
        # -----------------------------
        else:
            query_path = st.session_state.selected_image
            query_img = Image.open(query_path).convert("RGB")

            st.write("### Selected Image")
            st.image(query_img, width=250)

            if st.button("🔄 Choose another image"):
                st.session_state.selected_image = None
                st.session_state.results = None
                st.rerun()

            if st.button("Retrieve I2T"):
                # Benchmark‑faithful: find index, use projected embeddings
                if query_path in images:
                    i = images.index(query_path)
                    xv = Xv[i]
                else:
                    xv = embed_image_query(query_path, vision_model).reshape(-1) @ Wv

                sims = cosine_similarity(xv.reshape(1,-1), Xt).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(captions_list[k], float(sims[k])) for k in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="fusion",
                    query_type="i2t",
                    vision_model=vision_model,
                    text_model=text_model,
                    projection=projection_name,
                    fusion_operator=None,
                    dataset="Flickr8k",
                    query=query_path,
                    results=st.session_state.results
                )

                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Captions")
            for cap, score in st.session_state.results:
                st.write(f"**{cap}** — {score:.4f}")


    # ---------------- TEXT → IMAGE ----------------
    elif query_type == "Text → Image":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)
        
        caption_mode = st.radio(
                    "Caption Input Mode",
                    ["Select from dataset", "Enter custom text"],
                    horizontal=True,
                    key="mm_caption_mode_t2i"
                )

        if st.session_state.selected_caption is None:

            if caption_mode == "Enter custom text":
                custom_text = st.text_area(
                    "Enter custom caption",
                    placeholder="Type any caption you want...",
                    key="mm_custom_caption"
                )

                if st.button("Use this caption", key=f"use_custom_{query_type}"):
                    if custom_text.strip():
                        st.session_state.selected_caption = custom_text.strip()
                        st.session_state.results = None
                        st.rerun()
                    else:
                        st.warning("Please enter a caption before continuing.")

            else:
                
                scroll_container = st.container(height=350)
                with scroll_container:
                    for idx, cap in enumerate(captions_list[:50]):
                        if st.button(cap, key=f"{query_type}_cap_{idx}"):
                            st.session_state.selected_caption = cap
                            st.session_state.results = None
                            st.rerun()


        else:
            st.write("### Selected Caption")
            st.info(st.session_state.selected_caption)
            

            if st.button("🔄 Choose another caption"):
                st.session_state.selected_caption = None
                st.session_state.results = None
                st.rerun()

            if st.button("Retrieve T2I"):
                # Benchmark‑faithful: find index, use projected embeddings
                if caption_mode == "Enter custom text":
                    # embed custom caption
                    xt = embed_text_query(st.session_state.selected_caption, text_model).reshape(-1)
                    xt = xt @ Wt   # project into shared space
                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = Xt[j]

                sims = cosine_similarity(xt.reshape(1,-1), Xv).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(images[k], float(sims[k])) for k in idx]
                
                save_retrieval(
                    user=username,
                    retrieval_type="fusion",
                    query_type="t2i",
                    vision_model=vision_model,
                    text_model=text_model,
                    projection=projection_name,
                    fusion_operator=None,
                    dataset="Flickr8k",
                    query=st.session_state.selected_caption,
                    results=st.session_state.results
                )
                
                st.rerun()

        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Images")
            cols = st.columns(5)
            for idx, (path, score) in enumerate(st.session_state.results):
                img = Image.open(images[idx]).convert("RGB")
                with cols[idx % 5]:
                    st.image(img, width=150, caption=f"{score:.4f}")

    # ---------------- MULTIMODAL → TEXT ----------------
    elif query_type == "Multimodal → Text":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)

        st.header("🟣 Multimodal → Text Retrieval")
        
        
        fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"])

        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        i, c, f1, f2 = st.columns([1, 1, 1, 3])
        with i: image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            key="mm_image_mode_m2t"
        )
        
        with c:caption_mode = st.radio(
            "Caption Input Mode",
            ["Select from dataset", "Enter custom text"],
            key="mm_caption_mode_m2t"
        )
        with f1:
            fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"], key = "m2t")
        with f2:
            info = fusion_info[fusion_method]
            st.markdown(f"**{info['name']}**")
            st.markdown(info["desc"])


        i, c = st.columns([1,1])
        with i:
            # -----------------------------
            # STEP 1 — SELECT IMAGE
            # -----------------------------
            if st.session_state.selected_image is None:

                # ---------- UPLOAD MODE ----------
                if image_mode == "Upload image":
                    uploaded_mm = st.file_uploader(
                        "Upload an image",
                        type=["jpg", "jpeg", "png"],
                        key="mm_upload_m2t"
                    )

                    if uploaded_mm is not None:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                        with open(save_path_mm, "wb") as f:
                            f.write(uploaded_mm.getbuffer())

                        st.session_state.uploaded_mm_image = str(save_path_mm)

                        img = Image.open(save_path_mm).convert("RGB")
                        st.image(img, caption="Uploaded Query Image", use_container_width=True)

                        if st.button("Use this image", key="use_uploaded_image_m2t"):
                            st.session_state.selected_image = st.session_state.uploaded_mm_image
                            st.session_state.results = None
                            st.rerun()

                # ---------- DATASET MODE ----------
                else:
                    st.write("### Choose a query image from the dataset")

                    # Scrollable container
                    scroll_container = st.container(height=350)
                    with scroll_container:
                        cols = st.columns(5)

                        for idx, path in enumerate(images[:50]):
                            img = Image.open(path).convert("RGB")
                            with cols[idx % 5]:
                                st.image(img, width=150)
                                if st.button(f"Select Image {idx+1}", key=f"m2t_img_{idx}"):
                                    st.session_state.selected_image = path
                                    st.session_state.results = None
                                    st.rerun()

            # -----------------------------
            # IMAGE ALREADY SELECTED
            # -----------------------------
            else:
                img_path = st.session_state.selected_image
                img = Image.open(img_path).convert("RGB")
                st.image(img)

                if st.button("🔄 Change Image", key="change_img_m2t"):
                    st.session_state.selected_image = None
                    st.session_state.selected_caption = None
                    st.session_state.results = None
                    st.rerun()

            with c:
                # -----------------------------
                # STEP 2 — SELECT CAPTION
                # -----------------------------
                if st.session_state.selected_caption is None:

                    if caption_mode == "Enter custom text":
                        custom_text = st.text_area(
                            "Enter custom caption",
                            placeholder="Type any caption you want...",
                            key="mm_custom_caption_m2t"
                        )

                        if st.button("Use this caption", key="use_custom_caption_m2t"):
                            if custom_text.strip():
                                st.session_state.selected_caption = custom_text.strip()
                                st.session_state.results = None
                                st.rerun()
                            else:
                                st.warning("Please enter a caption before continuing.")

                    else:
                        st.write("### Choose a caption")

                        scroll_container = st.container(height=350)
                        with scroll_container:

                            for idx, cap in enumerate(captions_list[:50]):
                                if st.button(cap, key=f"m2t_cap_{idx}"):
                                    st.session_state.selected_caption = cap
                                    st.session_state.results = None
                                    st.rerun()



                # -----------------------------
                # CAPTION ALREADY SELECTED
                # -----------------------------
                else:
                    st.info(st.session_state.selected_caption)

                    if st.button("🔄 Change Caption", key="change_caption_m2t"):
                        st.session_state.selected_caption = None
                        st.session_state.results = None
                        st.rerun()

            if st.button("Retrieve M2T", key="retrieve_m2t"):
                # Get image embedding
                if img_path in images:
                    i = images.index(img_path)
                    xv = Xv[i]
                else:
                    xv = embed_image_query(img_path, vision_model).reshape(-1) @ Wv

                # Get text embedding
                if caption_mode == "Enter custom text":
                    xt = embed_text_query(st.session_state.selected_caption, text_model).reshape(-1)
                    xt = xt @ Wt
                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = Xt[j]

                # Fuse
                F = fuse_embeddings(xv, xt, fusion_method)

                # Retrieve
                sims = cosine_similarity(F.reshape(1,-1), Xt).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(captions_list[j], float(sims[j])) for j in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="multimodal",
                    query_type="m2t",
                    vision_model=vision_model,
                    text_model=text_model,
                    projection=projection_name,
                    fusion_operator=fusion_method,
                    dataset="Flickr8k",
                    query=f"{img_path} + {st.session_state.selected_caption}",
                    results=st.session_state.results
                )

                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Captions (M2T)")
            for cap, score in st.session_state.results:
                st.write(f"**{cap}** — {score:.4f}")

    # ---------------- MULTIMODAL → IMAGE ----------------
    if query_type == "Multimodal → Image":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)

        st.header("🟢 Multimodal → Image Retrieval")
        fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"])

        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        i, c, f1, f2 = st.columns([1, 1, 1, 3])
        with i: image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            key="mm_image_mode_m2t"
        )
        
        with c:caption_mode = st.radio(
                    "Caption Input Mode",
                    ["Select from dataset", "Enter custom text"],
                    key="mm_caption_mode"
                )

        with f1:
            fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"], key = "m2i")
        with f2:
            info = fusion_info[fusion_method]
            st.markdown(f"**{info['name']}**")
            st.markdown(info["desc"])
            
        
        i,c = st.columns([1,1])
        
        with i:
            # -----------------------------
            # STEP 1 — SELECT IMAGE
            # -----------------------------
            if st.session_state.selected_image is None:

                # ---------- UPLOAD MODE ----------
                if image_mode == "Upload image":
                    uploaded_mm = st.file_uploader(
                        "Upload an image",
                        type=["jpg", "jpeg", "png"],
                        key="mm_upload_m2i"
                    )

                    if uploaded_mm is not None:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                        with open(save_path_mm, "wb") as f:
                            f.write(uploaded_mm.getbuffer())

                        st.session_state.uploaded_mm_image = str(save_path_mm)

                        img = Image.open(save_path_mm).convert("RGB")
                        st.image(img, caption="Uploaded Query Image", use_container_width=True)

                        if st.button("Use this image", key="use_uploaded_image_m2i"):
                            st.session_state.selected_image = st.session_state.uploaded_mm_image
                            st.session_state.results = None
                            st.rerun()

                # ---------- DATASET MODE ----------
                else:
                    st.write("### Choose a query image from the dataset")

                    # Scrollable container
                    scroll_container = st.container(height=350)
                    with scroll_container:
                        cols = st.columns(5)

                        for idx, path in enumerate(images[:50]):
                            img = Image.open(path).convert("RGB")
                            with cols[idx % 5]:
                                st.image(img, width=150)
                                if st.button(f"Select Image {idx+1}", key=f"m2i_img_{idx}"):
                                    st.session_state.selected_image = path
                                    st.session_state.results = None
                                    st.rerun()

            # -----------------------------
            # IMAGE ALREADY SELECTED
            # -----------------------------
            else:
                img_path = st.session_state.selected_image
                img = Image.open(img_path).convert("RGB")
                st.image(img, width=250)

                if st.button("🔄 Change Image", key="change_img_m2i"):
                    st.session_state.selected_image = None
                    st.session_state.selected_caption = None
                    st.session_state.results = None
                    st.rerun()
            
            with c:
                # -----------------------------
                # STEP 2 — SELECT CAPTION
                # -----------------------------
                if st.session_state.selected_caption is None:

                    if caption_mode == "Enter custom text":
                        custom_text = st.text_area(
                            "Enter custom caption",
                            placeholder="Type any caption you want...",
                            key="mm_custom_caption_m2i"
                        )

                        if st.button("Use this caption", key="use_custom_caption_m2i"):
                            if custom_text.strip():
                                st.session_state.selected_caption = custom_text.strip()
                                st.session_state.results = None
                                st.rerun()
                            else:
                                st.warning("Please enter a caption before continuing.")

                    else:
                        st.write("### Choose a caption")
                        scroll_container = st.container(height=350)
                        with scroll_container:

                            for idx, cap in enumerate(captions_list[:50]):
                                if st.button(cap, key=f"m2i_cap_{idx}"):
                                    st.session_state.selected_caption = cap
                                    st.session_state.results = None
                                    st.rerun()

                # -----------------------------
                # CAPTION ALREADY SELECTED
                # -----------------------------
                else:
                    st.info(st.session_state.selected_caption)

                    if st.button("🔄 Change Caption", key="change_caption_m2i"):
                        st.session_state.selected_caption = None
                        st.session_state.results = None
                        st.rerun()

            if st.button("Retrieve M2I", key="retrieve_m2i"):
                query_path = st.session_state.selected_image

                # Image embedding
                if query_path in images:
                    i = images.index(query_path)
                    xv = Xv[i]
                else:
                    xv = embed_image_query(query_path, vision_model).reshape(-1) @ Wv

                # Caption embedding
                if caption_mode == "Enter custom text":
                    xt = embed_text_query(st.session_state.selected_caption, text_model).reshape(-1)
                    xt = xt @ Wt
                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = Xt[j]

                # Fuse
                F = fuse_embeddings(xv, xt, fusion_method)

                # Retrieve
                sims = cosine_similarity(F.reshape(1,-1), Xv).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(images[j], float(sims[j])) for j in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="multimodal",
                    query_type="m2i",
                    vision_model=vision_model,
                    text_model=text_model,
                    projection=projection_name,
                    fusion_operator=fusion_method,
                    dataset="Flickr8k",
                    query=f"{img_path} + {st.session_state.selected_caption}",
                    results=st.session_state.results
                )

                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Images")
            cols = st.columns(5)
            for idx, (img_path, score) in enumerate(st.session_state.results):
                img = Image.open(img_path).convert("RGB")
                with cols[idx % 5]:
                    st.image(img, width=150, caption=f"{score:.4f}")


# ---------------------------------------------------------
# SOTA MODEL RETRIEVAL 
# ---------------------------------------------------------
else:
    st.header("🚀 SOTA Model Retrieval")
    l, r = st.columns([1,1])
    
    sota_info = {
        "CLIP ViT‑B/32": {
            "vision": "ViT‑B/32 Vision Transformer",
            "text": "Transformer text encoder (BERT‑like)",
            "desc": "CLIP is trained with **contrastive learning** on 400M image‑text pairs. "
                    "Images and texts are projected into a shared embedding space by maximizing "
                    "the similarity of matching pairs and minimizing mismatched ones.",
            "hf": "https://huggingface.co/openai/clip-vit-base-patch32"
        },
        "OpenCLIP ViT‑L/14": {
            "vision": "ViT‑L/14 Vision Transformer",
            "text": "Transformer text encoder",
            "desc": "OpenCLIP is an open reproduction of CLIP trained on LAION‑2B. "
                    "Alignment is achieved through large‑scale contrastive learning.",
            "hf": "https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
        }
    }

    

    with l: 
        sota_choice = st.selectbox("Choose SOTA Model", ["CLIP ViT‑B/32", "OpenCLIP ViT‑L/14"])
        info = sota_info[sota_choice]
        st.markdown(f"**Vision encoder:** {info['vision']}")
        st.markdown(f"**Text encoder:** {info['text']}")

    
    vision_embs, text_embs = load_sota_embeddings(sota_choice)

    with r:
        st.markdown(info["desc"])
        st.markdown(f"[HuggingFace page]({info['hf']})")
        
    

    sota_query_type = st.radio(
        "Query Type",
        ["Image → Text", "Text → Image", "Multimodal → Text", "Multimodal → Image"]
    )
    

    # ---------------- IMAGE → TEXT ----------------
    if sota_query_type == "Image → Text":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)

        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            horizontal=True,
            key="sota_image_mode"
        )

        # -----------------------------
        # NO IMAGE SELECTED YET
        # -----------------------------
        if st.session_state.selected_image is None:

            # ---------- UPLOAD MODE ----------
            if image_mode == "Upload image":
                uploaded_mm = st.file_uploader(
                    "Upload an image",
                    type=["jpg", "jpeg", "png"],
                    key="sota_mm_upload"
                )

                if uploaded_mm is not None:
                    upload_dir = Path("data/uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)

                    save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                    with open(save_path_mm, "wb") as f:
                        f.write(uploaded_mm.getbuffer())

                    st.session_state.uploaded_mm_image = str(save_path_mm)

                    img = Image.open(save_path_mm).convert("RGB")
                    st.image(img, caption="Uploaded Query Image", use_container_width=True)

                    if st.button("Use this image", key="sota_use_uploaded_image"):
                        st.session_state.selected_image = st.session_state.uploaded_mm_image
                        st.session_state.results = None
                        st.rerun()

            # ---------- DATASET MODE ----------
            else:
                st.write("### Choose a query image from the dataset")

                # Scrollable container
                scroll_container = st.container(height=350)
                with scroll_container:
                    cols = st.columns(5)

                    for idx, path in enumerate(images[:50]):
                        img = Image.open(path).convert("RGB")
                        with cols[idx % 5]:
                            st.image(img, width=150)
                            if st.button(f"Select Image {idx+1}", key=f"{sota_query_type}_img_{idx}"):
                                st.session_state.selected_image = path
                                st.session_state.results = None
                                st.rerun()

        # -----------------------------
        # IMAGE ALREADY SELECTED
        # -----------------------------
        else:
            query_path = st.session_state.selected_image
            query_img = Image.open(query_path).convert("RGB")

            st.write("### Selected Image")
            st.image(query_img, width=250)

            if st.button("🔄 Choose another image"):
                st.session_state.selected_image = None
                st.session_state.results = None
                st.rerun()

            if st.button("Retrieve I2T"):

                # 1. Get query embedding
                if query_path in images:
                    i = images.index(query_path)
                    qv = vision_embs[i].reshape(1, -1)
                else:
                    img = Image.open(query_path).convert("RGB")
                    if sota_choice == "CLIP ViT‑B/32":
                        qv = clip_encode_image(img).reshape(1, -1)
                    else:
                        qv = openclip_encode_image(img).reshape(1, -1)

                # 2. Compute similarity
                sims = cosine_similarity(qv, text_embs).flatten()
                idx = np.argsort(sims)[::-1][:k]

                # 3. Save results
                st.session_state.results = [(captions_list[j], float(sims[j])) for j in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="multimodal",
                    query_type="i2t",
                    vision_model=sota_choice,
                    text_model=sota_choice,
                    projection=sota_choice,
                    fusion_operator=None,
                    dataset="Flickr8k",
                    query=query_path,
                    results=st.session_state.results
                )
                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Captions")
            for cap, score in st.session_state.results:
                st.write(f"**{cap}** — {score:.4f}")


    # ---------------- TEXT → IMAGE ----------------
    elif sota_query_type == "Text → Image":
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)
        
        caption_mode = st.radio(
                    "Caption Input Mode",
                    ["Select from dataset", "Enter custom text"],
                    horizontal=True,
                    key="mm_caption_mode_sota_t2i"
                )
        if st.session_state.selected_caption is None:

            if caption_mode == "Enter custom text":
                custom_text = st.text_area(
                    "Enter custom caption",
                    placeholder="Type any caption you want...",
                    key="mm_custom_caption"
                )

                if st.button("Use this caption", key=f"use_custom_{"i2t"}"):
                    if custom_text.strip():
                        st.session_state.selected_caption = custom_text.strip()
                        st.session_state.results = None
                        st.rerun()
                    else:
                        st.warning("Please enter a caption before continuing.")

            else:
                for idx, cap in enumerate(captions_list[:50]):
                    if st.button(cap, key=f"{"i2t"}_cap_{idx}"):
                        st.session_state.selected_caption = cap
                        st.session_state.results = None
                        st.rerun()



        else:
            st.write("### Selected Caption")
            st.info(st.session_state.selected_caption)

            if st.button("🔄 Choose another caption"):
                st.session_state.selected_caption = None
                st.session_state.results = None
                st.rerun()

            if st.button("Retrieve T2I"):
                # Benchmark‑faithful: find index, use SOTA embeddings
                # Custom caption → encode with CLIP/OpenCLIP
                if caption_mode == "Enter custom text":
                    if sota_choice == "CLIP ViT‑B/32":
                        xt = clip_encode_text(st.session_state.selected_caption)
                    else:
                        xt = openclip_encode_text(st.session_state.selected_caption)

                # Dataset caption → lookup embedding
                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = text_embs[j:j+1]

                # Compute similarity
                sims = cosine_similarity(xt, vision_embs).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(images[k], float(sims[k])) for k in idx]
                
                save_retrieval(
                        user=username,
                        retrieval_type="multimodal",
                        query_type="m2t",
                        vision_model=sota_choice,
                        text_model=sota_choice,
                        projection=sota_choice,
                        fusion_operator=fusion_method,
                        dataset="Flickr8k",
                        query=st.session_state.selected_caption,
                        results=st.session_state.results
                    )
                
                st.rerun()

        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Images")
            cols = st.columns(5)
            for idx, (path, score) in enumerate(st.session_state.results):
                img = Image.open(images[idx]).convert("RGB")
                with cols[idx % 5]:
                    st.image(img, width=150, caption=f"{score:.4f}")

    # ---------------- MULTIMODAL → TEXT (SOTA) ----------------
    elif sota_query_type == "Multimodal → Text":
        st.header("🟣 SOTA Multimodal → Text Retrieval")
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)

        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        i, c, f1, f2 = st.columns([1, 1, 1, 3])
        with i: image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            key="sota_image_mode_m2t"
        )
        
        with c:caption_mode = st.radio(
                    "Caption Input Mode",
                    ["Select from dataset", "Enter custom text"],
                    key="sota_caption_mode_m2t"
                )

        with f1:
            fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"], key = "sota_m2t")
        with f2:
            info = fusion_info[fusion_method]
            st.markdown(f"**{info['name']}**")
            st.markdown(info["desc"])
            
        i,c = st.columns([1,1])
        
        with i:
            # -----------------------------
            # STEP 1 — SELECT IMAGE
            # -----------------------------
            if st.session_state.selected_image is None:

                # ---------- UPLOAD MODE ----------
                if image_mode == "Upload image":
                    uploaded_mm = st.file_uploader(
                        "Upload an image",
                        type=["jpg", "jpeg", "png"],
                        key="sota_mm_upload_m2t"
                    )

                    if uploaded_mm is not None:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                        with open(save_path_mm, "wb") as f:
                            f.write(uploaded_mm.getbuffer())

                        st.session_state.uploaded_mm_image = str(save_path_mm)

                        img = Image.open(save_path_mm).convert("RGB")
                        st.image(img, caption="Uploaded Query Image", use_container_width=True)

                        if st.button("Use this image", key="sota_use_uploaded_image_m2t"):
                            st.session_state.selected_image = st.session_state.uploaded_mm_image
                            st.session_state.results = None
                            st.rerun()

                # ---------- DATASET MODE ----------
                else:
                    st.write("### Choose a query image from the dataset")

                    # Scrollable container
                    scroll_container = st.container(height=350)
                    with scroll_container:
                        cols = st.columns(5)

                        for idx, path in enumerate(images[:50]):
                            img = Image.open(path).convert("RGB")
                            with cols[idx % 5]:
                                st.image(img, width=150)
                                if st.button(f"Select Image {idx+1}", key=f"sota_m2t_img_{idx}"):
                                    st.session_state.selected_image = path
                                    st.session_state.results = None
                                    st.rerun()

            # -----------------------------
            # IMAGE ALREADY SELECTED
            # -----------------------------
            else:
                img_path = st.session_state.selected_image
                img = Image.open(img_path).convert("RGB")
                st.image(img, width=250)

                if st.button("🔄 Change Image", key="sota_change_img_m2t"):
                    st.session_state.selected_image = None
                    st.session_state.selected_caption = None
                    st.session_state.results = None
                    st.rerun()

                query_path = st.session_state.selected_image

            with c:
                # -----------------------------
                # STEP 2 — SELECT CAPTION
                # -----------------------------
                if st.session_state.selected_caption is None:

                    if caption_mode == "Enter custom text":
                        custom_text = st.text_area(
                            "Enter custom caption",
                            placeholder="Type any caption you want...",
                            key="sota_custom_caption_m2t"
                        )

                        if st.button("Use this caption", key="sota_use_custom_caption_m2t"):
                            if custom_text.strip():
                                st.session_state.selected_caption = custom_text.strip()
                                st.session_state.results = None
                                st.rerun()
                            else:
                                st.warning("Please enter a caption before continuing.")

                    else:
                        st.write("### Choose a caption")

                        scroll_container = st.container(height=350)
                        with scroll_container:

                            for idx, cap in enumerate(captions_list[:50]):
                                if st.button(cap, key=f"sota_m2t_cap_{idx}"):
                                    st.session_state.selected_caption = cap
                                    st.session_state.results = None
                                    st.rerun()

                # -----------------------------
                # CAPTION ALREADY SELECTED
                # -----------------------------
                else:
                    st.info(st.session_state.selected_caption)

                    if st.button("🔄 Change Caption", key="sota_change_caption_m2t"):
                        st.session_state.selected_caption = None
                        st.session_state.results = None
                        st.rerun()

            if st.button("Retrieve M2T", key="sota_retrieve_m2t"):

                # Image embedding
                if query_path in images:
                    i = images.index(query_path)
                    xv = vision_embs[i]
                else:
                    img = Image.open(query_path).convert("RGB")
                    if sota_choice == "CLIP ViT‑B/32":
                        xv = clip_encode_image(img).reshape(-1)
                    else:
                        xv = openclip_encode_image(img).reshape(-1)

                # Caption embedding
                if caption_mode == "Enter custom text":
                    if sota_choice == "CLIP ViT‑B/32":
                        xt = clip_encode_text(st.session_state.selected_caption)
                    else:
                        xt = openclip_encode_text(st.session_state.selected_caption)

                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = text_embs[j]

                # Fuse
                F = fuse_embeddings(xv, xt, fusion_method)

                # Retrieve
                sims = cosine_similarity(F.reshape(1, -1), text_embs).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(captions_list[j], float(sims[j])) for j in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="multimodal",
                    query_type="m2t",
                    vision_model=sota_choice,
                    text_model=sota_choice,
                    projection=sota_choice,
                    fusion_operator=fusion_method,
                    dataset="Flickr8k",
                    query=f"{img_path} + {st.session_state.selected_caption}",
                    results=st.session_state.results
                )

                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Captions (SOTA M2T)")
            for cap, score in st.session_state.results:
                st.write(f"**{cap}** — {score:.4f}")


    # ---------------- MULTIMODAL → IMAGE (SOTA) ----------------
    elif sota_query_type == "Multimodal → Image":
        st.header("🟢 SOTA Multimodal → Image Retrieval")
        k = st.slider("Number of results to display (K)", min_value=5, max_value=50, value=20, step=5)

        # -----------------------------
        # IMAGE INPUT MODE
        # -----------------------------
        i, c, f1, f2 = st.columns([1, 1, 1, 3])
        with i: image_mode = st.radio(
            "Image Input Mode",
            ["Select from dataset", "Upload image"],
            key="sota_image_mode_m2i"
        )
        
        with c:caption_mode = st.radio(
                    "Caption Input Mode",
                    ["Select from dataset", "Enter custom text"],
                    key="sota_caption_mode_m2i"
                )
        
        with f1:
            fusion_method = st.selectbox("Fusion Operator", ["add", "gated", "mul"], key="sota_m2i")
        with f2:
            info = fusion_info[fusion_method]
            st.markdown(f"**{info['name']}**")
            st.markdown(info["desc"])

        i,c = st.columns([1,1])

        with i:
            # -----------------------------
            # STEP 1 — SELECT IMAGE
            # -----------------------------
            if st.session_state.selected_image is None:

                # ---------- UPLOAD MODE ----------
                if image_mode == "Upload image":
                    uploaded_mm = st.file_uploader(
                        "Upload an image",
                        type=["jpg", "jpeg", "png"],
                        key="sota_mm_upload_m2i"
                    )

                    if uploaded_mm is not None:
                        upload_dir = Path("data/uploads")
                        upload_dir.mkdir(parents=True, exist_ok=True)

                        save_path_mm = upload_dir / f"{username}_{int(time.time())}.jpg"
                        with open(save_path_mm, "wb") as f:
                            f.write(uploaded_mm.getbuffer())

                        st.session_state.uploaded_mm_image = str(save_path_mm)

                        img = Image.open(save_path_mm).convert("RGB")
                        st.image(img, caption="Uploaded Query Image", use_container_width=True)

                        if st.button("Use this image", key="sota_use_uploaded_image_m2i"):
                            st.session_state.selected_image = st.session_state.uploaded_mm_image
                            st.session_state.results = None
                            st.rerun()

                # ---------- DATASET MODE ----------
                else:
                    st.write("### Choose a query image from the dataset")

                    # Scrollable container
                    scroll_container = st.container(height=350)
                    with scroll_container:

                        cols = st.columns(5)

                        for idx, path in enumerate(images[:50]):
                            img = Image.open(path).convert("RGB")
                            with cols[idx % 5]:
                                st.image(img, width=150)
                                if st.button(f"Select Image {idx+1}", key=f"sota_m2i_img_{idx}"):
                                    st.session_state.selected_image = path
                                    st.session_state.results = None
                                    st.rerun()

            # -----------------------------
            # IMAGE ALREADY SELECTED
            # -----------------------------
            else:
                img_path = st.session_state.selected_image
                img = Image.open(img_path).convert("RGB")
                st.image(img, width=250)

                if st.button("🔄 Change Image", key="sota_change_img_m2i"):
                    st.session_state.selected_image = None
                    st.session_state.selected_caption = None
                    st.session_state.results = None
                    st.rerun()

                query_path = st.session_state.selected_image

            with c:
                # -----------------------------
                # STEP 2 — SELECT CAPTION
                # -----------------------------
                if st.session_state.selected_caption is None:

                    if caption_mode == "Enter custom text":
                        custom_text = st.text_area(
                            "Enter custom caption",
                            placeholder="Type any caption you want...",
                            key="sota_custom_caption_m2i"
                        )

                        if st.button("Use this caption", key="sota_use_custom_caption_m2i"):
                            if custom_text.strip():
                                st.session_state.selected_caption = custom_text.strip()
                                st.session_state.results = None
                                st.rerun()
                            else:
                                st.warning("Please enter a caption before continuing.")

                    else:
                        st.write("### Choose a caption")

                        # Scrollable container
                        scroll_container = st.container(height=350)
                        with scroll_container:
                            for idx, cap in enumerate(captions_list[:50]):
                                if st.button(cap, key=f"sota_m2i_cap_{idx}"):
                                    st.session_state.selected_caption = cap
                                    st.session_state.results = None
                                    st.rerun()

                # -----------------------------
                # CAPTION ALREADY SELECTED
                # -----------------------------
                else:
                    st.info(st.session_state.selected_caption)

                    if st.button("🔄 Change Caption", key="sota_change_caption_m2i"):
                        st.session_state.selected_caption = None
                        st.session_state.results = None
                        st.rerun()

            if st.button("Retrieve M2I", key="sota_retrieve_m2i"):

                # Image embedding
                if query_path in images:
                    i = images.index(query_path)
                    xv = vision_embs[i]
                else:
                    img = Image.open(query_path).convert("RGB")
                    if sota_choice == "CLIP ViT‑B/32":
                        xv = clip_encode_image(img).reshape(-1)
                    else:
                        xv = openclip_encode_image(img).reshape(-1)

                # Caption embedding
                if caption_mode == "Enter custom text":
                    j = None
                    if sota_choice == "CLIP ViT‑B/32":
                        xt = clip_encode_text(st.session_state.selected_caption)
                    else:
                        xt = openclip_encode_text(st.session_state.selected_caption)

                else:
                    j = captions_list.index(st.session_state.selected_caption)
                    xt = text_embs[j]

                # Fuse
                F = fuse_embeddings(xv, xt, fusion_method)

                # Retrieve
                sims = cosine_similarity(F.reshape(1, -1), vision_embs).flatten()
                idx = np.argsort(sims)[::-1][:k]

                st.session_state.results = [(images[j], float(sims[j])) for j in idx]

                save_retrieval(
                    user=username,
                    retrieval_type="multimodal",
                    query_type="m2i",
                    vision_model=sota_choice,
                    text_model=sota_choice,
                    projection=sota_choice,
                    fusion_operator=fusion_method,
                    dataset="Flickr8k",
                    query=f"{img_path} + {st.session_state.selected_caption}",
                    results=st.session_state.results
                )

                st.rerun()

        # -----------------------------
        # DISPLAY RESULTS
        # -----------------------------
        if st.session_state.results is not None:
            st.subheader(f"Top‑{k} Retrieved Images (SOTA M2I)")
            cols = st.columns(5)
            for idx, (path, score) in enumerate(st.session_state.results):
                img = Image.open(path).convert("RGB")
                with cols[idx % 5]:
                    st.image(img, width=150, caption=f"{score:.4f}")

app_footer()
