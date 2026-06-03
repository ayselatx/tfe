import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.ui import fixed_image, section_title

from utils.auth import check_login_status, init_db
from utils.retrieval_multimodal import (
    clip_encode_image, clip_encode_text,
    openclip_encode_image, openclip_encode_text
)
from utils.loaders import load_sota_embeddings, DATASETS_DIR

init_db()

from utils.layout import app_header, app_footer
app_header()

if not check_login_status():
    st.switch_page("pages/login.py")

st.set_page_config(page_title="SOTA Retrieval", page_icon="🚀", layout="wide")
st.title("🚀 State‑of‑the‑Art Retrieval")

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
with st.sidebar:
    st.page_link("pages/my_account.py", label="My Account", icon="👤")
    st.page_link("pages/unimodal.py", label="Unimodal Benchmarking", icon="📚")
    st.page_link("pages/multimodal_sota.py", label="SOTA Retrieval", icon="🚀")
    st.page_link("pages/fusion_results.py", label="Fusion Results", icon="🔀")
    st.page_link("pages/multimodal_evaluation.py", label="Multimodal Evaluation", icon="📊")
    st.page_link("pages/rag.py", label="RAG Ameliorations", icon="✨")

# Load Flickr8k dataset
df = pd.read_pickle(f"{DATASETS_DIR}/df_Flickr8k.pkl")
images = df["image_path"].tolist()
captions = [cap for caps in df["captions"].tolist() for cap in caps]

# Load SOTA embeddings
model_choice = st.selectbox("Choose SOTA Model", ["CLIP ViT‑B/32", "OpenCLIP ViT‑L/14"])
vision_embs, text_embs = load_sota_embeddings(
    "clip" if "CLIP" in model_choice else "openclip_l14"
)

# Session state
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "selected_caption" not in st.session_state:
    st.session_state.selected_caption = None
if "results" not in st.session_state:
    st.session_state.results = None

tab1, tab2 = st.tabs(["Image → Text", "Text → Image"])

# ---------------------------------------------------------
# IMAGE → TEXT RETRIEVAL
# ---------------------------------------------------------
with tab1:
    st.header("Image → Text Retrieval")

    if st.session_state.selected_image is None:
        st.write("### Choose a query image")

        cols = st.columns(5)
        for idx, path in enumerate(images[:50]):
            img = Image.open(path).convert("RGB")
            with cols[idx % 5]:
                st.image(img, width=150)
                if st.button(f"Select {idx+1}", key=f"i2t_{idx}"):
                    st.session_state.selected_image = path
                    st.session_state.results = None
                    st.rerun()

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
            q = (
                clip_encode_image(query_img)
                if "CLIP" in model_choice
                else openclip_encode_image(query_img)
            )
            sims = cosine_similarity(q, text_embs).flatten()
            idx = np.argsort(sims)[::-1][:20]

            st.session_state.results = [(captions[i], float(sims[i])) for i in idx]
            st.rerun()

    if st.session_state.results is not None:
        st.subheader("Top‑20 Retrieved Captions")
        for cap, score in st.session_state.results:
            st.write(f"**{cap}** — {score:.4f}")

# ---------------------------------------------------------
# TEXT → IMAGE RETRIEVAL
# ---------------------------------------------------------
with tab2:
    st.header("Text → Image Retrieval")

    if st.session_state.selected_caption is None:
        st.write("### Choose a query caption")

        for idx, cap in enumerate(captions[:50]):
            if st.button(cap, key=f"t2i_{idx}"):
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
            q = (
                clip_encode_text(st.session_state.selected_caption)
                if "CLIP" in model_choice
                else openclip_encode_text(st.session_state.selected_caption)
            )
            sims = cosine_similarity(q, vision_embs).flatten()
            idx = np.argsort(sims)[::-1][:20]

            st.session_state.results = [(images[i], float(sims[i])) for i in idx]
            st.rerun()

    if st.session_state.results is not None:
        st.subheader("Top‑20 Retrieved Images")
        cols = st.columns(5)
        for idx, (path, score) in enumerate(st.session_state.results):
            img = Image.open(path).convert("RGB")
            with cols[idx % 5]:
                st.image(img, width=150, caption=f"{score:.4f}")

app_footer()
