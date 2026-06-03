import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils.auth import check_login_status, init_db
from utils.layout import app_header, app_footer
from utils.loaders import load_fusion_metadata
from utils.sql_fusion import (
    init_fusion_tables,
    load_fusion_clusters,
    save_fusion_clusters,
)
from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    vision_xai_dir,
    text_xai_dir,
)


# ---------------------------------------------------------
# PAGE CONFIG — MUST BE FIRST
# ---------------------------------------------------------
st.set_page_config(page_title="Alignment Results", page_icon="🔀", layout="wide")
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


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------
init_db()
init_fusion_tables()
if not check_login_status():
    st.switch_page("pages/login.py")

username = st.session_state.get("username", "anonymous")

# ---------------------------------------------------------
# DESCRIPTION
# ---------------------------------------------------------
st.title("🔀 Alignment of Unimodal Model Embeddings")


st.info(
    """
This page explores **aligning of unimodal vision and text encoder embeddings** into a shared space.

You can:
- select a **vision model**, **text model**, and **projection type**
- cluster fused image embeddings
- inspect **cluster assignments**, **cluster similarity heatmaps**, and **captions**
- export **CSV files** for cluster labels and similarity matrices
- benefit from **SQL caching** so repeated configurations load instantly
"""
)

tab_results, tab_fusion = st.tabs(["⚙️ Align", "📊 Alignment Results"])

# ---------------------------------------------------------
# LOAD METADATA
# ---------------------------------------------------------
meta = load_fusion_metadata()

from utils.paths import (
    IMAGE_PATHS,
    FLICKR8K_CAPTIONS,
    vision_emb_path,
    text_emb_path,
    STREAMLIT_DATA,
    STREAMLIT_DATA_UNIMODAL,
    STREAMLIT_DATA_VISION_DIR,
    STREAMLIT_DATA_TEXT_DIR,
    STREAMLIT_DATASETS_DIR,
)

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
df = pd.read_pickle(os.path.join(STREAMLIT_DATASETS_DIR, "df_Flickr8k.pkl"))
IMAGE_PATHS = df["image_path"].tolist()  # columns: image_name, image_path, captions


def get_captions(img_path: str):
    img_name = os.path.basename(img_path)
    row = df[df["image_name"] == img_name]
    if len(row) == 0:
        return []
    return row["captions"].iloc[0]


meta_no_random = meta[~meta["projection"].str.contains("random", case=False, na=False)]
best = meta_no_random.groupby(["vision_model", "text_model"]).first()


# ---------------------------------------------------------
# CLUSTERING UTILS
# ---------------------------------------------------------
def cluster_embeddings(X, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def cluster_centroids(X, labels):
    centroids = []
    for c in np.unique(labels):
        centroids.append(X[labels == c].mean(axis=0))
    return np.vstack(centroids)


def plot_cluster_heatmap(dists):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(dists, cmap="viridis")
    ax.set_title("Cluster Similarity Heatmap")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    fig.colorbar(cax)
    return fig

with tab_results:
    st.header("⚙️ Select Alignment Configuration")

    st.markdown("""
    ### 🔧 Alignment Configuration
    Configure your alignment pipeline by selecting a vision encoder, a text encoder, 
    and a projection method.  
    This tab handles **model selection**, **projection loading**, **clustering**, 
    and **CSV export**.  
    All computations are cached in SQL for fast re‑loading.
    """)
    

    col_v, col_t, col_p = st.columns(3)

    # -----------------------------
    # Column 1 — Vision model
    # -----------------------------
    with col_v:
        vision_options = sorted(meta["vision_model"].unique())
        vision_model = st.selectbox("Vision model", vision_options)

        # Vision model description
        vision_info = {
            "ResNet50": {
                "type": "Convolutional Neural Network (CNN)",
                "desc": "ResNet model pre-trained on ImageNet-1k at resolution 224x224.",
                "hf": "https://huggingface.co/microsoft/resnet-50"
            },
            "MobileNetV3": {
                "type": "Lightweight CNN",
                "desc": "MobileNet V3 Large model pre-trained on ImageNet-1k at resolution 224x224.",
                "hf": "https://huggingface.co/litert-community/MobileNet-v3-large"
            },
            "ViT": {
                "type": "Vision Transformer (ViT)",
                "desc": "The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. ",
                "hf": "https://huggingface.co/google/vit-base-patch16-224"
            },
            "PVT": {
                "type": "Pyramid Vision Transformer",
                "desc": "The Pyramid Vision Transformer (PVT) is a transformer encoder model (BERT-like) pretrained on ImageNet-1k (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.",
                "hf": "https://huggingface.co/Zetatech/pvt-tiny-224"
            }
        }

        if vision_model in vision_info:
            info = vision_info[vision_model]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"{info['desc']}")
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
                "desc": "BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.",
                "hf": "https://huggingface.co/bert-base-uncased"
            },
            "RoBERTa": {
                "type": "Masked Language Model (MLM)",
                "desc": "RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion.",
                "hf": "https://huggingface.co/roberta-base"
            },
            "GPT2": {
                "type": "Autoregressive Transformer",
                "desc": "GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion.",
                "hf": "https://huggingface.co/gpt2"
            }
        }

        if text_model in text_info:
            info = text_info[text_model]
            st.markdown(f"**Type:** {info['type']}")
            st.markdown(f"{info['desc']}")
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
            "rp": "A random Gaussian projection used as a baseline for comparison.",
            "wcca": "Canonical Correlation Analysis finds projections maximizing correlation between modalities.",
            "cpca": "Principal Components Analysis finds the components of highest variance."
        }

        if projection_name in proj_info:
            st.markdown(f"**Projection description:** {proj_info[projection_name]}")

    # Retrieve the matching row
    fusion_row = meta[
        (meta["vision_model"] == vision_model) &
        (meta["text_model"] == text_model) &
        (meta["projection"] == projection_name)
    ].iloc[0]

    st.markdown(
        f"**Selected Alignment:** `{vision_model}` × `{text_model}` — **{projection_name}**"
    )


    # ---------------------------------------------------------
    # LOAD PROJECTION MATRICES
    # ---------------------------------------------------------
    Wv = np.load(fusion_row["Wv_path"])
    Wt = np.load(fusion_row["Wt_path"])

    # ---------------------------------------------------------
    # LOAD UNIMODAL EMBEDDINGS
    # ---------------------------------------------------------
    Xv = np.load(vision_emb_path("Flickr8k", vision_model))
    Xt = np.load(text_emb_path("Flickr8k", text_model))

    # ---------------------------------------------------------
    # PROJECT INTO SHARED SPACE
    # ---------------------------------------------------------
    Xv_proj = Xv @ Wv
    Xt_proj = Xt @ Wt
    

    
    
    # ---------------------------------------------------------
    # CLUSTER NEIGHBOR ANALYSIS
    # ---------------------------------------------------------
    st.header("🔍 Cluster Neighbor Analysis")

    cluster_mode = st.selectbox(
        "Clustering mode",
        ["Images only", "Captions only", "Multimodal (Images + Captions)"],
        index=0
    )

    max_images = len(Xv_proj)
    num_images = st.number_input(
        "Number of items to analyze",
        min_value=50,
        max_value=max_images,
        value=50,
        step=10,
    )

    # -----------------------------
    # Select data based on mode
    # -----------------------------
    if cluster_mode == "Images only":
        X_cluster = Xv_proj[:num_images]
        cluster_type = "images"

    elif cluster_mode == "Captions only":
        X_cluster = Xt_proj[:num_images]
        cluster_type = "captions"

    else:  # Multimodal
        X_cluster = np.vstack([
            Xv_proj[:num_images],
            Xt_proj[:num_images]
        ])
        cluster_type = "multimodal"

    # -----------------------------
    # Number of clusters
    # -----------------------------
    n_clusters = st.number_input(
        "Select number of clusters",
        min_value=5,
        max_value=20,
        value=5,
    )

    # -----------------------------
    # SQL Caching
    # -----------------------------
    cached = load_fusion_clusters(
        vision_model=vision_model,
        text_model=text_model,
        projection=projection_name,
        num_images=num_images,
        n_clusters=n_clusters,
        mode=cluster_type,   # NEW: include mode in cache key
    )

    if cached is not None:
        labels_proj, centroids = cached
        st.success("Loaded clustering from SQL cache ✅")
    else:
        labels_proj, km_proj = cluster_embeddings(X_cluster, n_clusters=n_clusters)
        centroids = cluster_centroids(X_cluster, labels_proj)

        save_fusion_clusters(
            username=username,
            vision_model=vision_model,
            text_model=text_model,
            projection=projection_name,
            num_images=num_images,
            n_clusters=n_clusters,
            labels=labels_proj,
            centroids=centroids,
            mode=cluster_type,   # NEW
        )
        st.info("Computed clustering and saved to SQL cache ✅")

    # ---------------------------------------------------------
    # SORTING + DISPLAY
    # ---------------------------------------------------------
    sort_mode = st.radio(
        "Sort displayed items by:",
        ["Index", "Cluster index"],
        horizontal=True,
    )

    # Determine how many items exist in this mode
    if cluster_mode == "Multimodal (Images + Captions)":
        total_items = num_images * 2
    else:
        total_items = num_images

    max_display = min(50, total_items)

    if sort_mode == "Index":
        if cluster_mode == "Multimodal (Images + Captions)":
            interleaved = []
            for i in range(num_images):
                interleaved.append(i)               # image i
                interleaved.append(i + num_images)  # caption i
            display_indices = interleaved[:max_display]
        else:
            display_indices = list(range(max_display))
    else:
        pairs = [(labels_proj[i], i) for i in range(total_items)]
        pairs_sorted = sorted(pairs, key=lambda x: (x[0], x[1]))
        display_indices = [idx for (_, idx) in pairs_sorted[:max_display]]

    
    # ---------------------------------------------------------
    # DISPLAY BLOCK (Images only, Captions only, Multimodal)
    # ---------------------------------------------------------

    st.subheader(f"Cluster Labels (showing {max_display} of {total_items} items)")

    # =========================================================
    # MODE 1 — IMAGES ONLY
    # =========================================================
    if cluster_mode == "Images only":

        scroll = st.container(height=450)
        with scroll:
            cols = st.columns(10)

            for i, idx in enumerate(display_indices):
                lab = labels_proj[idx]

                with cols[i % 10]:
                    st.image(IMAGE_PATHS[idx], use_container_width=True)
                    st.markdown(
                        f"""
                        <div style='
                            text-align:center;
                            background:#eef;
                            border-radius:6px;
                            padding:4px;
                            font-size:13px;
                        '>
                            <b>Cluster {lab}</b><br/>
                            <span style='font-size:11px;'>Image {idx}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # =========================================================
    # MODE 2 — CAPTIONS ONLY
    # =========================================================
    elif cluster_mode == "Captions only":

        scroll = st.container(height=450)
        with scroll:

            for idx in display_indices:
                lab = labels_proj[idx]

                caption_list = get_captions(IMAGE_PATHS[idx])
                caption_text = caption_list[0] if caption_list else "(no caption found)"

                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        margin:8px 0;
                        background:#f9f9f9;
                        border-radius:6px;
                        border-left:4px solid #999;
                    ">
                        <b>Cluster {lab}</b><br/>
                        <span style='font-size:11px;'>Caption {idx}</span>
                        <br/><br/>
                        {caption_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


    # =========================================================
    # MODE 3 — MULTIMODAL (Images + Captions)
    # =========================================================
    else:

        # Split global sorted indices into two independent streams
        vision_indices = [idx for idx in display_indices if idx < num_images]
        text_indices   = [idx - num_images for idx in display_indices if idx >= num_images]

        # Two columns OUTSIDE scroll containers
        img_col, cap_col = st.columns([1, 1])

        # -----------------------------
        # LEFT COLUMN → IMAGES
        # -----------------------------
        with img_col:
            st.markdown("### Images")

            img_scroll = st.container(height=450)
            with img_scroll:
                img_cols = st.columns(5)

                for i, img_idx in enumerate(vision_indices):
                    lab = labels_proj[img_idx]

                    with img_cols[i % 5]:
                        st.image(IMAGE_PATHS[img_idx], use_container_width=True)
                        st.markdown(
                            f"""
                            <div style='
                                text-align:center;
                                background:#eef;
                                border-radius:6px;
                                padding:4px;
                                font-size:13px;
                            '>
                                <b>Cluster {lab}</b><br/>
                                <span style='font-size:11px;'>Image {img_idx}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        # -----------------------------
        # RIGHT COLUMN → CAPTIONS
        # -----------------------------
        with cap_col:
            st.markdown("### Captions")

            cap_scroll = st.container(height=450)
            with cap_scroll:

                for cap_idx in text_indices:
                    lab = labels_proj[cap_idx + num_images]

                    caption_list = get_captions(IMAGE_PATHS[cap_idx])
                    caption_text = caption_list[0] if caption_list else "(no caption found)"

                    st.markdown(
                        f"""
                        <div style="
                            padding:10px;
                            margin:8px 0;
                            background:#f9f9f9;
                            border-radius:6px;
                            border-left:4px solid #999;
                        ">
                            <b>Cluster {lab}</b><br/>
                            <span style='font-size:11px;'>Caption {cap_idx}</span>
                            <br/><br/>
                            {caption_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            
    # ---------------------------------------------------------
    # CLUSTER SUMMARY + CSV EXPORT
    # ---------------------------------------------------------
    st.subheader("📊 Cluster Summary")

    unique_clusters, counts = np.unique(labels_proj, return_counts=True)

    summary_df = pd.DataFrame({
        "cluster_index": unique_clusters,
        "count": counts,
    })

    st.table(summary_df.T)

    # -----------------------------
    # Build assignment table
    # -----------------------------
    if cluster_mode == "Images only":
        assign_df = pd.DataFrame({
            "type": ["image"] * num_images,
            "index": list(range(num_images)),
            "path": IMAGE_PATHS[:num_images],
            "cluster": labels_proj,
        })

    elif cluster_mode == "Captions only":
        assign_df = pd.DataFrame({
            "type": ["caption"] * num_images,
            "index": list(range(num_images)),
            "path": IMAGE_PATHS[:num_images],
            "cluster": labels_proj,
        })

    else:  # MULTIMODAL
        assign_df = pd.DataFrame({
            "type": ["image"] * num_images + ["caption"] * num_images,
            "index": list(range(num_images)) + list(range(num_images)),
            "path": IMAGE_PATHS[:num_images] + IMAGE_PATHS[:num_images],
            "cluster": labels_proj,
        })

    csv_assign = assign_df.to_csv(index=False)
    st.download_button(
        "Download Cluster Assignments CSV",
        csv_assign,
        f"alignment_{vision_model}_{text_model}_{projection_name}_clusters.csv",
        "text/csv",
    )

    # ---------------------------------------------------------
    # PREVIEW CLUSTER
    # ---------------------------------------------------------
    st.subheader("🔎 Preview Cluster")
    preview_scroll = st.container(height=450)
    with preview_scroll:
        selected_cluster = st.selectbox(
            "Select cluster to view",
            unique_clusters,
            index=0,
        )

        cluster_indices = [i for i, lab in enumerate(labels_proj) if lab == selected_cluster]
        total = len(cluster_indices)

        if total == 0:
            st.warning("No items in this cluster.")
        else:

            # =====================================================
            # IMAGES ONLY
            # =====================================================
            if cluster_mode == "Images only":

                st.subheader("Images")

                key_center = f"carousel_pos_cluster_{selected_cluster}"
                if key_center not in st.session_state:
                    st.session_state[key_center] = 0

                center_pos = st.session_state[key_center]

                col_prev, col_info, col_next = st.columns([1, 3, 1])

                if total > 5:
                    with col_prev:
                        if st.button("◀ Previous"):
                            st.session_state[key_center] = (center_pos - 5) % total
                            center_pos = st.session_state[key_center]

                    with col_next:
                        if st.button("Next ▶"):
                            st.session_state[key_center] = (center_pos + 5) % total
                            center_pos = st.session_state[key_center]

                with col_info:
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"Cluster <b>{selected_cluster}</b> — {total} images"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                window = [(center_pos + offset) % total for offset in range(5)]
                window = [cluster_indices[pos] for pos in window]

                cols = st.columns(min(5, total))
                for col, img_idx in zip(cols, window):
                    img_path = IMAGE_PATHS[img_idx]
                    with col:
                        st.image(img_path, use_container_width=True)
                        caps = get_captions(img_path)
                        st.markdown("**Captions:**")
                        for cap in caps:
                            st.markdown(f"- {cap}")
                        st.caption(f"Image index: {img_idx}")

            # =====================================================
            # CAPTIONS ONLY
            # =====================================================
            elif cluster_mode == "Captions only":

                st.subheader("Captions")

                for idx in cluster_indices:
                    caption_list = get_captions(IMAGE_PATHS[idx])
                    caption_text = caption_list[0] if caption_list else "(no caption found)"

                    st.markdown(
                        f"""
                        <div style="
                            padding:10px;
                            margin:8px 0;
                            background:#f9f9f9;
                            border-radius:6px;
                            border-left:4px solid #999;
                        ">
                            <b>Caption {idx}</b><br/><br/>
                            {caption_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # =====================================================
            # MULTIMODAL
            # =====================================================
            else:

                st.subheader("Images & Captions")

                img_col, cap_col = st.columns([1, 1])

                # Left: images
                with img_col:
                    st.markdown("### Images")
                    img_indices = [i for i in cluster_indices if i < num_images]
                    for img_idx in img_indices:
                        st.image(IMAGE_PATHS[img_idx], use_container_width=True)

                # Right: captions
                with cap_col:
                    st.markdown("### Captions")
                    cap_indices = [i - num_images for i in cluster_indices if i >= num_images]
                    for cap_idx in cap_indices:
                        caption_list = get_captions(IMAGE_PATHS[cap_idx])
                        caption_text = caption_list[0] if caption_list else "(no caption found)"
                        st.markdown(
                            f"""
                            <div style="
                                padding:10px;
                                margin:8px 0;
                                background:#f9f9f9;
                                border-radius:6px;
                                border-left:4px solid #999;
                            ">
                                <b>Caption {cap_idx}</b><br/><br/>
                                {caption_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
    # ---------------------------------------------------------
    # NEIGHBOR ANALYSIS + HEATMAP + CSV
    # ---------------------------------------------------------
    st.subheader("📌 Cluster Similarity & Neighbors")

    dists = cosine_similarity(centroids)

    fig_heat = plot_cluster_heatmap(dists)
    _, m, _ = st.columns([1, 3, 1])
    with m:st.pyplot(fig_heat)

    dists_df = pd.DataFrame(dists)
    csv_dists = dists_df.to_csv(index=False)
    st.download_button(
        "Download Cluster Similarity CSV",
        csv_dists,
        f"alignment_{vision_model}_{text_model}_{projection_name}_cluster_similarity.csv",
        "text/csv",
    )

    k = st.number_input(
        "Select cluster to analyze",
        min_value=0,
        max_value=n_clusters - 1,
        value=0,
    )

    sorted_idx = np.argsort(-dists[k])
    closest = sorted_idx[1:4]

    sorted_furthest = np.argsort(dists[k])
    furthest = [c for c in sorted_furthest if c not in closest and c != k][:3]

    st.markdown(f"**Analyzing neighbors of cluster {k}**")

    # ---------------------------------------------------------
    # CLOSEST CLUSTERS — 3 SIDE-BY-SIDE PREVIEWS
    # ---------------------------------------------------------
    st.subheader("Closest clusters")

    cols = st.columns(3)

    for col, c in zip(cols, closest):
        with col:
            st.markdown(f"#### Cluster {c}")

            # All items (image or caption) in this cluster
            cluster_items = [i for i, lab in enumerate(labels_proj) if lab == c]
            total_c = len(cluster_items)

            # Session key for navigation
            key_c = f"closest_cluster_pos_{c}"
            if key_c not in st.session_state:
                st.session_state[key_c] = 0

            pos = st.session_state[key_c]

            # Navigation buttons
            b_prev, b_next = st.columns([1, 1])
            with b_prev:
                if st.button("◀", key=f"closest_prev_{c}"):
                    st.session_state[key_c] = (pos - 1) % total_c
                    pos = st.session_state[key_c]
            with b_next:
                if st.button("▶", key=f"closest_next_{c}"):
                    st.session_state[key_c] = (pos + 1) % total_c
                    pos = st.session_state[key_c]

            # Current item
            item_idx = cluster_items[pos]

            # -----------------------------
            # IMAGE ITEM
            # -----------------------------
            if item_idx < num_images:
                img_path = IMAGE_PATHS[item_idx]
                st.image(img_path, width = 250)
                st.caption(f"Image index: {item_idx} — {pos+1}/{total_c}")

            # -----------------------------
            # CAPTION ITEM
            # -----------------------------
            else:
                cap_idx = item_idx - num_images
                caption_list = get_captions(IMAGE_PATHS[cap_idx])
                caption_text = caption_list[0] if caption_list else "(no caption found)"

                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        margin:8px 0;
                        background:#f9f9f9;
                        border-radius:6px;
                        border-left:4px solid #999;
                    ">
                        <b>Caption {cap_idx}</b><br/><br/>
                        {caption_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Caption index: {cap_idx} — {pos+1}/{total_c}")

    # ---------------------------------------------------------
    # FURTHEST CLUSTERS — 3 SIDE-BY-SIDE CAROUSELS
    # ---------------------------------------------------------
    st.subheader("Furthest clusters")

    cols = st.columns(3)

    for col, c in zip(cols, furthest):
        with col:
            st.markdown(f"#### Cluster {c}")

            cluster_items = [i for i, lab in enumerate(labels_proj) if lab == c]
            total_c = len(cluster_items)

            key_c = f"furthest_cluster_pos_{c}"
            if key_c not in st.session_state:
                st.session_state[key_c] = 0

            pos = st.session_state[key_c]

            b_prev, b_next = st.columns([1, 1])
            with b_prev:
                if st.button("◀", key=f"furthest_prev_{c}"):
                    st.session_state[key_c] = (pos - 1) % total_c
                    pos = st.session_state[key_c]
            with b_next:
                if st.button("▶", key=f"furthest_next_{c}"):
                    st.session_state[key_c] = (pos + 1) % total_c
                    pos = st.session_state[key_c]

            item_idx = cluster_items[pos]

            # IMAGE
            if item_idx < num_images:
                img_path = IMAGE_PATHS[item_idx]
                st.image(img_path, width = 250)
                st.caption(f"Image index: {item_idx} — {pos+1}/{total_c}")

            # CAPTION
            else:
                cap_idx = item_idx - num_images
                caption_list = get_captions(IMAGE_PATHS[cap_idx])
                caption_text = caption_list[0] if caption_list else "(no caption found)"

                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        margin:8px 0;
                        background:#f9f9f9;
                        border-radius:6px;
                        border-left:4px solid #999;
                    ">
                        <b>Caption {cap_idx}</b><br/><br/>
                        {caption_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(f"Caption index: {cap_idx} — {pos+1}/{total_c}")

app_footer()
# ---------------------------------------------------------
# SELECT FUSION CONFIG
# ---------------------------------------------------------
with tab_fusion:
    st.header("📊 Alignment Metadata")

    st.markdown("""
        ### 📊 Alignment Results
        This tab provides a complete quanitative analysis of the aligned models.
        """)
    
    st.dataframe(meta)

    st.header("📈 Best Alignment Configurations")
    st.dataframe(best)
    
app_footer()