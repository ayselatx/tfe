import streamlit as st
from utils.auth import check_login_status, init_db
init_db()
from utils.ui import fixed_image, section_title

from utils.layout import app_header, app_footer
app_header()

if not check_login_status():
    st.switch_page("pages/login.py")
    
    
from utils.loaders import load_vision_metrics, load_vision_models
from utils.retrieval_unimodal import retrieve_vision
from utils.plots import plot_recall_bars, plot_radar

st.set_page_config(page_title="Unimodal Vision", page_icon="🖼️", layout="wide")

st.title("🖼️ Unimodal — Vision Models")

with st.sidebar:
        st.page_link("pages/my_account.py", label="My Account", icon="👤")
        st.page_link("pages/unimodal.py", label="Unimodal Benchmarking", icon="📚")
        st.page_link("pages/unimodal_text.py", label="Unimodal Text", icon="📘")
        st.page_link("pages/unimodal_vision.py", label="Unimodal Vision", icon="🖼️")
        st.page_link("pages/unimodal_global.py", label="Unimodal Global", icon="🌍")
        st.page_link("pages/multimodal_sota.py", label="SOTA Retrieval", icon="🚀")
        st.page_link("pages/fusion_results.py", label="Fusion Results", icon="🔀")
        st.page_link("pages/multimodal_evaluation.py", label="Multimodal Evaluation", icon="📊")
        st.page_link("pages/rag.py", label="RAG Ameliorations", icon="✨")


metrics = load_vision_metrics()
models = load_vision_models()

from PIL import Image

# Load stress-test images
stress_images = []
for path in metrics["stress_images"]:
    img = Image.open(path).convert("RGB")
    stress_images.append((img, path))

# Initialize session state
if "selected_query" not in st.session_state:
    st.session_state.selected_query = None
if "results" not in st.session_state:
    st.session_state.results = None

st.header("🔎 Real‑Time Image Retrieval")

# ---------------------------------------------------------
# 1. SHOW GALLERY (only if no query selected yet)
# ---------------------------------------------------------
if st.session_state.selected_query is None:
    st.write("### Choose a query image")

    cols = st.columns(5)

    for idx, (img, path) in enumerate(stress_images):
        with cols[idx % 5]:
            st.image(img, width=150)
            if st.button(f"Select {idx+1}", key=f"select_{idx}"):
                st.session_state.selected_query = (img, path)
                st.session_state.results = None
                st.rerun()

# ---------------------------------------------------------
# 2. SHOW SELECTED QUERY + RETRIEVE BUTTON
# ---------------------------------------------------------
else:
    query_img, query_path = st.session_state.selected_query

    st.write("### Selected Query Image")
    st.image(query_img, width=250)

    # Reset button to go back to gallery
    if st.button("🔄 Choose another image"):
        st.session_state.selected_query = None
        st.session_state.results = None
        st.rerun()

    model_name = st.selectbox("Choose a vision model", list(models.keys()))

    if st.button("Retrieve"):
        results = retrieve_vision(query_path, model_name)
        st.session_state.results = results
        st.rerun()

# ---------------------------------------------------------
# 3. SHOW RETRIEVAL RESULTS
# ---------------------------------------------------------
if st.session_state.results is not None:
    st.write("### Top‑20 Retrieved Images")

    cols = st.columns(5)
    for idx, (img, score) in enumerate(st.session_state.results):
        with cols[idx % 5]:
            st.image(img, width=150, caption=f"{score:.4f}")


# --- Metrics ---
st.header("📊 Performance Metrics")
st.dataframe(metrics["performance"])

st.header("🧠 Explainability Metrics")
st.dataframe(metrics["explainability"])

st.header("⚡ Efficiency Metrics")
st.dataframe(metrics["efficiency"])

# --- Graphs ---
st.header("📈 Graphs")
plot_recall_bars(metrics["performance"])
plot_radar(metrics["explainability"])

app_footer()
