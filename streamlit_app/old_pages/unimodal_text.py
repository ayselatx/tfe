import streamlit as st
from utils.auth import check_login_status, init_db
from utils.retrieval_unimodal import retrieve_text
from utils.loaders import load_text_metrics, load_text_models
from utils.ui import fixed_image, section_title

init_db()

from utils.layout import app_header, app_footer
app_header()

if not check_login_status():
    st.switch_page("pages/login.py")

st.set_page_config(page_title="Unimodal Text", page_icon="📘", layout="wide")

st.title("Unimodal — Text Models")

with st.sidebar:
    st.page_link("pages/my_account.py", label="My Account", icon="👤")
    st.page_link("pages/unimodal_text.py", label="Unimodal Text", icon="📘")
    st.page_link("pages/unimodal_vision.py", label="Unimodal Vision", icon="🖼️")
    st.page_link("pages/unimodal_global.py", label="Unimodal Global", icon="🌍")
    st.page_link("pages/multimodal_sota.py", label="SOTA Retrieval", icon="🚀")
    st.page_link("pages/fusion_results.py", label="Fusion Results", icon="🔀")
    st.page_link("pages/multimodal_evaluation.py", label="Multimodal Evaluation", icon="📊")
    st.page_link("pages/rag.py", label="RAG Ameliorations", icon="✨")

metrics = load_text_metrics()
models = load_text_models()

# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "selected_caption" not in st.session_state:
    st.session_state.selected_caption = None
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------------
# 1. Caption selection (clickable list)
# ---------------------------------------------------------
st.header("Real‑Time Text Retrieval")

if st.session_state.selected_caption is None:
    st.write("### Choose a query caption")

    for idx, cap in enumerate(metrics["stress_captions"]):
        if st.button(cap, key=f"cap_{idx}"):
            st.session_state.selected_caption = cap
            st.session_state.results = None
            st.rerun()

# ---------------------------------------------------------
# 2. Show selected caption + retrieve button
# ---------------------------------------------------------
else:
    st.write("### Selected Caption")
    st.info(st.session_state.selected_caption)

    if st.button("🔄 Choose another caption"):
        st.session_state.selected_caption = None
        st.session_state.results = None
        st.rerun()

    model_name = st.selectbox("Choose a text model", list(models.keys()))

    if st.button("Retrieve"):
        st.session_state.results = retrieve_text(st.session_state.selected_caption, model_name)
        st.rerun()

# ---------------------------------------------------------
# 3. Show retrieval results
# ---------------------------------------------------------
if st.session_state.results is not None:
    st.subheader("Top‑20 Retrieved Captions")
    for cap, score in st.session_state.results:
        st.write(f"**{cap}** — {score:.4f}")

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
st.header("Performance Metrics")
st.dataframe(metrics["performance"])

st.header("Explainability Metrics")
st.dataframe(metrics["explainability"])

st.header("Efficiency Metrics")
st.dataframe(metrics["efficiency"])

# ---------------------------------------------------------
# Graphs
# ---------------------------------------------------------
st.header("Graphs")
plot_recall_bars(metrics["performance"])
plot_radar(metrics["explainability"])

app_footer()
