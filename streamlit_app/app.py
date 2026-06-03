import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="OSIRIS Platform",
    page_icon="🔎",
    layout="wide"
)

from utils.auth import check_login_status
from utils.validator import validate_app_data
from utils.sql_retrievals import (
    init_retrieval_tables,
    init_unimodal_tables,
    init_custom_models_table
)
from utils.layout import app_header, app_footer


# =========================
# Initialization
# =========================

validate_app_data()
init_unimodal_tables()
init_retrieval_tables()
init_custom_models_table()

# Redirect if not logged in
if not check_login_status():
    st.switch_page("pages/login.py")


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
# Header
# =========================

app_header()


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


# =========================
# Feature Grid
# =========================

def grid(items, columns=2):
    cols = st.columns(columns)

    for idx, item in enumerate(items):
        with cols[idx % columns]:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{item['title']}</h4>
                <p>{item['text']}</p>
            </div>
            """, unsafe_allow_html=True)


grid([
    {
        "title": "Unimodal",
        "text": "Image-to-Text and Text-to-Image retrieval analysis."
    },
    {
        "title": "Alignment",
        "text": "Projection and aligned embedding evaluation."
    },
    {
        "title": "Multimodal",
        "text": "Fusion operators and multimodal retrieval benchmarks."
    },
    {
        "title": "RAG",
        "text": "Retrieval-Augmented Generation integrated after retrieval to refine caption."
    }
])


# =========================
# Footer
# =========================

app_footer()