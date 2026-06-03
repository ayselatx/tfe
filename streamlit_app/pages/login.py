import streamlit as st
from utils.auth import authenticate
from utils.layout import app_header, app_footer
from utils.sql_retrievals import init_retrieval_tables
init_retrieval_tables()

st.set_page_config(page_title="Login", page_icon="🔐", layout="centered")

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

</style>
""", unsafe_allow_html=True)

#st.info(""" """)

st.title("Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if authenticate(username, password):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        st.success("Login successful")
        st.switch_page("app.py")
    else:
        st.error("Invalid username or password")

st.write("Don't have an account?")
if st.button("Create Account"):
    st.switch_page("pages/create_account.py")

app_footer()
