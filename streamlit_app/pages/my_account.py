import streamlit as st
import sqlite3
from utils.auth import check_login_status, init_db, update_username, update_email, update_password
from utils.layout import app_header, app_footer
from utils.sql_retrievals import (
    load_user_retrievals,
    load_retrieval_by_id,
    load_retrieval_by_id_all,
    load_user_retrievals_all,
    anonymize_retrieval,
    anonymize_unimodal_retrieval
)
from utils.paths import IMAGE_PATHS


init_db()
st.set_page_config(page_title="My Account", page_icon="👤", layout="centered")

# Redirect if not logged in
if not check_login_status():
    st.switch_page("pages/login.py")

def try_show_image(query):
    # If query looks like an image path
    if isinstance(query, str) and (query.endswith(".jpg") or query.endswith(".png")):
        try:
            st.image(query, width = 250)
        except:
            st.warning("Image preview unavailable.")

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
    ("Fusion", "pages/fusion_results.py", "🔀"),
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


st.title("👤 My Account")

username = st.session_state.get("username")

# Load user info
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute("SELECT username, email FROM users WHERE username=?", (username,))
row = c.fetchone()
conn.close()

current_username, current_email = row

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab_info, tab_history, tab_custom = st.tabs(["👤 Account Info", "📜 Retrieval History", "⭐ Saved Systems"])

# ---------------------------------------------------------
# TAB 1 — ACCOUNT INFO
# ---------------------------------------------------------
with tab_info:

    st.subheader("Account Information")

    st.write(f"**Username:** {current_username}")
    st.write(f"**Email:** {current_email}")

    # Logout
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = None
        st.switch_page("pages/login.py")
        
    st.markdown("---")

    # Collapsible update section
    with st.expander("Update Account Information"):
        st.write("Modify any field below and click the corresponding update button.")

        # --- Update Username ---
        st.subheader("Change Username")
        new_username = st.text_input("New Username", key="new_username_input")

        if st.button("Update Username"):
            if update_username(current_username, new_username):
                st.success("Username updated successfully")
                st.session_state["username"] = new_username
            else:
                st.error("Could not update username (maybe already taken)")

        st.markdown("---")

        # --- Update Email ---
        st.subheader("Change Email")
        new_email = st.text_input("New Email", key="new_email_input")

        if st.button("Update Email"):
            if update_email(current_username, new_email):
                st.success("Email updated successfully")
            else:
                st.error("Could not update email")

        st.markdown("---")

        # --- Update Password ---
        st.subheader("Change Password")
        new_password = st.text_input("New Password", type="password", key="new_password_input")

        if st.button("Update Password"):
            if update_password(current_username, new_password):
                st.success("Password updated successfully")
            else:
                st.error("Could not update password")

        st.markdown("---")

# ---------------------------------------------------------
# TAB 2 — RETRIEVAL HISTORY
# ---------------------------------------------------------
with tab_history:

    st.subheader("📜 Retrieval History")

    from datetime import datetime, timezone
    import pytz
    local_tz = pytz.timezone("Europe/Brussels")

    def to_local(ts):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "")).replace(tzinfo=timezone.utc)
            return dt.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S")
        except:
            return ts

    # Load unimodal
    unimodal_rows = load_user_retrievals(username)

    # Load fusion/multimodal/SOTA/RAG
    all_rows = load_user_retrievals_all(username)

    # Normalize unimodal rows
    normalized_unimodal = []
    for rid, modality, dataset, query, model, created_at in unimodal_rows:
        normalized_unimodal.append((
            rid, "unimodal", modality, model, None, None, None,
            dataset, query, created_at,
            None, None, None
        ))

    # Combine
    all_entries = normalized_unimodal + all_rows

    # Group by retrieval type
    groups = {
        "unimodal": [],
        "fusion": [],
        "multimodal": [],
        "rag": []
    }

    for entry in all_entries:
        rtype = entry[1]
        if rtype in groups:
            groups[rtype].append(entry)

    # -----------------------------
    # SECTION DEFINITIONS
    # -----------------------------
    sections = [
        ("🖼️ Unimodal Retrievals", "unimodal"),
        ("🔀 Alignment", "fusion"),
        ("📊 Multimodal Retrievals", "multimodal"),
        ("✨ RAG Refinements", "rag"),
    ]

    from utils.paths import IMAGE_PATHS

    # -----------------------------
    # RENDER SECTIONS VERTICALLY
    # -----------------------------
    for title, key in sections:

        st.markdown(f"## {title}")

        # Scrollable container
        box = st.container(height=400, border=True)

        with box:
            entries = groups[key]

            if not entries:
                st.info("No entries yet.")
                continue

            for (
                rid, rtype, qtype,
                vision, text, proj, fop,
                dataset, query, created_at,
                rag_prompt, rag_model, rag_refined
            ) in entries:

                created_local = to_local(created_at)

                if rtype == "unimodal":
                    header = f"🔹 {qtype.upper()} Search — {vision} — {created_local}"

                elif rtype == "fusion":
                    header = f"🔸 Aligning {vision} + {text} with {proj} — {created_local}"

                elif rtype == "multimodal":
                    # SOTA case
                    if text is None and proj is None:
                        # SOTA model
                        if fop:
                            header = f"🟣 {qtype.upper()} — {vision} (SOTA) + {fop} — {created_local}"
                        else:
                            header = f"🟣 {qtype.upper()} — {vision} (SOTA) — {created_local}"
                    else:
                        # Regular multimodal
                        fusion_part = f" + {fop}" if fop else ""
                        header = (
                            f"🟣 {qtype.upper()} — "
                            f"{vision} + {text} + {proj}{fusion_part} — {created_local}"
                        )

                elif rtype == "rag":
                    header = f"✨ RAG — {rag_model} — Image #{query} — {created_local}"

                else:
                    header = f"📁 Retrieval — {created_local}"

                with st.expander(f"{header}"):

                    st.write(f"**Dataset:** {dataset}")
                    st.write(f"**Query:** {query}")

                    # Image preview for any image path
                    try_show_image(query)

                    # -------------------------
                    # RAG SECTION
                    # -------------------------
                    if rtype == "rag":
                        st.markdown("### ✨ RAG Details")

                        # Show the image (RAG uses index)
                        try:
                            st.image(IMAGE_PATHS[int(query)], caption=f"Image #{query}", use_container_width=True)
                        except:
                            pass

                        st.write(f"**Model:** {rag_model}")
                        st.write("**Prompt:**")
                        st.code(rag_prompt)
                        st.write("**Refined Caption:**")
                        st.success(rag_refined)

                        # Delete button
                        st.warning("Delete this retrieval from your account?")
                        if st.button(f"Delete Retrieval #{rid}", key=f"del_rag_{rid}"):
                            anonymize_retrieval(rid)
                            st.success("Retrieval removed from your account.")
                            st.rerun()

                        continue  # Skip retrieval results

                    # -------------------------
                    # Load retrieval results
                    # -------------------------
                    if rtype == "unimodal":
                        data = load_retrieval_by_id(rid)
                    else:
                        data = load_retrieval_by_id_all(rid)

                    results = data["results"]

                    st.markdown("### Results")
                    for item, score in results:
                        st.write(f"- {item} — {score:.4f}")
                    st.markdown("---")
                    st.warning("Delete this retrieval from your account?")

                    if st.button(f"Delete Retrieval #{rid}", key=f"del_{rtype}_{rid}"):
                        anonymize_unimodal_retrieval(rid)
                        st.success("Retrieval removed from your account.")
                        st.rerun()



app_footer()


# ---------------------------------------------------------
# TAB 3 — Custom Model
# ---------------------------------------------------------
with tab_custom:

    st.subheader("⭐ Saved Custom Systems")

    from utils.sql_retrievals import load_user_custom_models, delete_custom_model, update_custom_model

    models = load_user_custom_models(username)

    if not models:
        st.info("You have not saved any custom system configurations yet.")
    else:
        for (
            mid, name, comment,
            alpha, beta, gamma,
            w_r1, w_r5, w_r10,
            w_faith, w_spar, w_rank, w_comp,
            w_inf, w_emb, w_mem,
            best_vision, best_text,
            timestamp
        ) in models:

            with st.expander(f"📌 {name} — {timestamp}"):

                st.markdown("### 📝 Description")
                st.write(comment if comment else "_No description provided._")

                st.markdown("### 🧠 Best Models")
                st.write(f"- **Vision:** {best_vision}")
                st.write(f"- **Text:** {best_text}")

                st.markdown("### ⚖️ Criteria Weights")

                weights_df = {
                    "Category": [
                        "α (Performance)", "β (Explainability)", "γ (Efficiency)",
                        "Recall@1", "Recall@5", "Recall@10",
                        "Faithfulness", "Sparsity", "Rank Corr.", "Complexity",
                        "Inference Time", "Embedding Time", "Memory"
                    ],
                    "Weight": [
                        alpha, beta, gamma,
                        w_r1, w_r5, w_r10,
                        w_faith, w_spar, w_rank, w_comp,
                        w_inf, w_emb, w_mem
                    ]
                }

                st.dataframe(weights_df, use_container_width=True)

                # -------------------------
                # MODIFY SECTION
                # -------------------------
                with st.expander("⚙️ Modify"):

                    st.markdown("### ✏️ Edit Model Information")

                    new_name = st.text_input("New Name", value=name, key=f"name_{mid}")
                    new_comment = st.text_area("New Description", value=comment, key=f"comment_{mid}")

                    if st.button("Save Changes", key=f"save_changes_{mid}"):
                        update_custom_model(mid, new_name, new_comment)
                        st.success("Model updated successfully.")
                        st.experimental_rerun()

                    st.markdown("---")

                    st.markdown("### 🗑️ Delete Model")
                    st.warning("Deleting this model is permanent.")

                    confirm = st.checkbox(f"Yes, delete '{name}' permanently", key=f"confirm_{mid}")

                    if st.button("Delete Model", key=f"delete_{mid}"):
                        if confirm:
                            delete_custom_model(mid)
                            st.success("Model deleted successfully.")
                            st.experimental_rerun()
                        else:
                            st.error("Please confirm deletion first.")
