import streamlit as st
from utils.auth import check_login_status, init_db
init_db()
from utils.ui import fixed_image, section_title

if not check_login_status():
    st.switch_page("pages/login.py")

from utils.loaders import load_global_unimodal_results, load_unimodal_metrics


from utils.layout import app_header, app_footer
app_header()


st.set_page_config(page_title="Unimodal Global", page_icon="🌍", layout="wide")

st.title("🌍 Unimodal — Global Summary")

with st.sidebar:
    st.page_link("pages/my_account.py", label="My Account", icon="👤")
    st.page_link("pages/unimodal_text.py", label="Unimodal Text", icon="📘")
    st.page_link("pages/unimodal_vision.py", label="Unimodal Vision", icon="🖼️")
    st.page_link("pages/unimodal_global.py", label="Unimodal Global", icon="🌍")
    st.page_link("pages/multimodal_sota.py", label="SOTA Retrieval", icon="🚀")
    st.page_link("pages/fusion_results.py", label="Fusion Results", icon="🔀")
    st.page_link("pages/multimodal_evaluation.py", label="Multimodal Evaluation", icon="📊")
    st.page_link("pages/rag.py", label="RAG Ameliorations", icon="✨")

df_vision_raw, df_text_raw, df_vision_norm, df_text_norm = load_unimodal_metrics()

def normalize_weights(*weights):
    total = sum(weights)
    if total == 0:
        # éviter division par zéro
        n = len(weights)
        return [1/n] * n
    return [w / total for w in weights]

def show_model_summary(model_row, title):
    st.subheader(title)

    # Extract metadata
    model_name = model_row["Model"]
    modality   = model_row["Modality"]

    # Top metadata row
    st.markdown(f"""
    ### **{model_name}**
    **Modality:** {modality}
    """)

    # Three columns
    col1, col2, col3 = st.columns(3)

    # PERFORMANCE
    with col1:
        st.markdown("### 📈 Performance")
        st.metric("Recall@1",  f"{model_row['Recall@1']:.3f}")
        st.metric("Recall@5",  f"{model_row['Recall@5']:.3f}")
        st.metric("Recall@10", f"{model_row['Recall@10']:.3f}")

    # EFFICIENCY
    with col2:
        st.markdown("### ⚡ Efficiency")
        st.metric("Inference Time", f"{model_row['Inference_time']:.3f}s")
        st.metric("Embedding Time", f"{model_row['Embedding_time']:.3f}s")
        st.metric("Memory",         f"{model_row['Memory_mb']:.1f} MB")

    # EXPLAINABILITY
    with col3:
        st.markdown("### 🧠 Explainability")
        st.metric("Faithfulness",   f"{model_row['Faithfulness']:.3f}")
        st.metric("Sparsity",       f"{model_row['Sparsity']:.3f}")
        st.metric("Rank Corr.",     f"{model_row['Rank_corr']:.3f}")
        st.metric("Complexity",     f"{model_row['Complexity']:.3f}")

# ============================================================
# WEIGHTING UI
# ============================================================

st.header("⚖️ Customize Model Selection Weights")

# -------------------------
# MAIN WEIGHTS (α, β, γ)
# -------------------------
st.subheader("Main Criteria Weights")
adjust = st.checkbox("Adjust metric weighting")

if adjust:
    st.info("Adjust weights for each category.")
    alpha = st.slider("Performance Weight (α)", 0.0, 1.0, 0.3, 0.01)
    beta  = st.slider("Explainability Weight (β)", 0.0, 1.0, 0.3, 0.01)
    gamma = st.slider("Efficiency Weight (γ)", 0.0, 1.0, 0.3, 0.01)
    total = alpha + beta + gamma
    if total == 0:
        st.error("At least one weight must be > 0.")
    else:
        total = alpha + beta + gamma
        st.success(f"Normalized Weights → α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

else:
    alpha = 0.33
    beta  = 0.33
    gamma = 0.33
    total = alpha + beta + gamma

alpha /= total
beta  /= total
gamma /= total


# -------------------------
# ADVANCED SUB-WEIGHTS
# -------------------------
st.subheader("Advanced Metric Weighting")
advanced = st.checkbox("Enable advanced metric weighting")

if advanced:
    st.info("Adjust sub-weights for each category.")

    # Performance sub-weights
    st.markdown("### Performance Sub-Weights")
    w_r1  = st.slider("Recall@1 Weight", 0.0, 1.0, 0.3, 0.01)
    w_r5  = st.slider("Recall@5 Weight", 0.0, 1.0, 0.3, 0.01)
    w_r10 = st.slider("Recall@10 Weight", 0.0, 1.0, 0.3, 0.01)
    perf_total = w_r1 + w_r5 + w_r10
    w_r1, w_r5, w_r10 = w_r1/perf_total, w_r5/perf_total, w_r10/perf_total

    # Explainability sub-weights
    st.markdown("### Explainability Sub-Weights")
    w_faith = st.slider("Faithfulness Weight", 0.0, 1.0, 0.25, 0.01)
    w_spar  = st.slider("Sparsity Weight", 0.0, 1.0, 0.25, 0.01)
    w_rank  = st.slider("Rank Correlation Weight", 0.0, 1.0, 0.25, 0.01)
    w_comp  = st.slider("Complexity Weight", 0.0, 1.0, 0.25, 0.01)
    exp_total = w_faith + w_spar + w_rank + w_comp
    w_faith, w_spar, w_rank, w_comp = (
        w_faith/exp_total, w_spar/exp_total, w_rank/exp_total, w_comp/exp_total
    )

    # Efficiency sub-weights
    st.markdown("### Efficiency Sub-Weights")
    w_inf = st.slider("Inference Time Weight", 0.0, 1.0, 0.3, 0.01)
    w_emb = st.slider("Embedding Time Weight", 0.0, 1.0, 0.3, 0.01)
    w_mem = st.slider("Memory Weight", 0.0, 1.0, 0.3, 0.01)
    eff_total = w_inf + w_emb + w_mem
    w_inf, w_emb, w_mem = w_inf/eff_total, w_emb/eff_total, w_mem/eff_total

else:
    # default weights
    w_r1, w_r5, w_r10 = 0.33, 0.33, 0.33
    w_faith, w_spar, w_rank, w_comp = 0.25, 0.25, 0.25, 0.25
    w_inf, w_emb, w_mem = 0.33, 0.33, 0.33

# ============================================================
# APPLY WEIGHTS (FIXED INDENTATION)
# ============================================================

def compute_scores(df):
    df["Performance_score"] = (
        w_r1  * df["Recall@1"] +
        w_r5  * df["Recall@5"] +
        w_r10 * df["Recall@10"]
    )

    df["Explainability_score"] = (
        w_faith * df["Faithfulness"] +
        w_spar  * df["Sparsity"] +
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

# recompute scores
compute_scores(df_vision_norm)
compute_scores(df_text_norm)

# recompute rankings
vision_best = df_vision_norm.sort_values("Final_score", ascending=False)
text_best   = df_text_norm.sort_values("Final_score", ascending=False)

# ============================================================
# DISPLAY RESULTS
# ============================================================


st.header("📝 Conclusion")
st.write("""
Based on your chosen weights for performance, explainability, and efficiency,
these models are selected as the best candidates for multimodal fusion.
""")

# ============================
# BEST VISION MODELS (RAW)
# ============================

# Best Vision Model
best_model_name = vision_best.iloc[0]["Model"]
best_model_raw = df_vision_raw[df_vision_raw["Model"] == best_model_name]

if best_model_raw.empty:
    st.error(f"Raw metrics not found for model: {best_model_name}")
else:
    show_model_summary(best_model_raw.iloc[0], "🏆 Best Vision Model")

# Second Best Vision Model
second_model_name = vision_best.iloc[1]["Model"]
second_model_raw = df_vision_raw[df_vision_raw["Model"] == second_model_name]

if second_model_raw.empty:
    st.error(f"Raw metrics not found for model: {second_model_name}")
else:
    show_model_summary(second_model_raw.iloc[0], "🥈 Second Best Vision Model")


# ============================
# BEST TEXT MODELS (RAW)
# ============================

# Best Text Model
best_text_name = text_best.iloc[0]["Model"]
best_text_raw = df_text_raw[df_text_raw["Model"] == best_text_name]

if best_text_raw.empty:
    st.error(f"Raw metrics not found for model: {best_text_name}")
else:
    show_model_summary(best_text_raw.iloc[0], "🏆 Best Text Model")

# Second Best Text Model
second_text_name = text_best.iloc[1]["Model"]
second_text_raw = df_text_raw[df_text_raw["Model"] == second_text_name]

if second_text_raw.empty:
    st.error(f"Raw metrics not found for model: {second_text_name}")
else:
    show_model_summary(second_text_raw.iloc[0], "🥈 Second Best Text Model")



#st.header("🏆 Best Vision Model")
#st.write(vision_best.iloc[0])

#st.header("🥈 Second Best Vision Model")
#st.write(vision_best.iloc[1])

#st.header("🏆 Best Text Model")
#st.write(text_best.iloc[0])

#st.header("🥈 Second Best Text Model")
#st.write(text_best.iloc[1])

st.header("📊 Full Vision Ranking")
st.dataframe(df_vision_norm)

st.header("📊 Full Text Ranking")
st.dataframe(df_text_norm)

app_footer()
