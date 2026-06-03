import os
import streamlit as st

def check_path(path, description, errors):
    """Utility to check if a file or folder exists."""
    if not os.path.exists(path):
        errors.append(f"❌ Missing {description}: `{path}`")
    return errors


def validate_app_data():
    """
    Validates that all required data files and folders exist.
    Returns True if everything is OK, False otherwise.
    """

    errors = []

    # -----------------------------
    # BASE FOLDERS
    # -----------------------------
    base_folders = {
        "Unimodal folder": "data/unimodal",
        "Multimodal folder": "data/multimodal",
        "Fusion indexes": "data/fusion_indexes",
        "Dataset folder": "data/dataset",
        "Images folder": "data/images",
    }

    for desc, path in base_folders.items():
        errors = check_path(path, desc, errors)

    # -----------------------------
    # UNIMODAL FILES
    # -----------------------------
    unimodal_required = [
        "data/unimodal/metrics/vision_retrieval_results.csv",
        "data/unimodal/metrics/text_retrieval_results.csv",
        "data/unimodal/metrics/Explainability_Vision.csv",
        "data/unimodal/metrics/Explainability_Text.csv",
        "data/unimodal/metrics/global_unimodal_metrics.pkl",
    ]

    for f in unimodal_required:
        errors = check_path(f, "Unimodal metrics file", errors)

    # -----------------------------
    # MULTIMODAL (SOTA) FILES
    # -----------------------------
    sota_required = [
        "data/multimodal/clip/vision_embeddings.npy",
        "data/multimodal/clip/text_embeddings.npy",
        "data/multimodal/openclip_l14/vision_embeddings.npy",
        "data/multimodal/openclip_l14/text_embeddings.npy",
    ]

    for f in sota_required:
        errors = check_path(f, "SOTA embeddings", errors)

    # -----------------------------
    # FUSION FILES
    # -----------------------------
    fusion_required = [
        "data/fusion_indexes/fusion_indexes/fusion_index_metadata.csv",
    ]

    for f in fusion_required:
        errors = check_path(f, "Fusion metadata", errors)

    # -----------------------------
    # DATASET FILES
    # -----------------------------
    dataset_required = [
        "data/dataset/df_Flickr8k.pkl",
    ]

    for f in dataset_required:
        errors = check_path(f, "Dataset file", errors)

    # -----------------------------
    # FINAL VALIDATION
    # -----------------------------
    if len(errors) > 0:
        st.error("🚨 **Critical data files are missing. The app cannot run.**")
        for e in errors:
            st.write(e)
        st.stop()
        return False

    st.success("✅ All required data files are present.")
    return True
