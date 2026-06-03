import streamlit as st
import os
import base64

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

def load_image_base64(filename):
    """Return base64-encoded image content."""
    path = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(path):
        st.warning(f"⚠️ Missing asset: {filename}")
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def app_header():
    fac_img = load_image_base64("fac.jpg")
    logo_img = load_image_base64("fpms_logo_long.png")
    school_img = load_image_base64("umons_logo.jpg")

    # --- Full-width banner ---
    if fac_img:
        st.markdown(
            f"""
            <div style="width:100%; text-align:center; margin-bottom:20px;">
                <img src="data:image/jpeg;base64,{fac_img}"
                     style="width:100%; max-height:260px; object-fit:cover; border-radius:4px;">
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Title + Subtitle ---
    st.markdown(
        """
        <div style="text-align:center; margin-top:10px;">
            <h1 style="font-size:38px; margin-bottom:5px; color:#333;">
                MAREL
            </h1>
            <h2 style="font-size:18px; color:#666; margin-top:0;">
                Multimodal Alignment, Retrieval, Explainability and Latent-Space Representations
            </h2>
            <p style="font-size:18px; color:#666; margin-top:0;">
                A Comprehensive Evaluation
            </p>            
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Logo centered under title ---
    if logo_img or school_img:
        col1, col2 = st.columns([1,1])

        with col1:
            if logo_img:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:10px;">
                        <img src="data:image/png;base64,{logo_img}" style="width:160px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            if school_img:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:10px;">
                        <img src="data:image/png;base64,{school_img}" style="width:160px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    st.markdown("<hr style='margin-top:25px; margin-bottom:25px;'>", unsafe_allow_html=True)

def app_footer():
    school_img = load_image_base64("umons_logo.jpg")
    logo_img = load_image_base64("fpms_logo_long.png")

    st.markdown(
        """
        <hr style="margin-top:40px;">
        <div style="text-align:center; font-size:14px; color:#777; padding:15px 0;">
            © 2026 UMONS — Faculté Polytechnique  
            <br> Aysel MUSCATO
            <br> Prof Sidi MAHMOUDI
            <br> Aurélie COOLS
            <br>Developed as part of the Master’s Thesis
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if logo_img or school_img:
        col1, col2 = st.columns([1,1])

        with col1:
            if logo_img:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:10px;">
                        <img src="data:image/png;base64,{logo_img}" style="width:160px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:
            if school_img:
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-top:10px;">
                        <img src="data:image/png;base64,{school_img}" style="width:160px;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )

