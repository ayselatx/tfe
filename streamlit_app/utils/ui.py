import streamlit as st

def fixed_image(img, size=256):
    st.image(img, width=size)

def section_title(text):
    st.markdown(f"## {text}")

def subsection(text):
    st.markdown(f"### {text}")
