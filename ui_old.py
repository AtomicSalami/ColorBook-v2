import streamlit as st
import numpy as np
from PIL import Image

from colorbook_old import (
    run_mozaik, run_colorbook
)

st.set_page_config(page_title="Roma-ColorBook", layout="wide")

st.title("Roma-ColorBook (Streamlit)")
st.caption("Fast palette clustering + robust small-region merging. Numbers have no white halo.")

tab1, tab2 = st.tabs(["Mozaik", "Color Book"])

with st.sidebar:
    st.header("Global settings")
    n_colors = st.slider("Palette size (colors)", 8, 36, 16, step=1)
    blur = st.slider("Pre-blur strength", 0.0, 5.0, 1.5, step=0.1)
    uploaded = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png", "webp"])

def _load_image():
    if not uploaded:
        st.info("Upload an image to begin.")
        st.stop()
    img = Image.open(uploaded).convert("RGB")
    return np.array(img), uploaded.name

with tab1:
    if uploaded:
        img_np, fname = _load_image()
        with st.spinner("Generating mozaik..."):
            out, tiles, legend, _ctx = run_mozaik(img_np, n_colors, blur)
        c1, c2 = st.columns([3,1])
        with c1: st.image(out, caption=f"Mozaik • tiles: {tiles}", use_column_width=True)
        with c2: st.image(legend, caption="Legend", use_column_width=True)

with tab2:
    mode = st.radio("Mode", ["Pixel-Art", "Natural"], index=1, horizontal=True)
    canny_low = st.slider("Natural: Canny low", 1, 5, 1, step=1)
    canny_high = st.slider("Natural: Canny high", 10, 25, 5, step=5)

    if uploaded:
        img_np, fname = _load_image()
        with st.spinner("Generating color book..."):
            out, tiles, legend, _ctx = run_colorbook(
                img_np, n_colors, blur, mode, canny_low, canny_high
            )
        c1, c2 = st.columns([3,1])
        with c1: st.image(out, caption=f"{mode} • tiles: {tiles}", use_column_width=True)
        with c2: st.image(legend, caption="Legend", use_column_width=True)
