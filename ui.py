# app_streamlit.py
import io
import streamlit as st
from PIL import Image
import numpy as np

from colorbook_core import colorbook_pipeline

st.set_page_config(page_title="Custom Paint‑By‑Numbers", page_icon="🎨", layout="wide")
st.title("🎨 Upload photo → automatic Paint‑By‑Numbers")

with st.sidebar:
    st.header("Controls")
    n_colors = st.slider("Palette size (numbered colors)", 8, 36, 18, 1)
    blur_palette = st.slider("Palette blur (for clustering)", 0.0, 3.0, 1.2, 0.1)
    pre_smooth = st.slider("Pre-smooth input (denoise)", 0.0, 3.0, 0.6, 0.1)
    min_area = st.slider("Merge regions smaller than … pixels", 20, 2000, 150, 10)
    merge_iters = st.slider("Max merge passes", 1, 12, 6, 1)
    draw_nums = st.checkbox("Draw numbers", value=True)

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.caption(f"Input size: {pil.width}×{pil.height}px")

    with st.spinner("Generating…"):
        pbn, outlines, legend, labels = colorbook_pipeline(
            pil_image=pil,
            n_colors=n_colors,
            blur_for_palette=blur_palette,
            pre_smooth=pre_smooth,
            min_region_area_px=min_area,
            merge_iters=merge_iters,
            draw_numbers_flag=draw_nums,
        )

    c1, c2 = st.columns([3,2], gap="large")
    with c1:
        st.subheader("Paint‑By‑Numbers")
        st.image(pbn, use_column_width=True)
        st.subheader("Outlines only")
        st.image(outlines, use_column_width=True)
    with c2:
        st.subheader("Color legend")
        st.image(legend, use_column_width=True)

    # Downloads
    def _dl(img: Image.Image, fname: str, fmt: str):
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        st.download_button(f"Download {fname}.{fmt.lower()}", buf.getvalue(), file_name=f"{fname}.{fmt.lower()}")

    st.markdown("---")
    colA, colB, colC = st.columns(3)
    with colA: _dl(pbn, "paint_by_numbers", "PNG")
    with colB: _dl(outlines, "outlines_only", "PNG")
    with colC: _dl(legend, "legend", "PNG")
else:
    st.info("Upload a portrait or pet photo to start.")
