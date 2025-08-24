# colorbook_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from sklearn.cluster import MiniBatchKMeans
import scipy.ndimage as ndi


# -----------------------------
# Helpers: fast palette + quantization
# -----------------------------
def _downscale_for_clustering(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

def learn_palette(img_rgb: np.ndarray, n_colors: int, blur: float = 1.2) -> np.ndarray:
    """Learn n_colors centers using MiniBatchKMeans on a blurred, downscaled preview."""
    if blur > 0:
        img_rgb = np.array(Image.fromarray(img_rgb).filter(ImageFilter.GaussianBlur(blur)))
    small = _downscale_for_clustering(img_rgb, 512)
    pixels = small.reshape(-1, 3)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=4096, n_init="auto")
    kmeans.fit(pixels)
    centers = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)
    return centers

def quantize_to_palette(img_rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Return per-pixel label map 1..K for nearest color in palette."""
    flat = img_rgb.reshape(-1, 3).astype(np.float32)[:, None, :]     # (N,1,3)
    pal  = palette.astype(np.float32)[None, :, :]                     # (1,K,3)
    d = np.linalg.norm(flat - pal, axis=2)                            # (N,K)
    labels = d.argmin(axis=1).astype(np.int32) + 1
    return labels.reshape(img_rgb.shape[:2])

# -----------------------------
# Region merging core
# -----------------------------
def _largest_shared_border(target_mask: np.ndarray, labels: np.ndarray) -> int | None:
    """
    Among neighbors around target_mask, return the label that shares the
    longest border length with the target. If none, return None.
    """
    # Border pixels = dilation minus region
    dil = ndi.binary_dilation(target_mask, iterations=1)
    border = dil & (~target_mask)

    h, w = labels.shape
    ys, xs = np.where(border)
    if ys.size == 0:
        return None

    counts: Dict[int, int] = {}
    for y, x in zip(ys, xs):
        # 4-neighborhood around border pixel
        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= ny < h and 0 <= nx < w:
                lab = int(labels[ny, nx])
                if lab != 0 and not target_mask[ny, nx]:
                    counts[lab] = counts.get(lab, 0) + 1

    if not counts:
        return None
    # pick neighbor with max border (ties broken by larger label id deterministically)
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

def merge_small_regions(labels: np.ndarray, min_area_px: int, max_iters: int = 5) -> np.ndarray:
    """
    Iteratively merges every connected component with area < min_area_px
    into the neighbor that shares the largest border with it.
    """
    lab = labels.copy()
    h, w = lab.shape

    for _ in range(max_iters):
        changed = False
        for lbl in np.unique(lab):
            if lbl == 0: 
                continue
            comp_map, n = ndi.label(lab == lbl)
            for cid in range(1, n + 1):
                region = (comp_map == cid)
                area = int(region.sum())
                if area >= min_area_px:
                    continue
                tgt = _largest_shared_border(region, lab)
                if tgt is not None and tgt != lbl:
                    lab[region] = tgt
                    changed = True
        if not changed:
            break

    # Optional cleanup: fill tiny holes wholly surrounded by one label
    # (1) close within each label; (2) majority-vote fill for 1px holes
    se = np.ones((3,3), bool)
    for lbl in np.unique(lab):
        if lbl == 0: 
            continue
        mask = (lab == lbl)
        closed = ndi.binary_opening(ndi.binary_closing(mask, structure=se), structure=se)
        lab[(mask ^ closed) & closed] = lbl
    return lab

# -----------------------------
# Number placement (no halo)
# -----------------------------
def component_centers_for_numbers(labels: np.ndarray) -> List[Tuple[int,int,int]]:
    """
    For each connected component of each label, return (label, y, x) where (y,x)
    is the argmax of the distance transform to the boundary -> safest interior.
    """
    out: List[Tuple[int,int,int]] = []
    for lbl in np.unique(labels):
        if lbl == 0: 
            continue
        comp, n = ndi.label(labels == lbl)
        for cid in range(1, n + 1):
            region = (comp == cid)
            if region.sum() == 0:
                continue
            # distance to the *background of this component* (i.e., boundary)
            dist = ndi.distance_transform_edt(region)
            y, x = np.unravel_index(int(dist.argmax()), dist.shape)
            out.append((int(lbl), int(y), int(x)))
    return out

# -----------------------------
# Rendering
# -----------------------------
@dataclass
class RenderConfig:
    outline_rgb: Tuple[int,int,int] = (25,25,25)
    outline_width: int = 1
    font: ImageFont.ImageFont | None = None  # default bitmap if None

def render_colorbook(
    labels: np.ndarray,
    palette: np.ndarray,
    draw_numbers: bool = True,
    config: RenderConfig = RenderConfig(),
) -> Image.Image:
    """
    Paint each label with its palette color, draw separating outlines, then numbers.
    """
    h, w = labels.shape
    out = Image.new("RGB", (w, h), (255,255,255))
    pix = np.array(out)

    # Map label->color
    id_to_color = {i+1: tuple(map(int, c)) for i, c in enumerate(palette)}
    for lid, rgb in id_to_color.items():
        pix[labels == lid] = rgb
    out = Image.fromarray(pix)

    # Draw outlines where 4-neighbors differ
    draw = ImageDraw.Draw(out)
    ly = np.pad(labels, ((0,1),(0,0)), mode='edge')
    lx = np.pad(labels, ((0,0),(0,1)), mode='edge')
    hmask = labels != ly[1:,:]    # horizontal borders (between rows)
    vmask = labels != lx[:,1:]    # vertical borders (between cols)
    # horizontal lines
    ys, xs = np.where(hmask)
    for y, x in zip(ys, xs):
        draw.line([(x, y+1), (x+1, y+1)], fill=config.outline_rgb, width=config.outline_width)
    # vertical lines
    ys, xs = np.where(vmask)
    for y, x in zip(ys, xs):
        draw.line([(x+1, y), (x+1, y+1)], fill=config.outline_rgb, width=config.outline_width)

    if draw_numbers:
        font = config.font or ImageFont.load_default()
        for lid, y, x in component_centers_for_numbers(labels):
            # Just black text; no white halo as requested
            draw.text((x, y), str(lid), fill=(0,0,0), font=font, anchor="mm")

    return out

def render_outlines_only(labels: np.ndarray, config: RenderConfig = RenderConfig()) -> Image.Image:
    h, w = labels.shape
    out = Image.new("RGB", (w, h), (255,255,255))
    draw = ImageDraw.Draw(out)
    ly = np.pad(labels, ((0,1),(0,0)), mode='edge')
    lx = np.pad(labels, ((0,0),(0,1)), mode='edge')
    hmask = labels != ly[1:,:]
    vmask = labels != lx[:,1:]
    ys, xs = np.where(hmask)
    for y, x in zip(ys, xs):
        draw.line([(x, y+1), (x+1, y+1)], fill=config.outline_rgb, width=config.outline_width)
    ys, xs = np.where(vmask)
    for y, x in zip(ys, xs):
        draw.line([(x+1, y), (x+1, y+1)], fill=config.outline_rgb, width=config.outline_width)
    return out

def make_legend(palette: np.ndarray) -> Image.Image:
    items = [(i+1, tuple(map(int, c))) for i, c in enumerate(palette)]
    sw, sh = 22, 20
    H = (len(items)+1)*(sh+6)
    W = 260
    img = Image.new("RGB", (W, min(H, 16000)), (255,255,255))
    d = ImageDraw.Draw(img)
    y = 6
    for idx, rgb in items:
        d.rectangle([6, y, 6+sw, y+sh], fill=rgb, outline=(0,0,0))
        d.text((6+sw+8, y+3), f"#{idx}  RGB {rgb}", fill=(0,0,0))
        y += sh+6
    return img

# -----------------------------
# Main pipeline
# -----------------------------
def colorbook_pipeline(
    pil_image: Image.Image,
    n_colors: int = 15,
    blur_for_palette: float = 1.2,
    pre_smooth: float = 0.6,
    min_region_area_px: int = 120,   # tune this! ~ area threshold in pixels
    merge_iters: int = 6,
    draw_numbers_flag: bool = True,
) -> Tuple[Image.Image, Image.Image, Image.Image, np.ndarray]:
    """
    Returns: (paint_by_numbers, outlines_only, legend, labels)
    """
    img = pil_image.convert("RGB")
    arr = np.array(img)
    if pre_smooth > 0:
        arr = np.array(Image.fromarray(arr).filter(ImageFilter.GaussianBlur(pre_smooth)))

    palette = learn_palette(arr, n_colors=n_colors, blur=blur_for_palette)
    labels = quantize_to_palette(arr, palette)
    labels = merge_small_regions(labels, min_area_px=min_region_area_px, max_iters=merge_iters)

    pbn = render_colorbook(labels, palette, draw_numbers=draw_numbers_flag)
    outlines = render_outlines_only(labels)
    legend = make_legend(palette)
    return pbn, outlines, legend, labels
