from __future__ import annotations
import os
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Optional deps
try:
    import cv2  # for Canny in Natural mode
except Exception:
    cv2 = None

import scipy.ndimage as ndi
from sklearn.cluster import MiniBatchKMeans

# =============================
# CONFIG
# =============================
TILE_SIZE_MM = 1
DPI = 150
WEBP_LIMIT_PX = 16000

# =============================
# SAVE
# =============================
def save_processed_image(image: Image.Image, original_filename: str) -> str:
    os.makedirs("output", exist_ok=True)
    base = os.path.splitext(os.path.basename(original_filename))[0]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join("output", f"{base}_{ts}.png")
    image.save(out)
    return out

# =============================
# SPEED: palette
# =============================
def _downscale_for_clustering(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return img
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))

def get_dominant_colors(img_array: np.ndarray, n_colors: int) -> np.ndarray:
    small = _downscale_for_clustering(img_array, 512)
    pixels = small.reshape(-1, 3)
    km = MiniBatchKMeans(n_clusters=n_colors, random_state=42, batch_size=4096, n_init="auto")
    km.fit(pixels)
    return np.clip(km.cluster_centers_, 0, 255).astype(np.uint8)

# =============================
# PREPROCESS
# =============================
def bricks_to_px(bricks: int) -> int:
    return int((bricks * TILE_SIZE_MM / 10) * DPI)

def preprocess_image(image: np.ndarray, n_colors: int = 25, blur_strength: float = 1.5):
    # target ~100x100 tiles while preserving aspect
    width_bricks, height_bricks = 100, 100
    img_original = Image.fromarray(image)
    ow, oh = img_original.size
    ar = (ow / oh) if oh else 1.0
    if width_bricks / height_bricks > ar:
        height_bricks = max(height_bricks, int(width_bricks / ar))
    else:
        width_bricks = max(width_bricks, int(height_bricks * ar))

    brick = bricks_to_px(1)
    width_px = min(width_bricks * brick, WEBP_LIMIT_PX)
    height_px = min(height_bricks * brick, WEBP_LIMIT_PX)

    img_resized = img_original.resize((width_px, height_px), Image.BILINEAR)
    img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=blur_strength))
    arr_blurred = np.array(img_blurred.convert("RGB"))
    palette = get_dominant_colors(arr_blurred, n_colors)
    return arr_blurred, palette, width_bricks, height_bricks, brick

# =============================
# LABEL MAP (vectorized)
# =============================
def _block_mean_rgb(arr: np.ndarray, brick: int) -> np.ndarray:
    H, W, C = arr.shape
    Hb = (H // brick) * brick
    Wb = (W // brick) * brick
    cropped = arr[:Hb, :Wb]
    resh = cropped.reshape(Hb // brick, brick, Wb // brick, brick, C)
    return resh.mean(axis=(1, 3))  # (Ty,Tx,3)

def create_label_map(arr: np.ndarray, palette: np.ndarray, brick: int):
    block_means = _block_mean_rgb(arr, brick)       # (Ty,Tx,3)
    Ty, Tx, _ = block_means.shape
    bm = block_means.reshape(-1, 3).astype(np.float32)[:, None, :]
    pal = palette.astype(np.float32)[None, :, :]
    idxs = np.linalg.norm(bm - pal, axis=2).argmin(axis=1).reshape(Ty, Tx)  # 0..K-1
    label_map = (idxs + 1).astype(np.int32)
    color_to_id = {tuple(map(int, c)): i + 1 for i, c in enumerate(palette)}
    return label_map, color_to_id

# =============================
# ANTI-CONFETTI: majority filter + robust merge
# =============================
def majority_filter_labels(label_map: np.ndarray, window: int = 5, iters: int = 1) -> np.ndarray:
    """Simple mode filter over a kxk window to collapse single-tile zigzags."""
    from collections import Counter
    lm = label_map.copy()
    H, W = lm.shape
    r = window // 2
    for _ in range(iters):
        out = lm.copy()
        for y in range(H):
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            for x in range(W):
                x0, x1 = max(0, x - r), min(W, x + r + 1)
                vals = lm[y0:y1, x0:x1].ravel()
                # prefer current label on ties to avoid drift
                counts = Counter(vals)
                winner = max(counts.items(), key=lambda kv: (kv[1], kv[0] == lm[y, x]))[0]
                out[y, x] = winner
        lm = out
    return lm

def merge_components_by_border(label_map: np.ndarray,
                               min_area_tiles: int = 20,
                               max_iters: int = 5) -> np.ndarray:
    """
    Iteratively merge connected components (4-neigh) smaller than min_area_tiles
    into the neighboring component that shares the largest border length.
    """
    lm = label_map.copy()
    H, W = lm.shape
    four = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)

    for _ in range(max_iters):
        changed = False

        # Build global component map (component ids across all labels)
        comp_id = np.zeros_like(lm, dtype=np.int32)
        comp_label: List[int] = []
        next_id = 1
        for lbl in np.unique(lm):
            if lbl == 0: 
                continue
            cc, n = ndi.label(lm == lbl, structure=four)
            for k in range(1, n + 1):
                comp_id[cc == k] = next_id
                comp_label.append(int(lbl))
                next_id += 1

        if next_id == 1:
            break

        areas = np.bincount(comp_id.ravel(), minlength=next_id)

        # Visit small components; merge into neighbor with longest shared border
        for cid in range(1, next_id):
            if areas[cid] >= min_area_tiles:
                continue
            mask = (comp_id == cid)
            if not mask.any():
                continue
            border = ndi.binary_dilation(mask, structure=four) & (~mask)
            ys, xs = np.where(border)
            if ys.size == 0:
                continue

            neighbor_counts: Dict[int, int] = {}
            for y, x in zip(ys, xs):
                for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if 0 <= ny < H and 0 <= nx < W and not mask[ny, nx]:
                        nid = int(comp_id[ny, nx])
                        if nid > 0:
                            neighbor_counts[nid] = neighbor_counts.get(nid, 0) + 1
            if not neighbor_counts:
                continue

            target_cid = max(neighbor_counts.items(), key=lambda kv: (kv[1], areas[kv[0]]))[0]
            target_lbl = comp_label[target_cid - 1]
            lm[mask] = target_lbl
            changed = True

        if not changed:
            break

    return lm

def fill_small_holes(label_map: np.ndarray, max_hole_area_tiles: int = 20) -> np.ndarray:
    """Fill tiny enclosed pockets of other labels inside a large region."""
    lm = label_map.copy()
    four = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
    for lbl in np.unique(lm):
        if lbl == 0:
            continue
        mask = (lm == lbl)
        filled = ndi.binary_fill_holes(mask)
        holes = filled & (~mask)
        if not holes.any():
            continue
        cc, n = ndi.label(holes, structure=four)
        for k in range(1, n + 1):
            hole = (cc == k)
            if hole.sum() <= max_hole_area_tiles:
                lm[hole] = int(lbl)
    return lm

# =============================
# DRAW HELPERS
# =============================
def _number_positions(label_map: np.ndarray) -> List[Tuple[int, int, int]]:
    """Per connected component, place at the distance-transform maximum (deep interior)."""
    pos: List[Tuple[int, int, int]] = []
    for lbl in np.unique(label_map):
        if lbl == 0:
            continue
        mask = (label_map == lbl)
        cc, n = ndi.label(mask)
        for k in range(1, n + 1):
            comp = (cc == k)
            if comp.sum() == 0:
                continue
            dist = ndi.distance_transform_edt(comp)
            ty, tx = np.unravel_index(int(dist.argmax()), dist.shape)
            pos.append((int(lbl), int(ty), int(tx)))
    return pos

def draw_tiles(label_map: np.ndarray,
               id_to_color: Dict[int, Tuple[int, int, int]],
               brick: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    Ty, Tx = label_map.shape
    img = Image.new("RGB", (Tx * brick, Ty * brick), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for ty in range(Ty):
        y = ty * brick
        for tx in range(Tx):
            x = tx * brick
            color_id = int(label_map[ty, tx])
            d.rectangle([x, y, x + brick, y + brick], fill=id_to_color[color_id], outline=None)
    return img, d

def draw_outlines(label_map: np.ndarray, draw: ImageDraw.ImageDraw, brick: int) -> None:
    Ty, Tx = label_map.shape
    width = max(1, brick // 6)
    for ty in range(Ty):
        for tx in range(Tx):
            c = label_map[ty, tx]
            x, y = tx * brick, ty * brick
            if tx + 1 < Tx and label_map[ty, tx + 1] != c:
                draw.line([(x + brick, y), (x + brick, y + brick)], fill=(40, 40, 40), width=width)
            if ty + 1 < Ty and label_map[ty + 1, tx] != c:
                draw.line([(x, y + brick), (x + brick, y + brick)], fill=(40, 40, 40), width=width)

def draw_numbers(label_map: np.ndarray, draw: ImageDraw.ImageDraw, brick: int) -> None:
    # No white halo / stroke — just crisp black numerals centered.
    font = ImageFont.load_default()
    for lbl, ty, tx in _number_positions(label_map):
        cx = int(tx * brick + brick / 2)
        cy = int(ty * brick + brick / 2)
        draw.text((cx, cy), str(lbl), fill=(0, 0, 0), font=font, anchor="mm")

def create_color_legend(color_to_id: Dict[Tuple[int, int, int], int]) -> Image.Image:
    items = sorted(color_to_id.items(), key=lambda kv: kv[1])
    square = 20
    w = 260
    h = (len(items) + 1) * (square + 6)
    legend = Image.new("RGB", (w, min(h, 16000)), (255, 255, 255))
    d = ImageDraw.Draw(legend)
    y = 6
    for (rgb, idx) in items:
        d.rectangle([6, y, 6 + square, y + square], fill=rgb, outline=(0, 0, 0))
        d.text((6 + square + 8, y + 4), f"#{idx} RGB {rgb}", fill=(0, 0, 0))
        y += square + 6
    return legend

# =============================
# PIPELINES
# =============================
def generate_pixel_art_output(
    arr_blurred: np.ndarray,
    color_centers: np.ndarray,
    brick: int,
    width_px: int,
    height_px: int,
    mode_choice: str = "Color-Book",
    majority_window: int = 5,
    majority_iters: int = 1,
    min_area_tiles: int = 20,
):
    label_map, color_to_id = create_label_map(arr_blurred, color_centers, brick)

    # Anti‑confetti stack
    label_map = majority_filter_labels(label_map, window=majority_window, iters=majority_iters)
    label_map = merge_components_by_border(label_map, min_area_tiles=min_area_tiles, max_iters=5)
    label_map = fill_small_holes(label_map, max_hole_area_tiles=min_area_tiles)
    label_map = merge_components_by_border(label_map, min_area_tiles=min_area_tiles, max_iters=2)

    id_to_color = {v: k for k, v in color_to_id.items()}
    output, draw = draw_tiles(label_map, id_to_color, brick)

    if mode_choice == "Color-Book":
        draw_outlines(label_map, draw, brick)
        draw_numbers(label_map, draw, brick)
    elif mode_choice == "Mozaik":
        Ty, Tx = label_map.shape
        for ty in range(Ty):
            for tx in range(Tx):
                x, y = tx * brick, ty * brick
                draw.rectangle([x, y, x + brick, y + brick], outline=(0, 0, 0))

    outlines_only = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    d2 = ImageDraw.Draw(outlines_only)
    draw_outlines(label_map, d2, brick)

    legend = create_color_legend(color_to_id)
    return output, outlines_only, label_map, legend, color_to_id

def generate_natural_output(
    arr_blurred: np.ndarray,
    label_map_tiles: np.ndarray,
    color_to_id: Dict[Tuple[int, int, int], int],
    color_centers: np.ndarray,
    brick: int,
    width_px: int,
    height_px: int,
    canny_low: int = 10,
    canny_high: int = 40,
):
    # Per‑pixel label map using same palette
    flat = arr_blurred.reshape(-1, 3).astype(np.float32)[:, None, :]
    pal = color_centers.astype(np.float32)[None, :, :]
    idxs = np.linalg.norm(flat - pal, axis=2).argmin(axis=1).reshape(height_px, width_px)
    label_map_smooth = (idxs + 1).astype(np.int32)

    id_to_color = {v: k for k, v in color_to_id.items()}
    rgb = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    for lid, col in id_to_color.items():
        rgb[label_map_smooth == lid] = col

    if cv2 is not None:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(g, threshold1=canny_low, threshold2=canny_high)
        rgb[edges > 0] = (0, 0, 0)

    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    # numbers from the TILE map (so they don’t overlap edges)
    draw_numbers(label_map_tiles, dr, brick)

    legend = create_color_legend(color_to_id)
    return img, legend

# =============================
# PUBLIC API for UI
# =============================
def run_mozaik(image_np: np.ndarray, n_colors: int, blur_strength: float):
    arr, palette, wb, hb, brick = preprocess_image(image_np, n_colors, blur_strength)
    width_px, height_px = wb * brick, hb * brick
    pix, _out, label_map, legend, color_to_id = generate_pixel_art_output(
        arr, palette, brick, width_px, height_px, mode_choice="Mozaik"
    )
    total_tiles = (width_px // brick) * (height_px // brick)
    return pix, total_tiles, legend, {
        "arr_blurred": arr,
        "color_centers": palette,
        "label_map": label_map,
        "brick_size": brick,
        "width_px": width_px,
        "height_px": height_px,
        "color_to_id": color_to_id,
    }

def run_colorbook(image_np: np.ndarray, n_colors: int, blur_strength: float, mode: str,
                  canny_low: int, canny_high: int):
    arr, palette, wb, hb, brick = preprocess_image(image_np, n_colors, blur_strength)
    width_px, height_px = wb * brick, hb * brick

    pix, _out, label_map, legend, color_to_id = generate_pixel_art_output(
        arr, palette, brick, width_px, height_px, mode_choice="Color-Book"
    )

    if mode == "Pixel-Art":
        total = (width_px // brick) * (height_px // brick)
        return pix, total, legend, {
            "arr_blurred": arr, "color_centers": palette, "label_map": label_map,
            "brick_size": brick, "width_px": width_px, "height_px": height_px,
            "color_to_id": color_to_id,
        }
    else:
        nat, legend = generate_natural_output(
            arr, label_map, color_to_id, palette, brick, width_px, height_px,
            canny_low=canny_low, canny_high=canny_high
        )
        total = (width_px // brick) * (height_px // brick)
        return nat, total, legend, {
            "arr_blurred": arr, "color_centers": palette, "label_map": label_map,
            "brick_size": brick, "width_px": width_px, "height_px": height_px,
            "color_to_id": color_to_id,
        }

__all__ = [
    "run_mozaik", "run_colorbook",
    "create_label_map", "majority_filter_labels",
    "merge_components_by_border", "fill_small_holes",
    "draw_outlines", "draw_numbers", "create_color_legend"
]
