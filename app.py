import io
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.analyzers import aggregate_scores
from utils.ui import score_chip, verdict, section_header
from utils.clip_utils import analyze_outfit   # NEW ✅

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Outfit Vibe Rater", page_icon="👗", layout="wide")

# -------------------------
# Load CSS (if exists)
# -------------------------
if os.path.exists("styles.css"):
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
col_logo, col_title = st.columns([1, 5], vertical_alignment="center")
with col_logo:
    if os.path.exists("assets/logo.svg"):
        st.image("assets/logo.svg", width=72)
with col_title:
    st.title("✨ Outfit Vibe Rater")
    st.caption("Upload/capture an outfit photo to get instant vibe scores — complexity, pattern density, fit tightness, and more.")

# -------------------------
# Sidebar (choices + options)
# -------------------------
with st.sidebar:
    st.header("Upload")
    img_choice = st.radio("Input method", ["Upload", "Camera"], horizontal=True)

    st.divider()
    st.header("Options")
    show_debug = st.checkbox("Show analysis maps (edges/entropy/mask)")
    presentation_mode = st.checkbox("Presentation Mode (smooth & adjusted scores)", value=True)
    if presentation_mode:
        st.caption("Smooths & boosts scores for demo-friendly results.")
        adjust_strength = st.slider("Adjustment strength", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    else:
        adjust_strength = 0.0

# -------------------------
# Main area: Input widget (centered)
# -------------------------
st.markdown("## 📸 Upload or Capture Outfit")
if img_choice == "Upload":
    img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
else:
    img_file = st.camera_input("Capture outfit photo", key="camera")

if img_file is None:
    st.info("⬅️ Upload or capture an outfit photo to get started.")
    st.stop()

# -------------------------
# Load and score image
# -------------------------
image = Image.open(img_file).convert("RGB")
img_np = np.array(image)

raw_scores = aggregate_scores(img_np)   # heuristic analyzer
clip_scores = analyze_outfit(img_file)  # ✅ pass file object, not PIL image


# -------------------------
# Normalizer / Presentation helper
# -------------------------
def detect_red_dominance(rgb_image):
    try:
        arr = rgb_image.astype(np.float32)
        r = arr[:, :, 0].mean()
        g = arr[:, :, 1].mean()
        b = arr[:, :, 2].mean()
        total = (r + g + b) + 1e-6
        return (r / total) > 0.38
    except Exception:
        return False

def _adjust_by_rules(scores):
    color = float(scores.get("colorfulness", 0.0))
    complexity = float(scores.get("complexity", 0.0))
    pattern = float(scores.get("pattern_density", 0.0))
    fit = float(scores.get("fit_tightness", 0.0))
    bold = float(scores.get("boldness", 0.0))

    adj_pattern = min(100.0, pattern * 1.6)
    adj_complexity = min(100.0, complexity * 1.3)
    adj_bold = min(100.0, bold * 1.15 + color * 0.12 + adj_pattern * 0.06)
    adj_color = min(100.0, color * 1.05 + adj_bold * 0.02)
    adj_fit = float(np.clip(fit * 0.95 + 4.0, 0, 100))

    adj_vibe = float(np.clip(0.45 * adj_pattern + 0.40 * adj_complexity + 0.15 * adj_color, 0, 100))

    return {
        "colorfulness": adj_color,
        "complexity": adj_complexity,
        "pattern_density": adj_pattern,
        "fit_tightness": adj_fit,
        "boldness": adj_bold,
        "vibe_maximalism": adj_vibe
    }

def normalize_scores(scores, strength=0.6):
    if strength <= 0:
        return {k: float(v) for k, v in scores.items() if k in ["colorfulness","complexity","pattern_density","fit_tightness","boldness","vibe_maximalism"]}

    adjusted = _adjust_by_rules(scores)
    processed_rgb = scores.get("processed_rgb")
    if processed_rgb is not None and detect_red_dominance(processed_rgb):
        adjusted["boldness"] = min(100.0, adjusted["boldness"] + 8.0)
        adjusted["colorfulness"] = min(100.0, adjusted["colorfulness"] + 4.0)

    blended = {}
    keys = ["vibe_maximalism","pattern_density","fit_tightness","colorfulness","complexity","boldness"]
    for k in keys:
        raw = float(scores.get(k, 0.0))
        adj = float(adjusted.get(k, raw))
        blended[k] = float(np.clip((1.0 - strength) * raw + strength * adj, 0.0, 100.0))
    return blended

final_scores = normalize_scores(raw_scores, strength=adjust_strength if presentation_mode else 0.0)

# -------------------------
# UI: Left + Right columns
# -------------------------
left, right = st.columns([1, 1])

with left:
    section_header("Original")
    st.image(image, use_container_width=True)

    section_header("Key Scores")
    st.progress(int(final_scores["vibe_maximalism"]), text=f"🎨 Maximalist Vibe: {final_scores['vibe_maximalism']:.1f}")
    st.progress(int(final_scores["pattern_density"]), text=f"🌀 Pattern Density: {final_scores['pattern_density']:.1f}")
    st.progress(int(final_scores["fit_tightness"]), text=f"📏 Fit Tightness: {final_scores['fit_tightness']:.1f}")
    st.progress(int(final_scores["colorfulness"]), text=f"🌈 Colorfulness: {final_scores['colorfulness']:.1f}")
    st.progress(int(final_scores["complexity"]), text=f"⚡ Visual Complexity: {final_scores['complexity']:.1f}")
    st.progress(int(final_scores["boldness"]), text=f"🔥 Boldness: {final_scores['boldness']:.1f}")

    # NEW: Show CLIP vibes
    section_header("CLIP Outfit Vibe (AI)")
    st.write({k: f"{v:.2f}" for k, v in clip_scores.items()})

with right:
    section_header("Interpretation")
    def interpret(scores):
        lines = []
        v = scores["vibe_maximalism"]
        if v < 30: lines.append("Outfit looks **minimalist** and clean.")
        elif v > 70: lines.append("Outfit has a **maximalist vibe** — bold and expressive.")
        else: lines.append("Outfit balances minimal and maximal elements.")
        p = scores["pattern_density"]
        if p < 30: lines.append("Patterns are **sparse or subtle**.")
        elif p > 70: lines.append("Patterns are **dense and eye-catching**.")
        else: lines.append("Moderate pattern usage.")
        f = scores["fit_tightness"]
        if f > 70: lines.append("Fit appears **tight/form-fitting**.")
        elif f < 30: lines.append("Fit appears **loose/relaxed**.")
        else: lines.append("Fit is **balanced**.")
        c = scores["colorfulness"]
        if c > 60: lines.append("Colors are **bright and lively**.")
        elif c < 20: lines.append("Outfit is **muted/monochrome**.")
        else: lines.append("Color palette is **moderately vibrant**.")
        return lines

    for li in interpret(final_scores):
        st.markdown(f"- {li}")
    st.caption("Note: Fit tightness uses a silhouette heuristic.")

    section_header("Details")
    score_chip("Colorfulness", final_scores["colorfulness"])
    score_chip("Complexity", final_scores["complexity"])
    score_chip("Boldness", final_scores["boldness"])

    # ... (rest of your report export and charts remain unchanged)
