import numpy as np
import streamlit as st

def score_chip(label, score):
    s = int(round(score))
    st.markdown(f"<span class='score-chip'>{label}: <strong>{s}</strong>/100</span>", unsafe_allow_html=True)

def verdict(label, score, invert=False):
    v = int(round(score))
    if invert:
        v = 100 - v
    if v < 35:
        css = "badge-good"
        text = "Low"
    elif v < 70:
        css = "badge-warn"
        text = "Medium"
    else:
        css = "badge-high"
        text = "High"
    st.markdown(f"<span class='result-badge {css}'>{label}: {text}</span>", unsafe_allow_html=True)

def section_header(title, subtitle=None):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
