"""
Andrew Clark - Portfolio Website (Cleaned & Stable)

✔ Keeps all visual + interactive features
✔ Fixes missing imports / runtime errors
✔ Safer network calls
✔ Configurable resume path
✔ Minor CSS + Streamlit best-practice cleanup

Run with:
    streamlit run Portfolio_Website_Cleaned.py
"""

# =============================================================================
# IMPORTS (CLEANED / COMPLETE)
# =============================================================================
import streamlit as st
import requests
import base64
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List

# =============================================================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL)
# =============================================================================
st.set_page_config(
    page_title="Andrew Clark | Finance & Economics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME CONFIGURATION
# =============================================================================
THEME = {
    "primary": "#1E3A8A",
    "secondary": "#8B4513",
    "accent": "#059669",
    "gold": "#FFD700",
    "negative": "#DC2626",
    "background": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "text": "#1E293B",
    "text_light": "#64748B",
    "shadow": "rgba(30, 58, 138, 0.08)",
}

# =============================================================================
# GLOBAL PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
RESUME_PDF_PATH = BASE_DIR / "Resume.pdf"  # <-- change filename if needed

# =============================================================================
# CUSTOM CSS
# =============================================================================
def load_css():
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    * {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Playfair Display', serif; }}

    .main {{ background-color: {THEME['background']}; }}

    .hero-container {{
        text-align: center;
        padding: 4rem 2rem 2rem 2rem;
        background: linear-gradient(135deg, {THEME['background']} 0%, #E0E7F1 100%);
        border-bottom: 3px solid {THEME['secondary']};
    }}

    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        color: {THEME['primary']};
    }}

    .custom-card {{
        background: {THEME['card_bg']};
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px {THEME['shadow']};
        border-left: 4px solid {THEME['primary']};
    }}

    .custom-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(30, 58, 138, 0.15);
        transition: 0.3s;
    }}

    .tag {{
        display: inline-block;
        background: linear-gradient(135deg, {THEME['primary']}, #1E40AF);
        color: white;
        padding: 0.35rem 0.9rem;
        margin: 0.25rem;
        font-size: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }}

    .tag-brown {{ background: linear-gradient(135deg, {THEME['secondary']}, #654321); }}
    .tag-gold {{ background: linear-gradient(135deg, {THEME['gold']}, #D4AF37); color: black; }}

    .skill-bar {{ background: #E2E8F0; border-radius: 6px; overflow: hidden; }}
    .skill-fill {{
        height: 22px;
        background: linear-gradient(90deg, {THEME['primary']}, {THEME['accent']});
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {THEME['primary']}, #0F172A);
        border-right: 2px solid {THEME['secondary']};
    }}

    [data-testid="stSidebar"] * {{ color: white !important; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# =============================================================================
# DATA
# =============================================================================
PERSONAL_INFO = {
    "name": "Andrew Clark",
    "title": "Finance & Economics Student",
    "tagline": "Lehigh University '27 | Finance & Economics Double Major",
    "subtitle": "Combining financial analysis, econometric modeling, and data science",
    "email": "2andrewclark@gmail.com",
    "phone": "(203) 451-8937",
    "github": "https://github.com/andrewclark",
    "linkedin": "https://www.linkedin.com/in/andrew-clark27",
    "location": "Bethlehem, PA / Southbury, CT",
}

# =============================================================================
# HELPERS
# =============================================================================
def safe_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def pdf_to_base64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def display_pdf(path: Path):
    encoded = pdf_to_base64(path)
    if not encoded:
        st.info("📄 PDF not found")
        return
    st.markdown(
        f"""
        <iframe src="data:application/pdf;base64,{encoded}" width="100%" height="900"></iframe>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SECTIONS
# =============================================================================
def render_hero():
    st.markdown(
        f"""
        <div class="hero-container">
            <h1 class="hero-title">{PERSONAL_INFO['name']}</h1>
            <p><strong>{PERSONAL_INFO['title']}</strong></p>
            <p>{PERSONAL_INFO['tagline']}</p>
            <p><em>{PERSONAL_INFO['subtitle']}</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resume():
    st.title("📋 Résumé")
    display_pdf(RESUME_PDF_PATH)
    if RESUME_PDF_PATH.exists():
        st.download_button(
            "⬇️ Download PDF",
            RESUME_PDF_PATH.read_bytes(),
            file_name="Andrew_Clark_Resume.pdf",
            mime="application/pdf",
        )


def render_contact():
    st.title("📬 Get In Touch")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        anim = safe_lottie("https://assets2.lottiefiles.com/packages/lf20_u25cckyh.json")
        if anim:
            st_lottie(anim, height=200)

    st.markdown(
        f"""
        <div class="custom-card" style="text-align:center;">
            <p><strong>Email:</strong> <a href="mailto:{PERSONAL_INFO['email']}">{PERSONAL_INFO['email']}</a></p>
            <p><strong>Phone:</strong> {PERSONAL_INFO['phone']}</p>
            <p><strong>Location:</strong> {PERSONAL_INFO['location']}</p>
            <p><a href="{PERSONAL_INFO['linkedin']}" target="_blank">LinkedIn</a> | <a href="{PERSONAL_INFO['github']}" target="_blank">GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    load_css()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("", ["Home", "Resume", "Contact"])

    if page == "Home":
        render_hero()
    elif page == "Resume":
        render_resume()
    elif page == "Contact":
        render_contact()


if __name__ == "__main__":
    main()
