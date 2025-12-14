# Portfolio Website — CLEAN VERSION (No Google / ML / Scraping Code)
# Andrew Clark | Finance & Economics Portfolio
# Run with: streamlit run Portfolio_Website_Cleaned.py

import streamlit as st
import requests
import base64

# ============================================================================
# CONFIGURATION & THEME
# ============================================================================
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
    "shadow": "rgba(30, 58, 138, 0.08)"
}

st.set_page_config(
    page_title="Andrew Clark | Finance & Economics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
def load_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    * {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Playfair Display', serif; }}

    .main {{ background-color: {THEME['background']}; }}

    .custom-card {{
        background: {THEME['card_bg']};
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px {THEME['shadow']};
        border-left: 4px solid {THEME['primary']};
    }}

    .tag {{
        display: inline-block;
        background: {THEME['primary']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        margin: 0.2rem;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {THEME['primary']} 0%, #0F172A 100%);
    }}
    [data-testid="stSidebar"] * {{ color: white !important; }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA
# ============================================================================
PERSONAL_INFO = {
    "name": "Andrew Clark",
    "title": "Finance & Economics Student",
    "tagline": "Lehigh University '27 | Finance & Economics Double Major",
    "subtitle": "Financial analysis, econometrics, and data science",
    "email": "2andrewclark@gmail.com",
    "phone": "(203) 451-8937",
    "github": "https://github.com/andrewclark",
    "linkedin": "https://www.linkedin.com/in/andrew-clark27",
    "location": "Bethlehem, PA / Southbury, CT"
}

PROJECTS = [
    {
        "title": "Boston Housing Remodeling Impact Analysis",
        "description": "Econometric study using hedonic pricing models on 68k+ observations. Found statistically significant renovation premiums.",
        "tech_stack": ["R", "Stata", "Econometrics", "Fixed Effects"],
        "code_snippet": "# Hedonic regression with neighborhood fixed effects"
    },
    {
        "title": "Financial Data Analytics Platform",
        "description": "SQL-based analytics tools for multi-entity financial reporting at Tusk Strategies.",
        "tech_stack": ["SQL", "Retool", "SAGE Intacct"],
        "code_snippet": "SELECT entity_name, SUM(cash_balance) FROM financials GROUP BY entity_name;"
    }
]

SKILLS = {
    "Python": 85,
    "R": 90,
    "SQL": 85,
    "Stata": 88,
    "Financial Modeling": 95
}

# ============================================================================
# HELPERS
# ============================================================================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ============================================================================
# SECTIONS
# ============================================================================
def render_home():
    st.markdown(f"""
    <div class="custom-card" style="text-align:center;">
        <h1>{PERSONAL_INFO['name']}</h1>
        <h3>{PERSONAL_INFO['title']}</h3>
        <p>{PERSONAL_INFO['tagline']}</p>
        <p><em>{PERSONAL_INFO['subtitle']}</em></p>
    </div>
    """, unsafe_allow_html=True)


def render_projects():
    st.title("💻 Projects")
    for project in PROJECTS:
        st.markdown(f"""
        <div class="custom-card">
            <h3>{project['title']}</h3>
            <p>{project['description']}</p>
            {''.join([f'<span class="tag">{t}</span>' for t in project['tech_stack']])}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("View Code Snippet"):
            st.code(project['code_snippet'])


def render_skills():
    st.title("🛠 Skills")
    for skill, level in SKILLS.items():
        st.progress(level / 100, text=f"{skill} — {level}%")


def render_contact():
    st.title("📬 Contact")
    st.markdown(f"""
    <div class="custom-card">
        <p><strong>Email:</strong> {PERSONAL_INFO['email']}</p>
        <p><strong>Phone:</strong> {PERSONAL_INFO['phone']}</p>
        <p><strong>LinkedIn:</strong> <a href="{PERSONAL_INFO['linkedin']}" target="_blank">Profile</a></p>
        <p><strong>GitHub:</strong> <a href="{PERSONAL_INFO['github']}" target="_blank">Repo</a></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    load_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Projects", "Skills", "Contact"])

    if page == "Home":
        render_home()
    elif page == "Projects":
        render_projects()
    elif page == "Skills":
        render_skills()
    elif page == "Contact":
        render_contact()


if __name__ == "__main__":
    main()

