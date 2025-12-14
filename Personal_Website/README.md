Andrew Clark — Portfolio Streamlit App

Quick start

1. Add your résumé PDF to `assets/Andrew_Clark_Resume.pdf`.
2. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate   # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run locally:

```bash
streamlit run app.py
```

Notes
- The app keeps all Streamlit UI, custom CSS, Lottie animations, and data-driven layout.
- Colab-specific artifacts were removed so the project is ready for GitHub and Streamlit Sharing (Streamlit Community Cloud).
- To deploy: push to GitHub and connect the repository to Streamlit Community Cloud; ensure `requirements.txt` contains all optional libraries you use.
