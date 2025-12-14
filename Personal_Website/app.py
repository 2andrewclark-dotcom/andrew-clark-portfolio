"""
Andrew Clark - Portfolio Website
Professional portfolio showcasing finance, economics, and quantitative research

Requirements (requirements.txt):
streamlit>=1.28.0
streamlit-lottie>=0.0.5
requests>=2.31.0
Pillow>=10.0.0

Run with: streamlit run app.py
"""

import streamlit as st
import requests
from streamlit_lottie import st_lottie
import base64

# ============================================================================
# CONFIGURATION & THEME
# ============================================================================
# Finance / Economics Theme - Deep navy, gold, Lehigh brown, market green
THEME = {
    "primary": "#1E3A8A",       # Deep navy blue
    "secondary": "#8B4513",     # Lehigh Brown
    "accent": "#059669",        # Market green
    "gold": "#FFD700",          # Gold accent
    "negative": "#DC2626",      # Market red
    "background": "#F8FAFC",    # Cool gray background
    "card_bg": "#FFFFFF",       # Clean white cards
    "text": "#1E293B",          # Deep slate text
    "text_light": "#64748B",    # Muted slate
    "shadow": "rgba(30, 58, 138, 0.08)"
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Andrew Clark | Finance & Economics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLES
# ============================================================================
def load_css():
    """Load custom CSS for animations, cards, and styling (keeps navy & gold theme)."""
    css_code = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        * {{
            font-family: 'Inter', sans-serif;
        }}

        h1, h2, h3, .hero-title {{
            font-family: 'Playfair Display', serif;
        }}

        .main {{
            background-color: {THEME['background']};
        }}

        .hero-container {{
            text-align: center;
            padding: 4rem 2rem 2rem 2rem;
            animation: fadeInDown 0.8s ease-out;
            background: linear-gradient(135deg, {THEME['background']} 0%, #E0E7F1 100%);
            border-bottom: 3px solid {THEME['secondary']};
        }}

        .hero-title {{
            font-size: 4rem;
            font-weight: 800;
            color: {THEME['primary']};
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
        }}

        .hero-tagline {{
            font-size: 1.4rem;
            color: {THEME['text']};
            margin-bottom: 2rem;
            font-weight: 500;
            letter-spacing: 0.03em;
        }}

        .hero-subtitle {{
            font-size: 0.95rem;
            color: {THEME['text']};
            font-style: italic;
            border-left: 4px solid {THEME['secondary']};
            padding-left: 1rem;
            margin: 1rem auto;
            max-width: 700px;
            text-align: left;
            font-weight: 500;
        }}

        .custom-card {{
            background: {THEME['card_bg']};
            border-radius: 8px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px {THEME['shadow']};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeInUp 0.6s ease-out;
            border-left: 4px solid {THEME['primary']};
            position: relative;
        }}

        .custom-card::before {{
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, transparent 50%, {THEME['secondary']}15 50%);
            border-top-right-radius: 8px;
        }}

        .custom-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(30, 58, 138, 0.12);
            border-left-color: {THEME['secondary']};
        }}

        .project-card {{
            position: relative;
            overflow: hidden;
        }}

        .tag {{
            display: inline-block;
            background: linear-gradient(135deg, {THEME['primary']}, #1E40AF);
            color: white;
            padding: 0.35rem 1rem;
            border-radius: 4px;
            margin: 0.25rem;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            transition: all 0.2s;
            border: 1px solid rgba(255,255,255,0.2);
        }}

        .tag:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        }}

        .tag-brown {{
            background: linear-gradient(135deg, {THEME['secondary']}, #654321);
        }}

        .custom-button {{
            background: linear-gradient(135deg, {THEME['primary']}, #1E40AF);
            color: white;
            padding: 0.75rem 2rem;
            border-radius: 4px;
            text-decoration: none;
            display: inline-block;
            margin: 0.5rem;
            transition: all 0.3s;
            border: 2px solid {THEME['primary']};
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-size: 0.85rem;
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.2);
        }}

        .custom-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(30, 58, 138, 0.3);
            background: linear-gradient(135deg, {THEME['secondary']}, #654321);
            border-color: {THEME['secondary']};
        }}

        .skill-bar {{
            background: #E2E8F0;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
            border: 1px solid #CBD5E1;
        }}

        .skill-fill {{
            height: 24px;
            background: linear-gradient(90deg, {THEME['primary']}, {THEME['accent']});
            border-radius: 4px;
            transition: width 1.5s ease-out;
            animation: fillBar 1.5s ease-out;
            position: relative;
            overflow: hidden;
        }}

        .skill-fill::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }}

        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fillBar {{
            from {{ width: 0; }}
        }}

        .section-content {{
            animation: fadeIn 0.5s ease-out;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {THEME['primary']} 0%, #0F172A 100%);
            border-right: 2px solid {THEME['secondary']};
        }}

        /* Make sidebar text white for readability */
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        .stat-box {{
            background: linear-gradient(135deg, {THEME['primary']}, #1E40AF);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 3px solid {THEME['secondary']};
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        }}

        .stat-number {{
            font-size: 2.5rem;
            font-weight: 700;
            font-family: 'Playfair Display', serif;
            color: {THEME['gold']};
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }}

        .stat-label {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.5rem;
            color: #FFFFFF;
            font-weight: 600;
        }}

        .divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent, {THEME['secondary']}, transparent);
            margin: 2rem 0;
        }}

        .pdf-container {{
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 16px {THEME['shadow']};
        }}

        @media (max-width: 768px) {{
            .hero-title {{
                font-size: 2rem;
            }}
            .hero-tagline {{
                font-size: 1rem;
            }}
        }}
        </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# ============================================================================
# PERSONAL DATA (including twitter to avoid KeyError)
# ============================================================================
PERSONAL_INFO = {
    "name": "Andrew Clark",
    "title": "Finance & Economics Student",
    "tagline": "Lehigh University '27 | Finance & Economics Double Major",
    "subtitle": "Combining financial analysis, econometric modeling, and data science to solve complex problems in finance and economics",
    "email": "2andrewclark@gmail.com",
    "phone": "(203) 451-8937",
    "github": "https://github.com/andrewclark",
    "linkedin": "https://www.linkedin.com/in/andrew-clark27",
    "twitter": "",
    "location": "Bethlehem, PA / Southbury, CT"
}

# Coding Projects (kept from original)
PROJECTS = [
    {
        "title": "Boston Housing Remodeling Impact Analysis",
        "description": "Econometric study examining how property renovations affect housing assessment values in Greater Boston. Built hedonic pricing models with 68,169 observations, controlling for neighborhood fixed effects. Found that kitchen remodels yield 7.34% value premiums while bathroom remodels yield 5.86% (statistically significant difference). Model achieved R² of 84.95%.",
        "tech_stack": ["R", "Stata", "Econometrics", "Hedonic Pricing", "Fixed Effects"],
        "github": "",
        "demo": "",
        "image": "",
        "code_snippet": """# Data cleaning and variable creation in R
boston_clean_var <- boston_clean %>%
  mutate(
    remodeled = if_else(YR_REMODEL > 2005, 1, 0, missing = 0),
    ln_value = log(TOTAL_VALUE),
    ln_area = log(LIVING_AREA),
    age = 2025 - YR_BUILT,
    bthrm_remodeled = ifelse(
      BTHRM_STYLE1 != "N - No Remodeling" &
      !is.na(BTHRM_STYLE1), 1, 0),
    ktch_remodeled = ifelse(
      KITCHEN_STYLE1 != "N - No Remodeling" &
      !is.na(KITCHEN_STYLE1), 1, 0)
  )"""
    },
    {
        "title": "Financial Data Analytics Platform (Tusk Strategies)",
        "description": "Built SQL-based data aggregation tools using Retool to provide real-time financial summaries across 8 different entities including VC funds, consulting services, and 501(c) organizations. Automated cash flow modeling and receivables tracking, improving financial visibility for firm management.",
        "tech_stack": ["SQL", "Retool", "SAGE Intacct", "Financial Modeling"],
        "github": "",
        "demo": "",
        "image": "",
        "code_snippet": """-- SQL query for entity cash flow aggregation
SELECT
    entity_name,
    SUM(receivables) as total_receivables,
    SUM(payables) as total_payables,
    SUM(cash_balance) as current_cash,
    AVG(collection_period) as avg_collection_days
FROM
    financial_transactions
WHERE
    transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY
    entity_name
ORDER BY
    total_receivables DESC;"""
    },
     {
        "title": "Alternative Data Analyzer for Energy Companies",
    "description": "This Python script implements a comprehensive Alternative Data Analyzer (ADA) system designed to generate trading signals for energy sector stocks. The system aggregates and analyzes seven distinct data sources: stock price momentum, search trends (via SerpApi/Google Trends), social sentiment (Finviz scraping), news sentiment (FinBERT via Hugging Face's 'transformers'), Baker Hughes oil rig count, EIA inventory, and STEO government forecasts. It combines these features using a weighted scoring model to produce actionable BUY/HOLD/SELL recommendations, and exports the analysis to an interactive Streamlit dashboard.",
    "tech_stack": ["Python", "Pandas", "NumPy", "yfinance", "requests", "BeautifulSoup4",  "transformers (FinBERT)","Plotly", "Streamlit"],
    "github": "https://alt-data-analyzer.streamlit.app/",
    "demo": "",
    "image": "",
        "code_snippet": "# ============================================================================\n# DATA COLLECTION MODULE 4: News Sentiment (UPGRADED TO FINBERT)\n# ============================================================================\n\nclass PerTickerNewsSentimentCollector:\n    \"\"\"\n    News sentiment analyzer using FinBERT (financial domain-specific model)\n    Much more accurate than VADER for financial text\n    \"\"\"\n\n    def __init__(self, cache: CacheManager):\n        self.cache = cache\n        self.vader = None  # Keep as fallback\n        self.finbert_model = None\n        self.finbert_tokenizer = None\n\n        # Try to load FinBERT (requires transformers library)\n        self._load_finbert()\n\n        # Fallback to VADER if FinBERT fails\n        if self.finbert_model is None:\n            try:\n                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer\n                self.vader = SentimentIntensityAnalyzer()\n                print(\"  ⚠️  Using VADER (FinBERT not available)\")\n            except ImportError:\n                print(\"  ⚠️  Neither FinBERT nor VADER available\")\n\n    def _load_finbert(self):\n        \"\"\"Load FinBERT model (called during init)\"\"\"\n        try:\n            from transformers import AutoTokenizer, AutoModelForSequenceClassification\n            import torch\n\n            print(\" Loading FinBERT model (this may take 45ish seconds)...\")\n\n            # Use ProsusAI/finbert - best financial sentiment model\n            model_name = \"ProsusAI/finbert\"\n\n            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)\n            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)\n\n            # Set to eval mode\n            self.finbert_model.eval()\n\n            print(\"  ✓ FinBERT loaded successfully\")\n\n        except ImportError:\n            print(\"  ⚠️  transformers library not installed\")\n            print(\"      Install with: !pip install transformers torch\")\n        except Exception as e:\n            print(f\"  ⚠️  Could not load FinBERT: {e}\")\n\n    def analyze_headline_detailed(self, headline: str) -> Dict:\n        \"\"\"\n        Perform sentiment analysis on a single headline\n        Uses FinBERT if available, falls back to VADER\n        \"\"\"\n        # Try FinBERT first\n        if self.finbert_model is not None:\n            return self._analyze_with_finbert(headline)\n\n        # Fallback to VADER\n        elif self.vader is not None:\n            vader_scores = self.vader.polarity_scores(headline)\n            return {\n                'compound': vader_scores['compound'],\n                'pos': vader_scores['pos'],\n                'neg': vader_scores['neg'],\n                'neu': vader_scores['neu'],\n                'headline': headline,\n                'method': 'vader'\n            }\n\n        # No sentiment analyzer available\n        else:\n            return {\n                'compound': 0.0,\n                'pos': 0.33,\n                'neg': 0.33,\n                'neu': 0.34,\n                'headline': headline,\n                'method': 'none'\n            }\n\n    def _analyze_with_finbert(self, text: str) -> Dict:\n        \"\"\"\n        Use FinBERT model for sentiment analysis\n        Returns scores in same format as VADER for compatibility\n        \"\"\"\n        try:\n            import torch\n\n            # Tokenize input\n            inputs = self.finbert_tokenizer(\n                text,\n                return_tensors=\"pt\",\n                truncation=True,\n                max_length=512,\n                padding=True\n            )\n\n            # Get predictions\n            with torch.no_grad():\n                outputs = self.finbert_model(**inputs)\n                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)\n\n            # FinBERT outputs: [positive, negative, neutral]\n            positive = predictions[0][0].item()\n            negative = predictions[0][1].item()\n            neutral = predictions[0][2].item()\n\n            # Convert to compound score (like VADER)\n            # Positive - Negative, scaled to [-1, 1]\n            compound = positive - negative\n\n            return {\n                'compound': compound,\n                'pos': positive,\n                'neg': negative,\n                'neu': neutral,\n                'headline': text,\n                'method': 'finbert'\n            }\n\n        except Exception as e:\n            print(f\"  ⚠️  FinBERT analysis failed: {e}\")\n            return {\n                'compound': 0.0,\n                'pos': 0.33,\n                'neg': 0.33,\n                'neu': 0.34,\n                'headline': text,\n                'method': 'error'\n            }\n\n    def generate_google_news_rss_url(self, ticker: str, company_name: str, lookback_days: int = 365) -> str:\n        \"\"\"Generates a Google News RSS feed URL for a given ticker and lookback period.\"\"\"\n        end_date = datetime.now()\n        start_date = end_date - timedelta(days=lookback_days)\n\n        end_date_str = end_date.strftime('%Y-%m-%d')\n        start_date_str = start_date.strftime('%Y-%m-%d')\n\n        company_name_url = company_name.replace(' ', '+')\n        rss_url = f'https://news.google.com/rss/search?q={company_name_url}+({ticker})+after:{start_date_str}+before:{end_date_str}&ceid=US:en&hl=en-US&gl=US'\n        return rss_url\n\n    def get_company_name(self, ticker: str) -> str:\n        \"\"\"Get company name from ticker using yfinance\"\"\"\n        try:\n            import yfinance as yf\n            ticker_info = yf.Ticker(ticker)\n            company_name = ticker_info.info.get('longName', ticker)\n            return company_name\n        except Exception as e:\n            print(f\"  ⚠️  Could not fetch company name for {ticker}: {e}\")\n            return ticker\n\n    @retry_on_failure(max_retries=2, delay=1.0)\n    def fetch_ticker_news(self, ticker: str, max_articles: int = 30, lookback_days: int = 365) -> List[Dict]:\n        \"\"\"Fetch news for a specific ticker using Google News RSS\"\"\"\n        try:\n            import feedparser\n\n            company_name = self.get_company_name(ticker)\n            print(f\"  Fetching headlines for: {company_name} ({ticker})\")\n\n            rss_url = self.generate_google_news_rss_url(ticker, company_name, lookback_days)\n            feed = feedparser.parse(rss_url)\n\n            articles = []\n            if feed.entries:\n                for entry in feed.entries[:max_articles]:\n                    try:\n                        published_date = datetime(*entry.published_parsed[:6])\n                    except:\n                        published_date = datetime.now()\n\n                    articles.append({\n                        'title': entry.title,\n                        'summary': entry.get('summary', ''),\n                        'link': entry.link,\n                        'published': published_date,\n                        'ticker': ticker\n                    })\n\n                print(f\"  ✓ Found {len(articles)} articles for {ticker}\")\n            else:\n                print(f\"  ⚠️  No news found for {ticker}\")\n\n            return articles\n\n        except Exception as e:\n            print(f\"  ⚠️  Error fetching news for {ticker}: {e}\")\n            import traceback\n            traceback.print_exc()\n            return []\n\n    def analyze_ticker_sentiment(self, ticker: str, max_articles: int = 30, lookback_days: int = 365) -> Dict:\n        \"\"\"\n        Fetch news and perform sentiment analysis for a single ticker\n        Now uses FinBERT for better accuracy\n        \"\"\"\n        cache_key = f\"ticker_sentiment_{ticker}_{lookback_days}d_finbert\"\n        cached_data = self.cache.get(cache_key, max_age_hours=2)\n\n        if cached_data is not None:\n            return cached_data\n\n        articles = self.fetch_ticker_news(ticker, max_articles=max_articles, lookback_days=lookback_days)\n\n        if not articles:\n            return self._default_sentiment_metrics(ticker)\n\n        detailed_analyses = []\n        for article in articles:\n            headline = article.get('title', '')\n\n            if headline:\n                analysis = self.analyze_headline_detailed(headline)\n                analysis['article'] = article\n                detailed_analyses.append(analysis)\n\n        if detailed_analyses:\n            compounds = [a['compound'] for a in detailed_analyses]\n\n            metrics = {\n                'ticker': ticker,\n                'avg_sentiment': np.mean(compounds),\n                'sentiment_std': np.std(compounds),\n                'median_sentiment': np.median(compounds),\n                'positive_count': sum(1 for c in compounds if c > 0.05),\n                'negative_count': sum(1 for c in compounds if c < -0.05),\n                'neutral_count': sum(1 for c in compounds if -0.05 <= c <= 0.05),\n                'positive_ratio': sum(1 for c in compounds if c > 0.05) / len(compounds),\n                'negative_ratio': sum(1 for c in compounds if c < -0.05) / len(compounds),\n                'article_count': len(detailed_analyses),\n                'top_positive': sorted(detailed_analyses, key=lambda x: x['compound'], reverse=True)[:3],\n                'top_negative': sorted(detailed_analyses, key=lambda x: x['compound'])[:3],\n                'timestamp': datetime.now().isoformat(),\n                'analysis_method': detailed_analyses[0].get('method', 'unknown')\n            }\n        else:\n            metrics = self._default_sentiment_metrics(ticker)\n\n        self.cache.set(cache_key, metrics)\n        return metrics\n\n    def analyze_multiple_tickers(self, tickers: List[str], max_articles: int = 30, lookback_days: int = 365) -> Dict[str, Dict]:\n        \"\"\"Analyze sentiment for multiple tickers\"\"\"\n        results = {}\n\n        for i, ticker in enumerate(tickers):\n            print(f\"  Analyzing {ticker} ({i+1}/{len(tickers)})...\")\n            results[ticker] = self.analyze_ticker_sentiment(ticker, max_articles=max_articles, lookback_days=lookback_days)\n            time.sleep(0.5)\n\n        return results\n\n    def get_ticker_sentiment_summary(self, ticker: str, metrics: Dict) -> str:\n        \"\"\"Generate human-readable sentiment summary for a ticker\"\"\"\n        avg = metrics['avg_sentiment']\n\n        if avg > 0.1:\n            sentiment_label = \"Strongly Positive\"\n        elif avg > 0.05:\n            sentiment_label = \"Positive\"\n        elif avg > -0.05:\n            sentiment_label = \"Neutral\"\n        elif avg > -0.1:\n            sentiment_label = \"Negative\"\n        else:\n            sentiment_label = \"Strongly Negative\"\n\n        summary = f\"\"\"\n{ticker} News Sentiment:\n  Overall: {sentiment_label} (Score: {avg:+.3f})\n  Articles: {metrics['article_count']} analyzed\n  Distribution: {metrics['positive_ratio']:.1%} positive, {metrics['negative_ratio']:.1%} negative\n  Method: {metrics.get('analysis_method', 'unknown')}\n        \"\"\"\n\n        return summary.strip()\n\n    def create_sentiment_dataframe(self, ticker_results: Dict[str, Dict]) -> pd.DataFrame:\n        \"\"\"Convert ticker sentiment results into a pandas DataFrame\"\"\"\n        rows = []\n        for ticker, metrics in ticker_results.items():\n            rows.append({\n                'ticker': ticker,\n                'avg_sentiment': metrics['avg_sentiment'],\n                'sentiment_std': metrics['sentiment_std'],\n                'median_sentiment': metrics['median_sentiment'],\n                'positive_ratio': metrics['positive_ratio'],\n                'negative_ratio': metrics['negative_ratio'],\n                'article_count': metrics['article_count'],\n                'method': metrics.get('analysis_method', 'unknown')\n            })\n\n        return pd.DataFrame(rows)\n\n    def _default_sentiment_metrics(self, ticker: str) -> Dict:\n        \"\"\"Return default neutral sentiment metrics for a ticker\"\"\"\n        return {\n            'ticker': ticker,\n            'avg_sentiment': 0.0,\n            'sentiment_std': 0.0,\n            'median_sentiment': 0.0,\n            'positive_count': 0,\n            'negative_count': 0,\n            'neutral_count': 0,\n            'positive_ratio': 0.5,\n            'negative_ratio': 0.5,\n            'article_count': 0,\n            'top_positive': [],\n            'top_negative': [],\n            'timestamp': datetime.now().isoformat(),\n            'analysis_method': 'none'\n        }"
    },
    {
        "title": "Experimental HedgeFund Powered by AI (Advanced Data Science for Finance)",
        "description": "An LLM-powered stock investment selector project that automates the process of gathering macro-economic data, fetching financial news and statistics for S&P 500 firms, and generating investment reports. The core function is to use a Language Model (Gemini) to create an investor-focused macroeconomic report, and then use another LLM/platform (Expected Parrot's EDSL) to score individual stocks and determine a final recommended portfolio with optimal weights. The project is designed as an educational experiment in financial modeling.",
    "tech_stack": ["Python", "Jupyter Notebook", "Pandas", "Requests", "BeautifulSoup (for scraping)", "yfinance", "feedparser", "fredapi", "Gemini (using `google.generativeai`)", "Expected Parrot Domain-Specific Language (EDSL)"],
    "github": "",
    "demo": "The notebook serves as the primary demonstration of the workflow. The key external platform used for the AI experiments is Expected Parrot: [Expected Parrot](https://www.expectedparrot.com/login?ref=DRSS8CT9).",
    "image": "",
    "code_snippet": "```python\nfrom datetime import datetime, timedelta\n\ndef generate_google_news_rss_url(ticker, lookback_days):\n    \"\"\"Generates a Google News RSS feed URL for a given ticker and lookback period.\"\"\"\n    end_date = datetime.now()\n    start_date = end_date - timedelta(days=lookback_days)\n\n    end_date_str = end_date.strftime('%Y-%m-%d')\n    start_date_str = start_date.strftime('%Y-%m-%d')\n\n    # Construct the URL with ticker and date range\n    rss_url = f'[https://news.google.com/rss/search?q=](https://news.google.com/rss/search?q=){ticker}+after:{start_date_str}+before:{end_date_str}&ceid=US:en&hl=en-US&gl=US'\n    return rss_url\n```"}
]

# Research Papers (kept)
PAPERS = [
    {
        "title": "Does Remodeling Affect Housing Prices in the Greater Boston Area?",
        "abstract": "This paper conducts an empirical study examining the impact of post-2005 remodeling on Greater Boston Area housing assessment prices. Using data from Boston.gov's Property Assessment FY2025 dataset with 68,169 observations, we employ a hedonic pricing model controlling for neighborhood fixed effects (ZIP code) and structural characteristics. Our model achieves an R² of 84.95%. Results show that general property improvements yield a 6.72% increase in assessment values, kitchen remodels provide a 7.34% premium, and bathroom remodels contribute a 5.86% premium. Statistical testing confirms the kitchen remodel premium is significantly different from bathroom remodels (F-test, p=0.0005). This research contributes to understanding renovation investment returns in urban housing markets.",
        "authors": "Andrew Clark",
        "journal": "ECO 357 - Econometrics, Lehigh University",
        "year": "2025",
        "pdf_path": "",
        "keywords": ["Econometrics", "Hedonic Pricing", "Real Estate", "Fixed Effects", "Housing Valuation"]
    }
]

# Skills with proficiency levels (kept)
SKILLS = {
    "Python": 85,
    "R Programming": 90,
    "SQL": 85,
    "Stata": 88,
    "Excel / Financial Modeling": 95,
    "Econometrics": 90,
    "Financial Analysis": 92,
    "Data Visualization": 85,
    "Retool / SAGE Intacct": 82,
    "Statistical Methods": 88
}

# Achievements & Highlights (kept)
HIGHLIGHTS = [
    {
        "type": "Academic",
        "title": "Lehigh Trustees' Scholarship",
        "description": "Merit-based scholarship recipient. Dean's List: Fall 2023, Spring 2024, Fall 2024, Spring 2025. Perfect 4.0 GPA in Economics & Finance majors.",
        "icon": "🎓"
    },
    {
        "type": "Athletics",
        "title": "Track & Field Team Captain",
        "description": "Ranked #8 All-Time at Lehigh in Heptathlon and #9 for the Decathlon. All-Academic Patriot League Team. Academic Mentor and Big Mentor for team.",
        "icon": "🏃"
    },
    {
        "type": "Experience",
        "title": "Dual Finance Internships",
        "description": "Completed simultaneous internships at Tusk Strategies (Finance) and Park Avenue Capital (Financial Planning) summer 2025. Managed multi-million dollar portfolios.",
        "icon": "💼"
    },
    {
        "type": "Leadership",
        "title": "Student-Athlete Council",
        "description": "Serve as liaison between student-athletes and administration. Develop policies to enhance student-athlete well-being and engagement.",
        "icon": "🤝"
    }
]

# Experience (kept)
EXPERIENCE = [
    {
        "title": "Finance Intern",
        "company": "Tusk Strategies",
        "location": "New York, NY",
        "dates": "May 2025 – August 2025",
        "bullets": [
            "Managed accounting operations across 8 entities including VC funds, consulting services, and 501(c) organizations",
            "Modeled future cash flows, collections, and financial health using advanced Excel and SQL",
            "Built automated data summaries in Retool using SQL queries, improving financial reporting efficiency",
            "Analyzed receivables projections and expense tracking across multiple business units"
        ]
    },
    {
        "title": "Financial Planning Analyst Intern",
        "company": "Park Avenue Capital",
        "location": "New York, NY / Southbury, CT",
        "dates": "May 2025 – August 2025",
        "bullets": [
            "Created complex financial profiles for high-net-worth clients with $500K+ portfolios",
            "Analyzed financial statements to provide investment and insurance recommendations",
            "Presented weekly client presentations, resulting in acquiring a $9M client",
            "Generated comprehensive financial summaries and reports using proprietary software"
        ]
    },
    {
        "title": "Financial Representative Assistant",
        "company": "Park Avenue Capital",
        "location": "New York, NY / Southbury, CT",
        "dates": "June 2024 – August 2024",
        "bullets": [
            "Prepared data and materials for financial planning meetings",
            "Maintained CRM systems and updated client information",
            "Supported operational and administrative functions for wealth management team"
        ]
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def get_base64_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        return base64_pdf
    except:
        return None

def display_pdf(pdf_path, width=1000, height=900):
    base64_pdf = get_base64_pdf(pdf_path)
    if base64_pdf:
        pdf_display = f'''
        <div class="pdf-container">
            <iframe src="data:application/pdf;base64,{base64_pdf}"
                    width="{width}" height="{height}" type="application/pdf" style="border:none;">
            </iframe>
        </div>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.info("📄 Add your PDF file to the directory to display it here")

def create_download_button(file_path, button_text, file_name):
    try:
        with open(file_path, "rb") as f:
            st.download_button(
                label=button_text,
                data=f,
                file_name=file_name,
                mime="application/pdf"
            )
    except Exception:
        st.info(f"Add {file_name} to enable download")

def create_project_card(project):
    st.markdown(f"""
        <div class="custom-card project-card">
            <h3 style="color: {THEME['primary']};"> {project['title']}</h3>
            <p style="color: {THEME['text']}; line-height: 1.6;">{project['description']}</p>
            <div style="margin: 1rem 0;">
                {''.join([f'<span class="tag">{tech}</span>' for tech in project['tech_stack']])}
            </div>
        </div>
    """, unsafe_allow_html=True)

    if project.get('code_snippet'):
        lang = 'python' if any(k in project['title'].lower() for k in ['python','alpha','trading']) else 'r'
        with st.expander("📝 View Code Snippet"):
            st.code(project['code_snippet'], language=lang)

    col1, col2 = st.columns(2)
    with col1:
        if project.get('github'):
            st.markdown(f'<a href="{project["github"]}" target="_blank" class="custom-button" style= "color: white" >Link</a>',
                       unsafe_allow_html=True)

def create_skill_bar(skill_name, proficiency):
    st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                <span style="font-weight: 600; color: {THEME['text']};">{skill_name}</span>
                <span style="color: {THEME['text_light']};">{proficiency}%</span>
            </div>
            <div class="skill-bar">
                <div class="skill-fill" style="width: {proficiency}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE SECTIONS
# ============================================================================
def render_hero():
    st.markdown(f"""
        <div class="hero-container">
            <h1 class="hero-title">{PERSONAL_INFO['name']}</h1>
            <p style="color: {THEME['secondary']}; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
                {PERSONAL_INFO['title']}
            </p>
            <p class="hero-tagline">{PERSONAL_INFO['tagline']}</p>
            <p class="hero-subtitle">{PERSONAL_INFO['subtitle']}</p>
        </div>
    """, unsafe_allow_html=True)

    ticker_html = f"""
        <div style="background: linear-gradient(90deg, {THEME['primary']}, #0F766E);
                    color: white; padding: 1rem; text-align: center;
                    font-family: monospace; letter-spacing: 0.1em; font-size: 0.9rem;
                    border-top: 3px solid {THEME['secondary']};
                    border-bottom: 3px solid {THEME['secondary']};
                    font-weight: 600;">
            <marquee behavior="scroll" direction="left" scrollamount="5">
                🎓 GPA: 3.81/4.0 (4.0 in Major)  •  💼 2 Finance Internships (Summer 2025)  •
                📊 Econometrics Research  •  🏃 D1 Student-Athlete  •
                💻 Python | R | SQL | Stata Expert  •  🏆 Lehigh Trustees' Scholar
            </marquee>
        </div>
    """
    st.markdown(ticker_html, unsafe_allow_html=True)

def render_projects():
    st.markdown('<div class="section-content">', unsafe_allow_html=True)

    header_html = f"""
        <h1 style="font-family: 'Playfair Display', serif; color: {THEME['primary']};
                   border-bottom: 3px solid {THEME['secondary']}; padding-bottom: 0.5rem;">
            💻 Projects & Research
        </h1>
        <p style="color: {THEME['text_light']}; margin: 1rem 0 2rem 0; font-size: 1.05rem;">
            Data science, econometric modeling, and financial analysis projects
        </p>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    for i, project in enumerate(PROJECTS):
        create_project_card(project)
        if i < len(PROJECTS) - 1:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_research():
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    st.title("📚 Academic Research")

    for paper in PAPERS:
        paper_html = f"""
            <div class="custom-card">
                <h3 style="color: {THEME['primary']};">📄 {paper['title']}</h3>
                <p style="color: {THEME['text_light']}; font-style: italic;">
                    {paper['authors']} • {paper['journal']} • {paper['year']}
                </p>
                <h4>Abstract</h4>
                <p style="color: {THEME['text']}; line-height: 1.7;">{paper['abstract']}</p>
                <div style="margin: 1rem 0;">
                    {''.join([f'<span class="tag">{kw}</span>' for kw in paper['keywords']])}
                </div>
            </div>
        """
        st.markdown(paper_html, unsafe_allow_html=True)

        if paper.get('pdf_path'):
            tab1, tab2 = st.tabs(["📖 View Paper", "⬇️ Download"])
            with tab1:
                display_pdf(paper['pdf_path'])
            with tab2:
                create_download_button(paper['pdf_path'], "📥 Download PDF", f"{paper['title']}.pdf")
        else:
            st.info("📄 Full paper available upon request")

    st.markdown('</div>', unsafe_allow_html=True)

def render_experience():
    st.markdown('<div class="section-content">', unsafe_allow_html=True)

    header = f"""
        <h1 style="font-family: 'Playfair Display', serif; color: {THEME['primary']};
                   border-bottom: 3px solid {THEME['secondary']}; padding-bottom: 0.5rem;">
            💼 Professional Experience
        </h1>
    """
    st.markdown(header, unsafe_allow_html=True)

    for exp in EXPERIENCE:
        exp_html = f"""
            <div class="custom-card">
                <h3 style="color: {THEME['primary']};">{exp['title']}</h3>
                <p style="font-weight: 600; color: {THEME['text']}; margin: 0.5rem 0;">
                    {exp['company']} | {exp.get('location','')}
                </p>
                <p style="color: {THEME['text_light']}; font-style: italic; margin-bottom: 1rem;">
                    {exp['dates']}
                </p>
                <ul style="line-height: 1.8; color: {THEME['text']};">
                    {''.join([f'<li>{bullet}</li>' for bullet in exp['bullets']])}
                </ul>
            </div>
        """
        st.markdown(exp_html, unsafe_allow_html=True)

    # Education section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    edu_header = f"""
        <h2 style="font-family: 'Playfair Display', serif; color: {THEME['primary']}; margin-top: 2rem;">
            🎓 Education
        </h2>
    """
    st.markdown(edu_header, unsafe_allow_html=True)

    edu_html = f"""
        <div class="custom-card">
            <h3 style="color: {THEME['primary']};">Lehigh University</h3>
            <p style="font-weight: 600;">Bachelor of Science in Business and Economics</p>
            <p style="color: {THEME['text_light']};">Expected Graduation: May 2027 | Bethlehem, PA</p>
            <div style="margin: 1rem 0;">
                <span class="tag">Finance Major</span>
                <span class="tag">Economics Major</span>
                <span class="tag tag-brown">GPA: 3.81/4.0</span>
                <span class="tag tag-brown">Major GPA: 4.0/4.0</span>
            </div>
            <p style="margin-top: 1rem;"><strong>Honors:</strong> Lehigh Trustees' Scholarship, Dean's List (4 semesters)</p>
            <p><strong>Relevant Coursework:</strong> Econometrics, Advanced Data Science for Finance,
            Investments, Corporate Financial Policy, Statistical Methods II, Intermediate Micro/Macro Analysis</p>
        </div>
    """
    st.markdown(edu_html, unsafe_allow_html=True)

# -------------------------
# NEW: render_resume (integrates your uploaded PDF + inline content)
# -------------------------
def render_resume():
    st.markdown('<div class="section-content">', unsafe_allow_html=True)
    st.title("📋 Résumé — Andrew Clark")

    # Inline formatted résumé based on extracted PDF content
    st.header("Education")
    st.markdown(f"""
    <div class="custom-card">
      <strong>Lehigh University</strong><br>
      Bachelor of Science in Business and Economics (College of Business)<br>
      Expected Graduation: May 2027<br>
      Cumulative GPA: 3.81 • Major GPA: 4.00<br>
      Majors: Economics & Finance<br>
      <strong>Honors:</strong> Lehigh Trustees' Scholarship; Dean's List (Fall 2023, Spring/Fall 2024, Spring 2025)
      <div style="margin-top:0.75rem;"><strong>Relevant Coursework:</strong> Investments, Corporate Financial Policy, Econometrics & Data Science for Finance</div>
    </div>
    """, unsafe_allow_html=True)

    st.header("Internship & Work Experience")
    # Tusk Strategies
    st.markdown(f"""
    <div class="custom-card">
      <h4>Finance Intern — Tusk Strategies</h4>
      <p style="font-style:italic; color:{THEME['text_light']};">New York, NY • May 2025 – August 2025</p>
      <ul style="line-height:1.8;">
        <li>Managed an eight-entity accounting book</li>
        <li>Modeled future cash flows, collections, and financial health of the firm</li>
        <li>Analyzed current holdings, receivables projections, and expenses</li>
        <li>Learned operations for VC funds, EFS, consulting services, 501(c)(3), and 501(c)(4) organizations</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Park Avenue Capital - Financial Planning Analyst Intern
    st.markdown(f"""
    <div class="custom-card">
      <h4>Financial Planning Analyst Intern — Park Avenue Capital</h4>
      <p style="font-style:italic; color:{THEME['text_light']};">New York, NY / Southbury, CT • May 2025 – August 2025</p>
      <ul style="line-height:1.8;">
        <li>Created and modified complex financial profiles for clients</li>
        <li>Analyzed financial statements to provide recommendations</li>
        <li>Participated in client meetings</li>
        <li>Generated financial summaries and reports</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Park Avenue Capital - Financial Representative Assistant
    st.markdown(f"""
    <div class="custom-card">
      <h4>Financial Representative Assistant — Park Avenue Capital</h4>
      <p style="font-style:italic; color:{THEME['text_light']};">New York, NY / Southbury, CT • June 2024 – August 2024</p>
      <ul style="line-height:1.8;">
        <li>Prepared data and materials for financial planning meetings</li>
        <li>Organized data and maintained CRM to support financial planners</li>
        <li>Supported operational and administrative tasks</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.header("Campus & Community Involvement")
    st.markdown(f"""
    <div class="custom-card">
      <h4>Men's Track & Field Team Captain — Lehigh University</h4>
      <p style="font-style:italic; color:{THEME['text_light']};">August 2023 – Present</p>
      <ul style="line-height:1.8;">
        <li>Leadership roles: Captain, Event group leader, Academic Mentor, Big Mentor</li>
        <li>Balanced high academic and athletic performance; All-Academic Patriot League Team</li>
        <li>Ranked Top 10 All-Time for Lehigh Track & Field (#8 Heptathlon / #9 Decathlon)</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="custom-card">
      <h4>Student-Athlete Council — Lehigh University</h4>
      <p style="font-style:italic; color:{THEME['text_light']};">January 2025 – Present</p>
      <ul style="line-height:1.8;">
        <li>Served as liaison between athletes, coaches, and administration</li>
        <li>Collaborated on policies to enhance student-athlete well-being and engagement</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.header("Skills & Tools")
    # Skills section from PDF
    skills_html = f"""
    <div class="custom-card">
      <strong>Skills:</strong>
      <ul style="line-height:1.8;">
        <li>Proficient in SAGE Intacct, NETX360, YCharts, DBeaver, Retool, Microsoft Office, Google Suite</li>
      </ul>
      <strong>Coding:</strong>
      <ul style="line-height:1.8;">
        <li>R, Python, SQL, Stata</li>
      </ul>
      <strong>Interests:</strong>
      <ul style="line-height:1.8;">
        <li>Sailing, Reading, Guitar</li>
      </ul>
    </div>
    """
    st.markdown(skills_html, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Embedded PDF viewer + download button (place your PDF in `assets/`)
    resume_path = "assets/Andrew_Clark_Resume.pdf"
    st.subheader("📄 Full Résumé (PDF)")
    try:
        display_pdf(resume_path, width=1000, height=900)
        create_download_button(resume_path, "⬇️ Download Résumé PDF", "Andrew_Clark_Resume.pdf")
    except Exception:
        st.info("Resume PDF not found in assets/ — add 'Andrew_Clark_Resume.pdf' to the assets/ folder or change the path.")

    st.markdown('</div>', unsafe_allow_html=True)

import urllib.parse

def render_contact():
    """Render contact section with updated contact details"""
    st.markdown('<div class="section-content">', unsafe_allow_html=True)

    # Section header
    st.markdown(f"""
        <h1 style="font-family: 'Playfair Display', serif; color: {THEME['primary']};
                    border-bottom: 3px solid {THEME['secondary']}; padding-bottom: 0.5rem;">
                📬 Get In Touch
        </h1>
    """, unsafe_allow_html=True)

    # Centered Lottie animation
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        lottie_url = "https://assets2.lottiefiles.com/packages/lf20_u25cckyh.json"
        lottie_json = load_lottie_url(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=200, key="contact_animation")

    # Contact card
    contact_card = f"""
        <div class="custom-card" style="text-align: center;">
            <h3>Let's Connect!</h3>
            <p style="color: {THEME['text_light']};">
                I'm always interested in new opportunities and collaborations.
            </p>
            <div style="margin: 2rem 0; line-height: 2.2; font-size: 1.05rem;">
                <p><strong>Email:</strong> <a href="mailto:{PERSONAL_INFO['email']}"
                    style="text-decoration: underline; color: {THEME['primary']};">{PERSONAL_INFO['email']}</a></p>
                <p><strong>Phone:</strong> <a href="tel:{PERSONAL_INFO['phone']}"
                    style="text-decoration: underline; color: {THEME['text']};">{PERSONAL_INFO['phone']}</a></p>
                <p><strong>Location:</strong> <span style="color:{THEME['text']};">{PERSONAL_INFO['location']}</span></p>
                <hr style="border-top: 1px solid {THEME['text_light']}33; margin: 15px 0;">
                <p><strong>LinkedIn:</strong> <a href="{PERSONAL_INFO['linkedin']}" target="_blank"
                    style="text-decoration: underline; color: {THEME['primary']};">Andrew Clark</a></p>
                <p><strong>GitHub:</strong> <a href="{PERSONAL_INFO['github']}" target="_blank"
                    style="text-decoration: underline; color: {THEME['primary']};">{PERSONAL_INFO['github'].split('/')[-1]}</a></p>
            </div>
        </div>
    """
    st.markdown(contact_card, unsafe_allow_html=True)

def render_highlights():
    """Render other work and highlights"""
    st.markdown('<div class="section-content">', unsafe_allow_html=True)

    highlights_header = f"""
        <h1 style="font-family: 'Playfair Display', serif; color: {THEME['primary']};
                   border-bottom: 3px solid {THEME['secondary']}; padding-bottom: 0.5rem;">
            ✨ Recognition & Achievements
        </h1>
    """
    st.markdown(highlights_header, unsafe_allow_html=True)

    # Create grid layout for highlights
    cols = st.columns(2)
    for i, highlight in enumerate(HIGHLIGHTS):
        with cols[i % 2]:
            highlight_card = f"""
                <div class="custom-card" style="text-align: center; min-height: 220px;">
                    <div style="font-size: 3.5rem; margin-bottom: 1rem;">
                        {highlight['icon']}
                    </div>
                    <span class="tag tag-gold">{highlight['type']}</span>
                    <h4 style="color: {THEME['primary']}; margin: 1rem 0 0.5rem 0; font-family: 'Playfair Display', serif;">
                        {highlight['title']}
                    </h4>
                    <p style="color: {THEME['text_light']}; font-size: 0.95rem; line-height: 1.6;">
                        {highlight['description']}
                    </p>
                </div>
            """
            st.markdown(highlight_card, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Load custom CSS
    load_css()

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to:",
        ["Home", "Projects", "Research", "Resume", "Highlights", "Contact"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    sidebar_info = f"""
        <div style="text-align: center; color: white;">
            <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                <strong>{PERSONAL_INFO['name']}</strong>
            </p>
            <p style="font-size: 0.8rem; opacity: 0.8;">
                {PERSONAL_INFO['tagline']}
            </p>
            <p style="font-size: 0.75rem; margin-top:0.25rem;">{PERSONAL_INFO['location']}</p>
        </div>
    """
    st.sidebar.markdown(sidebar_info, unsafe_allow_html=True)

    # Render selected page
    if page == "Home":
        render_hero()
        st.markdown("---")

        # Show brief overview with market-style stats
        col1, col2, col3 = st.columns(3)
        with col1:
            stat1 = f"""
                <div class="stat-box">
                    <div class="stat-number">{len(PROJECTS)}</div>
                    <div class="stat-label">Quant Projects</div>
                </div>
            """
            st.markdown(stat1, unsafe_allow_html=True)
        with col2:
            stat2 = f"""
                <div class="stat-box">
                    <div class="stat-number">{len(PAPERS)}</div>
                    <div class="stat-label">Publications</div>
                </div>
            """
            st.markdown(stat2, unsafe_allow_html=True)
        with col3:
            stat3 = f"""
                <div class="stat-box">
                    <div class="stat-number">{len(HIGHLIGHTS)}</div>
                    <div class="stat-label">Awards & Grants</div>
                </div>
            """
            st.markdown(stat3, unsafe_allow_html=True)

    elif page == "Projects":
        render_projects()

    elif page == "Research":
        render_research()

    elif page == "Resume":
        render_resume()

    elif page == "Highlights":
        render_highlights()

    elif page == "Contact":
        render_contact()


if __name__ == "__main__":
    main()
