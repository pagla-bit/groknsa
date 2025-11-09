# app.py
import streamlit as st
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import pandas as pd
import urllib.parse
from typing import List, Dict, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import time
import html
from dateutil import parser as date_parser
import logging
from functools import wraps
import os

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# -----------------------
# Constants
# -----------------------
FINVIZ_BASE = "https://finviz.com"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
REQUEST_TIMEOUT = 15
MAX_TICKERS = 200
MAX_HEADLINES_PER_SITE = 20
FINVIZ_PAGE_SIZE = 20

# -----------------------
# Config & Page Setup
# -----------------------
st.set_page_config(page_title="Ticker News Sentiment", layout="wide")

# Dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
theme = "dark" if dark_mode else "light"

# Custom CSS
st.markdown(f"""
<style>
    .reportview-container {{ background: {"#1e1e1e" if dark_mode else "#ffffff"} }}
    .sidebar .sidebar-content {{ background: {"#2d2d2d" if dark_mode else "#f0f2f6"} }}
    .Widget>label {{ color: {"#ffffff" if dark_mode else "#000000"} }}
    h1, h2, h3, .stMarkdown {{ color: {"#ffffff" if dark_mode else "#000000"} }}
    .stTextInput > div > div > input {{ color: {"#ffffff" if dark_mode else "#000000"}; background: {"#333" if dark_mode else "#fff"} }}
    .stButton>button {{ background: {"#4CAF50" if not dark_mode else "#66BB6A"}; color: white }}
    table {{ color: {"#ddd" if dark_mode else "#333"} }}
    a {{ color: {"#90caf9" if dark_mode else "#1e88e5"} }}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Rate Limiter
# -----------------------
def rate_limit(calls_per_second: float = 1.5):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# -----------------------
# Cached Model Loaders
# -----------------------
@st.cache_resource(show_spinner=False)
def load_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=True)
def load_finbert_pipeline():
    return pipeline(
        "text-classification",
        model="yiyanghkust/finbert-tone",
        device=-1,
        truncation=True,
        max_length=512
    )

# -----------------------
# Networking Helper
# -----------------------
@rate_limit(calls_per_second=1.5)
def fetch_url(url: str, headers=None, params=None) -> Optional[requests.Response]:
    headers = headers or {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log.warning(f"Fetch failed: {url} | {e}")
        return None

# -----------------------
# Finviz Screener Parser (Cached)
# -----------------------
@st.cache_data(ttl=3600, show_spinner=False)
def extract_tickers_from_finviz_screener(screener_url: str, max_total: int = 500) -> List[str]:
    """Extract tickers from Finviz screener with pagination."""
    if not screener_url:
        return []
    parsed = urllib.parse.urlparse(screener_url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    query = urllib.parse.parse_qs(parsed.query)
    base_params = {k: v[0] for k, v in query.items() if k != "r"}
    tickers = []
    seen = set()
    start = 1

    while len(tickers) < max_total:
        params = dict(base_params)
        params["r"] = str(start)
        resp = fetch_url(base, params=params)
        if not resp:
            break
        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.find_all("a", href=True)
        page_tickers = []
        for a in anchors:
            href = a["href"]
            if "quote.ashx?t=" in href:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                t = q.get("t")
                if t:
                    ticker = t[0].strip().upper()
                    if ticker and ticker not in seen:
                        page_tickers.append(ticker)
                        seen.add(ticker)
                        tickers.append(ticker)
                        if len(tickers) >= max_total:
                            break
        if not page_tickers:
            break
        start += FINVIZ_PAGE_SIZE
        if start > 2000:
            break
    return tickers

# -----------------------
# News Parsers
# -----------------------
def parse_finviz(ticker: str, max_items: int = 10) -> List[Dict]:
    url = f"{FINVIZ_BASE}/quote.ashx"
    params = {"t": ticker}
    resp = fetch_url(url, params=params)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    news_table = soup.find("table", class_="fullview-news-outer")
    results = []

    if news_table:
        for tr in news_table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                timestamp = tds[0].get_text(strip=True)
                a = tds[1].find("a")
                if a:
                    title = a.get_text(strip=True)
                    link = a.get("href")
                    link = link if link.startswith("http") else f"{FINVIZ_BASE}/{link.lstrip('/')}" if link else ""
                    results.append({"title": title, "link": link, "source": "finviz", "timestamp": timestamp})
                    if len(results) >= max_items:
                        break
    return results[:max_items]

def parse_google_news(ticker: str, max_items: int = 10) -> List[Dict]:
    query = urllib.parse.quote_plus(f"{ticker} stock")
    url = f"{GOOGLE_NEWS_RSS}?q={query}&hl=en-US&gl=US&ceid=US:en"
    resp = fetch_url(url)
    if not resp:
        return []
    try:
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:max_items]
        results = []
        for item in items:
            title = item.title.get_text(strip=True) if item.title else ""
            link = item.link.get_text(strip=True) if item.link else ""
            ts = item.pubDate.get_text(strip=True) if item.pubDate else ""
            results.append({"title": title, "link": link, "source": "google", "timestamp": ts})
        return results
    except Exception as e:
        log.warning(f"Google RSS parse error: {e}")
        return []

# -----------------------
# High-level fetch (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_cached_all_sites_for_ticker(ticker: str, sites_tuple: Tuple[str, ...], max_items: int) -> List[Dict]:
    sites = list(sites_tuple)
    all_items = []
    for s in sites:
        if s == "finviz":
            all_items.extend(parse_finviz(ticker, max_items))
        elif s == "google":
            all_items.extend(parse_google_news(ticker, max_items))
    seen = set()
    deduped = []
    for it in all_items:
        key = it.get("title")
        if key and key not in seen:
            deduped.append(it)
            seen.add(key)
    return deduped

# -----------------------
# Sentiment Helpers
# -----------------------
def analyze_vader(vader, text: str) -> str:
    vs = vader.polarity_scores(text)
    c = vs["compound"]
    return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

def analyze_finbert_batch(pipe, texts: List[str]) -> List[str]:
    try:
        batch = [t[:512] for t in texts]
        results = pipe(batch)
        return [
            "positive" if "pos" in r["label"].lower() else
            "negative" if "neg" in r["label"].lower() else "neutral"
            for r in results
        ]
    except Exception as e:
        log.warning(f"FinBERT batch failed: {e}")
        return ["neutral"] * len(texts)

# -----------------------
# Process Ticker
# -----------------------
def process_ticker(ticker: str, sites: List[str], run_vader: bool, run_finbert: bool, max_per_site: int) -> Dict:
    """Process one ticker: fetch news, analyze sentiment, return summary."""
    ticker = ticker.strip().upper()
    headlines = fetch_cached_all_sites_for_ticker(ticker, tuple(sites), int(max_per_site))
    vader = load_vader() if run_vader else None
    finbert_pipe = load_finbert_pipeline() if run_finbert else None

    vpos = vneg = vneu = 0
    fpos = fneg = fneu = 0
    detailed = []

    titles = [h.get("title", "") for h in headlines]
    f_labels = analyze_finbert_batch(finbert_pipe, titles) if run_finbert and finbert_pipe else ["n/a"] * len(titles)

    for i, h in enumerate(headlines):
        title = h.get("title", "")
        link = h.get("link", "")
        source = h.get("source", "")
        timestamp = h.get("timestamp", "")

        v_label = analyze_vader(vader, title) if run_vader and vader else "n/a"
        f_label = f_labels[i] if run_finbert else "n/a"

        if v_label == "positive": vpos += 1
        elif v_label == "negative": vneg += 1
        else: vneu += 1

        if f_label == "positive": fpos += 1
        elif f_label == "negative": fneg += 1
        else: fneu += 1

        detailed.append({
            "title": title, "link": link, "source": source, "timestamp": timestamp,
            "vader": v_label, "finbert": f_label
        })

    return {
        "ticker": ticker,
        "vader_pos": vpos, "vader_neg": vneg, "vader_neu": vneu,
        "finbert_pos": fpos, "finbert_neg": fneg, "finbert_neu": fneu,
        "n_headlines": len(headlines), "headlines": detailed
    }

# -----------------------
# UI
# -----------------------
st.title("📈 Stock Ticker News Sentiment (VADER + FinBERT)")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ("Manual (text)", "Upload Excel", "Finviz Screener URL"))
    run_vader = st.checkbox("VADER (fast)", value=True)
    run_finbert = st.checkbox("FinBERT (slower)", value=True)
    sites_multiselect = st.multiselect("News sources", ["finviz", "google"], default=["finviz", "google"])
    max_news = st.number_input("Max headlines per site", 1, MAX_HEADLINES_PER_SITE, 3)
    max_workers = st.slider("Concurrent workers", 2, 20, 8)

    st.write("---")
    tickers = []
    screener_url = ""
    uploaded_file = None

    if input_mode == "Manual (text)":
        raw = st.text_area("Tickers (comma, newline, space)", placeholder="AAPL, MSFT, TSLA")
        if raw:
            tickers = [t.strip().upper() for t in raw.replace(",", " ").split() if t.strip()]
    elif input_mode == "Upload Excel":
        uploaded_file = st.file_uploader("Upload .xlsx (single column or 'ticker')", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                if "ticker" in [c.lower() for c in df.columns]:
                    col = next(c for c in df.columns if c.lower() == "ticker")
                elif df.shape[1] == 1:
                    col = df.columns[0]
                else:
                    st.warning("No 'ticker' column or single column found.")
                    col = None
                if col:
                    tickers = [str(x).strip().upper() for x in df[col].dropna().astype(str).tolist() if str(x).strip()]
            except Exception as e:
                st.error(f"File error: {e}")
    else:
        screener_url = st.text_input("Finviz screener URL:", placeholder="https://finviz.com/screener.ashx?v=111&f=...")
        if screener_url:
            with st.spinner("Fetching screener..."):
                tickers = extract_tickers_from_finviz_screener(screener_url, MAX_TICKERS)
                if not tickers:
                    st.warning("No tickers found.")

    if len(tickers) > MAX_TICKERS:
        st.warning(f"Limiting to {MAX_TICKERS} tickers.")
        tickers = tickers[:MAX_TICKERS]

    col1, col2 = st.columns([1, 1])
    with col1:
        run_button = st.button("Run Analysis", type="primary")
    with col2:
        refresh = st.button("Refresh")

    if refresh:
        st.rerun()

if tickers:
    st.markdown(f"**Tickers ({len(tickers)}):** {', '.join(tickers[:40])}{'...' if len(tickers)>40 else ''}")

if run_button:
    if not tickers:
        st.error("No tickers.")
    elif not sites_multiselect:
        st.error("Select news source.")
    elif not (run_vader or run_finbert):
        st.error("Select sentiment model.")
    else:
        t0 = time.time()
        st.info("Analyzing...")

        if run_vader: _ = load_vader()
        if run_finbert: _ = load_finbert_pipeline()

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Use ProcessPoolExecutor for thread safety
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_ticker, t, sites_multiselect, run_vader, run_finbert, max_news): t
                for t in tickers
            }
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                tkr = futures[future]
                try:
                    res = future.result()
                except Exception as e:
                    log.error(f"{tkr} failed: {e}")
                    res = {k: 0 for k in ["vader_pos", "vader_neg", "vader_neu", "finbert_pos", "finbert_neg", "finbert_neu"]}
                    res.update({"ticker": tkr, "n_headlines": 0, "headlines": []})
                results.append(res)
                progress_bar.progress((i + 1) / len(tickers))
                status_text.text(f"Processed {i+1}/{len(tickers)}")

        progress_bar.empty()
        status_text.empty()

        # Summary Table
        st.subheader("Sentiment Summary")
        df_res = pd.DataFrame(results).set_index("ticker")
        display_df = df_res[["vader_pos", "vader_neg", "vader_neu", "finbert_pos", "finbert_neg", "finbert_neu"]]

        def color_ticker(ticker, row):
            pos = row["vader_pos"] + row["finbert_pos"]
            neg = row["vader_neg"] + row["finbert_neg"]
            color = "green" if pos > neg else "red" if neg > pos else "gray"
            return f"<span style='color:{color};font-weight:bold'>{html.escape(ticker)}</span>"

        header = "<tr><th>Ticker</th><th style='text-align:center'>VADER</th><th style='text-align:center'>FINBERT</th></tr>"
        rows = []
        for ticker, row in display_df.reset_index().iterrows():
            tv = color_ticker(row["ticker"], row)
            vader_cell = f"🟢 {row['vader_pos']}  🔴 {row['vader_neg']}  🟡 {row['vader_neu']}"
            finbert_cell = f"🟢 {row['finbert_pos']}  🔴 {row['finbert_neg']}  🟡 {row['finbert_neu']}"
            rows.append(f"<tr><td>{tv}</td><td style='text-align:center'>{vader_cell}</td><td style='text-align:center'>{finbert_cell}</td></tr>")
        st.markdown(f"<table style='width:100%'>{header}{''.join(rows)}</table>", unsafe_allow_html=True)

        # Top Movers
        st.subheader("Top Movers")
        movers = [{"ticker": r["ticker"], "pos": r["vader_pos"]+r["finbert_pos"], "neg": r["vader_neg"]+r["finbert_neg"]} for r in results]
        df_m = pd.DataFrame(movers)
        top_pos = df_m.nlargest(3, "pos")
        top_neg = df_m.nlargest(3, "neg")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Positive**")
            html = "<table><tr><th>#</th><th>Ticker</th><th>Pos</th></tr>"
            for i, r in top_pos.iterrows():
                html += f"<tr><td>{i+1}</td><td><b style='color:green'>{r['ticker']}</b></td><td>{int(r['pos'])}</td></tr>"
            st.markdown(html + "</table>", unsafe_allow_html=True)
        with c2:
            st.markdown("**Top Negative**")
            html = "<table><tr><th>#</th><th>Ticker</th><th>Neg</th></tr>"
            for i, r in top_neg.iterrows():
                html += f"<tr><td>{i+1}</td><td><b style='color:red'>{r['ticker']}</b></td><td>{int(r['neg'])}</td></tr>"
            st.markdown(html + "</table>", unsafe_allow_html=True)

        # Detailed Headlines
        st.subheader("Detailed Headlines")
        all_details = []
        for r in results:
            for h in r["headlines"]:
                all_details.append({
                    "ticker": r["ticker"],
                    "timestamp_raw": h["timestamp"],
                    "title": h["title"],
                    "link": h["link"],
                    "vader": h["vader"],
                    "finbert": h["finbert"]
                })

        # Filters
        col1, col2 = st.columns([1, 2])
        with col1:
            ticker_filter = st.selectbox("Filter Ticker", ["All"] + sorted({r["ticker"] for r in results}))
        with col2:
            search_term = st.text_input("Search headlines")

        if ticker_filter != "All":
            all_details = [d for d in all_details if d["ticker"] == ticker_filter]
        if search_term:
            all_details = [d for d in all_details if search_term.lower() in d["title"].lower()]

        # Parse timestamps
        def safe_parse(ts):
            if not ts or len(ts) < 5: return None
            if any(x in ts.lower() for x in ["ago", "hour", "yesterday"]):
                try: return date_parser.parse(ts, fuzzy=False)
                except: pass
            try: return date_parser.parse(ts, fuzzy=False)
            except: return None

        for d in all_details:
            d["_dt"] = safe_parse(d["timestamp_raw"])

        # Sort
        with_dt = sorted([d for d in all_details if d["_dt"]], key=lambda x: x["_dt"], reverse=True)
        without_dt = [d for d in all_details if not d["_dt"]]
        sorted_details = with_dt + without_dt

        # Build table
        source_icons = {
            "finviz.com": "Chart",
            "news.google.com": "Search",
            "finance.yahoo.com": "Money",
            "cnbc.com": "TV",
            "bloomberg.com": "Globe",
            "reuters.com": "Newspaper"
        }
        def get_icon(link):
            if not link: return "Link"
            host = urllib.parse.urlparse(link).netloc.replace("www.", "")
            return source_icons.get(host, "Link")

        def ball(l): return {"positive": "Green Circle", "negative": "Red Circle", "neutral": "Yellow Circle"}.get(l, "")

        header = "<tr><th>Ticker</th><th>Time</th><th>Source</th><th>Headline</th><th>VADER</th><th>FINBERT</th></tr>"
        rows = []
        for d in sorted_details:
            ts = d["_dt"].strftime("%Y-%m-%d %H:%M") if d["_dt"] else html.escape(d["timestamp_raw"])
            icon = get_icon(d["link"])
            link = f"<a href='{html.escape(d['link'])}' target='_blank'>{html.escape(d['title'])}</a>" if d["link"] else html.escape(d["title"])
            rows.append(
                f"<tr>"
                f"<td>{html.escape(d['ticker'])}</td>"
                f"<td>{ts}</td>"
                f"<td>{icon} {html.escape(urllib.parse.urlparse(d['link']).netloc.replace('www.', '')[:15] if d['link'] else '')}</td>"
                f"<td>{link}</td>"
                f"<td style='text-align:center'>{ball(d['vader'])}</td>"
                f"<td style='text-align:center'>{ball(d['finbert'])}</td>"
                f"</tr>"
            )
        st.markdown(f"<table style='width:100%'>{header}{''.join(rows)}</table>", unsafe_allow_html=True)

        # Download
        csv = pd.DataFrame(results).to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv")

        st.success(f"Done in {time.time()-t0:.1f}s — {len(results)} tickers.")
