"""
Streamlit dashboard – Persistent Spectral Graph + Hodge Laplacian Engine
Professional layout with cards, metric explanations, and spectral/topological visuals.
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

# Page config
st.set_page_config(
    page_title="Topo‑Spectral Engine",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .universe-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .etf-card {
        background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .etf-card:hover {
        transform: translateY(-5px);
    }
    .etf-ticker {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .etf-score {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔷 Persistent Spectral Graph + Hodge Laplacian Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Topological‑spectral market structure | Regime‑aware ETF ranking</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔷 Topo‑Spectral Engine")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Graph Laplacian + Hodge 1‑Laplacian + Persistent Homology")
st.sidebar.markdown("**Filtration:** Rips complex on correlation distance")
st.sidebar.markdown("---")
st.sidebar.caption("Data: [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)")

# Expandable explanation
with st.expander("📖 What do the scores mean?"):
    st.markdown("""
    <div class="info-box">
    The <strong>topological‑spectral score</strong> for each ETF is computed as:
    <br><br>
    <code>score = eigenvector_centrality × (1 + persistent_loops) × (1 / (1 + λ₂))</code>
    <br><br>
    - <strong>Eigenvector centrality</strong>: How influentially connected the ETF is in the correlation graph.<br>
    - <strong>Persistent loops (1‑dim)</strong>: Number of independent cyclical structures in the market that survive above the 0.5 correlation threshold.<br>
    - <strong>λ₂ (algebraic connectivity)</strong>: Second smallest Laplacian eigenvalue – low values indicate fragile, cluster‑prone markets.<br>
    <br>
    <strong>Interpretation:</strong><br>
    - High score → ETF is structurally central in a market with strong persistent cycles and low connectivity → often a leading indicator of regime shifts.<br>
    - Low score → ETF is peripheral or the market is highly connected (tightly coupled).<br>
    </div>
    """, unsafe_allow_html=True)

# Load data
OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'topo_spectral_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run `trainer.py` first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("📈 Top ETFs by Topological‑Spectral Score")
st.markdown("*Ranked by combined centrality, persistent cycles, and market fragility.*")

# Display each universe
for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            score = etf["score"]
            # Color based on score threshold
            if score >= 0.01:
                delta_color = "normal"
                strength = "High"
            elif score >= 0.005:
                delta_color = "off"
                strength = "Medium"
            else:
                delta_color = "inverse"
                strength = "Low"
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">score = {score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Structural importance: {strength}")
    
    # Spectral/topological metrics in a row
    meta = uni_data.get("metadata", {})
    if meta:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("λ₂ (connectivity)", f"{meta.get('lap_eigs', [0])[0]:.4f}" if meta.get('lap_eigs') else "N/A")
        with col2:
            st.metric("Persistent loops (1‑dim)", meta.get('persistence_1', 0))
        with col3:
            st.metric("Persistent components (0‑dim)", meta.get('persistence_0', 0))
        with col4:
            st.metric("Hodge 1‑Laplacian (1st eig)", f"{meta.get('hodge_eigs', [0])[0]:.4f}" if meta.get('hodge_eigs') else "N/A")
        
        # Expandable detailed table
        with st.expander("📊 Full ETF centrality & details"):
            centrality = meta.get("centrality", {})
            if centrality:
                df_cent = pd.DataFrame(list(centrality.items()), columns=["ETF", "Centrality"])
                df_cent = df_cent.sort_values("Centrality", ascending=False)
                st.dataframe(df_cent, use_container_width=True, hide_index=True)
            st.write("**Graph Laplacian eigenvalues (first 3):**", meta.get('lap_eigs', []))
            st.write("**Hodge 1‑Laplacian eigenvalues (first 3):**", meta.get('hodge_eigs', []))
    st.divider()

# Historical trend (if multiple JSON files)
st.header("📉 Historical Evolution of Top ETF Scores (Last 30 days)")
with st.spinner("Loading historical data..."):
    json_files = [f for f in files if f.endswith('.json') and 'topo_spectral_' in f]
    json_files.sort(reverse=True)
    history = []
    for fname in json_files[:30]:
        try:
            fs = HfFileSystem(token=HF_TOKEN)
            with fs.open(f"datasets/{OUTPUT_REPO}/{fname}", "r") as f:
                hist_data = json.load(f)
                run_date = hist_data['run_date']
                for uni, val in hist_data['universes'].items():
                    if 'top_etfs' in val and val['top_etfs']:
                        top_score = val['top_etfs'][0]['score']
                        history.append({"date": run_date, "universe": uni, "top_score": top_score})
        except:
            pass
    if history:
        df_hist = pd.DataFrame(history)
        fig = px.line(df_hist, x="date", y="top_score", color="universe",
                      title="Top ETF Score Over Time",
                      labels={"top_score": "Score (higher = more central)", "date": "Run Date"})
        fig.update_layout(height=400, legend_title="Universe")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to plot trend.")

st.caption("Higher score indicates stronger topological‑spectral importance. The engine runs daily with a rolling 60‑day correlation window.")
