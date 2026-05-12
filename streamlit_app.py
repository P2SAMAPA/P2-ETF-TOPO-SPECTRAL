import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Topo‑Spectral Engine", layout="wide")
st.title("🔷 Persistent Spectral Graph + Hodge Laplacian Engine")
st.caption("Combines topological persistence with spectral graph theory. Detects market structure shifts and ranks ETFs by topological centrality.")

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
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Graph Laplacian + Hodge 1‑Laplacian + Persistent Homology")

universes = data["universes"]
for universe_name, uni_data in universes.items():
    st.subheader(f"🌍 {universe_name}")
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.info("No predictions")
        continue
    cols = st.columns(3)
    for i, etf in enumerate(top_etfs):
        with cols[i]:
            st.metric(f"#{i+1} {etf['ticker']}", f"score = {etf['score']:.3f}")
    # Show metadata expander
    with st.expander("Topological & Spectral Metrics"):
        meta = uni_data["metadata"]
        st.write(f"**Graph Laplacian eigenvalues (first 3):** {meta['lap_eigs']}")
        st.write(f"**Hodge 1‑Laplacian eigenvalues:** {meta['hodge_eigs']}")
        st.write(f"**Persistent 0‑dim components (threshold 0.5):** {meta['persistence_0']}")
        st.write(f"**Persistent 1‑dim loops:** {meta['persistence_1']}")
        # Centrality table
        cent_df = pd.DataFrame(list(meta['centrality'].items()), columns=["ETF", "Centrality"]).sort_values("Centrality", ascending=False)
        st.dataframe(cent_df, use_container_width=True)
    st.divider()

st.caption("Top ETFs are scored by eigenvector centrality weighted by topological persistence (cycles) and inverse algebraic connectivity. Higher score = more central in a tightly clustered market structure with persistent loops.")
