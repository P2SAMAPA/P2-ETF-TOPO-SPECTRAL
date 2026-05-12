# 🔷 Persistent Spectral Graph + Hodge Laplacian Engine

**Topological‑spectral market structure analysis for ETF universes.**  
Combines graph Laplacian eigenvalues, Hodge 1‑Laplacian, and persistent homology to detect regime shifts and rank ETFs by topological centrality.

---

## 🧠 Core Idea

Financial markets can be modelled as a **weighted graph** where nodes are ETFs and edge weights are absolute correlations of daily returns. This engine analyses the graph using:

- **Graph Laplacian eigenvalues** – algebraic connectivity & cluster strength.
- **Hodge 1‑Laplacian eigenvalues** – higher‑order interactions (edge loops, cycle structure).
- **Persistent homology** (0‑dim and 1‑dim barcodes) – how connected components and loops form and persist as we threshold the correlation network.

From these, we compute a **topological‑spectral score** for each ETF:
score = eigenvector_centrality × (1 + persistent_loops) × (1 / (1 + λ₂))

text

where λ₂ is the second smallest eigenvalue of the graph Laplacian (a measure of connectivity).  
ETFs with high centrality, in a market with many persistent cycles and low algebraic connectivity, receive the highest scores – indicating they are **structurally important** and may lead to regime breaks.

---

## 📦 Repository Structure
P2-ETF-TOPO-SPECTRAL/
├── .github/workflows/daily_run.yml # Daily automation
├── config.py # Universes, rolling window, thresholds
├── data_manager.py # Load master_data.parquet
├── topological_engine.py # Core: graph, Laplacians, persistence
├── trainer.py # Rolling window evaluation & ranking
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── us_calendar.py # Next trading day helper
├── requirements.txt # Dependencies
└── README.md # This file

text

---

## 🚀 How It Works (Daily Run)

1. **Load data** – `master_data.parquet` from Hugging Face (`P2SAMAPA/fi-etf-macro-signal-master-data`).
2. **For each universe** (FI_COMMODITIES, EQUITY_SECTORS, COMBINED):
   - Compute ETF log‑returns.
   - Build **correlation graph** using the last `CORR_WINDOW` days (default 60).
   - Compute:
     - Graph Laplacian eigenvalues (smallest k).
     - Hodge 1‑Laplacian eigenvalues (edge Laplacian via incidence matrix).
     - 0‑dim and 1‑dim persistence barcodes using GUDHI (Rips complex from distance matrix `1 - corr`).
   - Compute eigenvector centrality.
   - Combine scores and rank ETFs.
   - Save top 3 ETFs together with all spectral/topological metrics.
3. Upload results to `P2SAMAPA/p2-etf-topo-spectral-results`.
4. Streamlit dashboard loads the latest result and displays top ETFs and diagnostic metrics.

---

## 📊 Output (JSON per day)

```json
{
  "run_date": "2026-05-12",
  "universes": {
    "FI_COMMODITIES": {
      "top_etfs": [
        {"ticker": "GLD", "score": 0.87},
        {"ticker": "SLV", "score": 0.76},
        {"ticker": "TLT", "score": 0.65}
      ],
      "metadata": {
        "centrality": {"GLD": 0.32, "SLV": 0.28, ...},
        "lap_eigs": [0.12, 0.34, 0.67],
        "hodge_eigs": [0.45, 0.89],
        "persistence_0": 3,
        "persistence_1": 1
      }
    },
    ...
  }
}
persistence_0 – number of connected components that survive beyond threshold 0.5.

persistence_1 – number of 1‑dimensional loops (holes) that persist.

Low λ₂ (algebraic connectivity) + high persistence_1 → fragile, highly interwoven market structure.

🖥️ Streamlit Dashboard
Shows top 3 ETFs per universe with their topological‑spectral score.

Expandable section with:

Graph Laplacian eigenvalues

Hodge 1‑Laplacian eigenvalues

Persistent component counts

Eigenvector centrality table (all ETFs)

⚙️ Configuration (config.py)
Parameter	Description
CORR_WINDOW	Rolling window for correlation (days)
FILTRATION_STEPS	Steps for persistence computation
PERSISTENCE_THRESHOLD	Threshold to count persistent classes
TOP_N	Number of top ETFs to output (default 3)
