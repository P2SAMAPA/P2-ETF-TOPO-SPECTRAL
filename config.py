import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-topo-spectral-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Rolling window for correlation (days)
CORR_WINDOW = 30

# Persistence
PERSISTENCE_THRESHOLD = 0.5

# Bias parameters – aggressive chasing
RETURN_SCALE = 500.0      # multiplies average daily return
VOL_SCALE = 200.0         # multiplies daily volatility
MOMENTUM_SCALE = 300.0    # multiplies recent return (5 days)

# Use signed correlation? (false = absolute)
USE_SIGNED_CORR = False

# Topological weight (1 = full topology, 0 = pure return chasing)
TOPOLOGICAL_WEIGHT = 0.3   # keep topology but biases are amplified

MOMENTUM_DAYS = 5
TOP_N = 3
