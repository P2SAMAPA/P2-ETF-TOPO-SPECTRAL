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

# Rolling window for correlation (days) – shorter = more responsive for return chasing
CORR_WINDOW = 30   # was 60

# Persistence thresholds
FILTRATION_STEPS = 50
PERSISTENCE_THRESHOLD = 0.5

# Bias parameters: increase these to chase returns/volatility
RETURN_BIAS = 2.0        # multiplies score by ETF's average daily return (positive only)
VOLATILITY_BIAS = 1.0    # multiplies score by ETF's volatility (0 = ignore)
MOMENTUM_DAYS = 5        # recent return window for momentum
MOMENTUM_BIAS = 1.5      # multiplies score by recent return

# Use signed correlation (True = keep negative edges, may increase risk awareness)
USE_SIGNED_CORR = False   # keep False for stability, True for chasing opposite moves

TOP_N = 3
