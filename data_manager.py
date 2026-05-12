"""
Data loading and return preparation for topological‑spectral engine.
"""
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    """Load master_data.parquet from Hugging Face."""
    path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=config.HF_TOKEN
    )
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, universe_tickers):
    """
    Extract log returns for given tickers.
    Returns DataFrame with index = date, columns = tickers.
    """
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    # Drop rows with all NaN (if any)
    returns = returns.dropna(how='all')
    return returns
