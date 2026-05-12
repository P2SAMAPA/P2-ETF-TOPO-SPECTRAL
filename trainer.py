import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from topological_engine import TopoSpectralEngine

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        # Prepare returns matrix for universe
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.CORR_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        engine = TopoSpectralEngine(returns, corr_window=config.CORR_WINDOW)
        try:
            result = engine.run()
        except Exception as e:
            print(f"  Error: {e}")
            all_results[universe_name] = {"top_etfs": []}
            continue

        scores = result["scores"]
        assets = result["assets"]
        # Sort by score descending, top N
        sorted_idx = np.argsort(scores)[::-1]
        top_etfs = [{"ticker": assets[i], "score": float(scores[i])} for i in sorted_idx[:config.TOP_N]]
        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]} (scores: {[e['score'] for e in top_etfs]})")

        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "metadata": {
                "centrality": {assets[i]: float(result["centrality"][i]) for i in range(len(assets))},
                "lap_eigs": result["lap_eigs"],
                "hodge_eigs": result["hodge_eigs"],
                "persistence_0": result["persistence_0"],
                "persistence_1": result["persistence_1"]
            },
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/topo_spectral_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Persistent Spectral + Hodge Laplacian Engine complete ===")

if __name__ == "__main__":
    main()
