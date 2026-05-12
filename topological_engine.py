"""
Persistent Spectral Graph + Hodge Laplacian Engine.
Computes graph Laplacian eigenvalues, Hodge 1-Laplacian eigenvalues,
persistent homology barcodes, and a return‑biased topological score.
"""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import networkx as nx
import gudhi as gd
import config

class TopoSpectralEngine:
    def __init__(self, returns_df, corr_window=60, filtration_steps=50):
        self.returns = returns_df
        self.corr_window = corr_window
        self.filtration_steps = filtration_steps
        self.n_assets = returns_df.shape[1]
        self.assets = returns_df.columns.tolist()

    def compute_correlation_graph(self):
        """Weighted graph from rolling correlation (last `corr_window` days)."""
        if len(self.returns) < self.corr_window:
            raise ValueError("Not enough data")
        recent = self.returns.iloc[-self.corr_window:]
        corr = recent.corr().abs().values if not config.USE_SIGNED_CORR else recent.corr().values
        adj = corr - np.eye(self.n_assets)
        G = nx.from_numpy_array(adj, create_using=nx.Graph())
        if config.USE_SIGNED_CORR:
            edges_to_remove = [(u, v) for u, v, w in G.edges(data=True) if w['weight'] <= 0]
            G.remove_edges_from(edges_to_remove)
        return G, adj

    def graph_laplacian_eigenvalues(self, G, k=5):
        """Compute smallest k eigenvalues of graph Laplacian (excluding zero)."""
        L = nx.laplacian_matrix(G).astype(float)
        L_dense = L.toarray()
        eigvals = np.linalg.eigvalsh(L_dense)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[:min(k, len(eigvals))]

    def hodge_1_laplacian_eigenvalues(self, G):
        """Compute eigenvalues of 1st-order Hodge Laplacian (edge Laplacian)."""
        edges = list(G.edges())
        if not edges:
            return []
        n_nodes = G.number_of_nodes()
        n_edges = len(edges)
        B = np.zeros((n_nodes, n_edges))
        for i, (u, v) in enumerate(edges):
            B[u, i] = 1.0
            B[v, i] = -1.0
        L1_simple = B.T @ B
        eigvals = np.linalg.eigvalsh(L1_simple)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[:5].tolist()

    def persistent_homology(self, adj):
        """Compute persistence barcodes for graph filtration."""
        dist = 1 - np.abs(adj)
        rips = gd.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
        barcode_0 = [interval for dim, interval in persistence if dim == 0]
        barcode_1 = [interval for dim, interval in persistence if dim == 1]
        thresh = config.PERSISTENCE_THRESHOLD
        count_0 = sum(1 for (b, d) in barcode_0 if b < thresh and (d == float('inf') or d > thresh))
        count_1 = sum(1 for (b, d) in barcode_1 if b < thresh and (d == float('inf') or d > thresh))
        return count_0, count_1

    def eigenvector_centrality(self, G):
        """Compute eigenvector centrality for each node."""
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight='weight')
        except:
            cent = {n: 1.0/self.n_assets for n in G.nodes}
        return [cent[i] for i in range(self.n_assets)]

    def run(self):
        """Compute spectral/topological features and return return‑biased scores."""
        G, adj = self.compute_correlation_graph()
        lap_eigs = self.graph_laplacian_eigenvalues(G, k=3)
        hodge_eigs = self.hodge_1_laplacian_eigenvalues(G)
        perc_0, perc_1 = self.persistent_homology(adj)
        cent = self.eigenvector_centrality(G)

        # ---- Convert cent to numpy array (fixes the multiplication error) ----
        cent = np.array(cent)

        # ----- Return / volatility / momentum biases -----
        recent = self.returns.iloc[-self.corr_window:]
        avg_return = recent.mean().values
        volatility = recent.std().values
        momentum = self.returns.iloc[-config.MOMENTUM_DAYS:].mean().values

        avg_return = np.maximum(avg_return, 0)
        momentum = np.maximum(momentum, 0)

        if len(lap_eigs) > 0:
            connectivity_factor = 1.0 / (1.0 + lap_eigs[0] * 10)
        else:
            connectivity_factor = 1.0

        loop_boost = 1 + config.PERSISTENCE_THRESHOLD * perc_1

        return_bias = 1 + config.RETURN_BIAS * avg_return
        vol_bias = 1 + config.VOLATILITY_BIAS * volatility
        mom_bias = 1 + config.MOMENTUM_BIAS * momentum

        # All terms are now numpy arrays or scalars
        scores = cent * connectivity_factor * loop_boost * return_bias * vol_bias * mom_bias

        return {
            "scores": scores,
            "centrality": cent.tolist(),
            "lap_eigs": lap_eigs.tolist(),
            "hodge_eigs": hodge_eigs,
            "persistence_0": perc_0,
            "persistence_1": perc_1,
            "assets": self.assets,
            "avg_returns": avg_return.tolist(),
            "volatility": volatility.tolist(),
            "momentum": momentum.tolist()
        }
