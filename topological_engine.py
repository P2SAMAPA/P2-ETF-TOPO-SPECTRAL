"""
Persistent Spectral Graph + Hodge Laplacian Engine – Return‑Chasing Mode (Solution 2).
Topology is kept, but return/volatility/momentum biases are multiplied by large scales.
"""
import numpy as np
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

    def graph_laplacian_eigenvalues(self, G, k=3):
        L = nx.laplacian_matrix(G).astype(float).toarray()
        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[:min(k, len(eigvals))]

    def hodge_1_laplacian_eigenvalues(self, G):
        edges = list(G.edges())
        if not edges:
            return []
        n_nodes = G.number_of_nodes()
        n_edges = len(edges)
        B = np.zeros((n_nodes, n_edges))
        for i, (u, v) in enumerate(edges):
            B[u, i] = 1.0
            B[v, i] = -1.0
        L1 = B.T @ B
        eigvals = np.linalg.eigvalsh(L1)
        eigvals = np.sort(eigvals)
        return eigvals[eigvals > 1e-6][:5].tolist()

    def persistent_homology(self, adj):
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
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight='weight')
        except:
            cent = {n: 1.0/self.n_assets for n in G.nodes}
        cent_list = [cent[i] for i in range(self.n_assets)]
        return np.array(cent_list)   # already numpy

    def run(self):
        G, adj = self.compute_correlation_graph()
        lap_eigs = self.graph_laplacian_eigenvalues(G, k=3)
        hodge_eigs = self.hodge_1_laplacian_eigenvalues(G)
        perc_0, perc_1 = self.persistent_homology(adj)
        cent = self.eigenvector_centrality(G)

        # ----- Topological part -----
        if len(lap_eigs) > 0:
            connectivity_factor = 1.0 / (1.0 + lap_eigs[0] * 10)
        else:
            connectivity_factor = 1.0
        loop_boost = 1 + config.PERSISTENCE_THRESHOLD * perc_1
        topo_score = cent * connectivity_factor * loop_boost

        # ----- Return / volatility / momentum biases (aggressive) -----
        recent = self.returns.iloc[-self.corr_window:]
        avg_return = recent.mean().values
        volatility = recent.std().values
        momentum = self.returns.iloc[-config.MOMENTUM_DAYS:].mean().values

        # Direct multiplication (no baseline)
        return_bias = avg_return * config.RETURN_SCALE
        vol_bias = volatility * config.VOL_SCALE
        mom_bias = momentum * config.MOMENTUM_SCALE

        # Combined bias (sum of three)
        bias_score = return_bias + vol_bias + mom_bias

        # Final score: topological part + bias (with optional weight)
        scores = config.TOPOLOGICAL_WEIGHT * topo_score + (1 - config.TOPOLOGICAL_WEIGHT) * bias_score

        # For diagnostic, also keep separate components
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
