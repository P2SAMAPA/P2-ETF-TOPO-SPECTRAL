import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix
import networkx as nx
import gudhi as gd

class TopoSpectralEngine:
    def __init__(self, returns_df, corr_window=60, filtration_steps=50):
        self.returns = returns_df
        self.corr_window = corr_window
        self.filtration_steps = filtration_steps
        self.n_assets = returns_df.shape[1]
        self.assets = returns_df.columns.tolist()

    def compute_correlation_graph(self):
        """Weighted graph from rolling correlation (last `corr_window` days)."""
        # Use all data to get last window correlation
        if len(self.returns) < self.corr_window:
            raise ValueError("Not enough data")
        recent = self.returns.iloc[-self.corr_window:]
        corr = recent.corr().abs().values
        # Adjacency matrix (weighted)
        adj = corr - np.eye(self.n_assets)  # remove self
        G = nx.from_numpy_array(adj, create_using=nx.Graph())
        # keep only positive weights
        edges_to_remove = [(u,v) for u,v,w in G.edges(data=True) if w['weight'] <= 0]
        G.remove_edges_from(edges_to_remove)
        return G, adj

    def graph_laplacian_eigenvalues(self, G, k=5):
        """Compute smallest k eigenvalues of graph Laplacian (excluding 0)."""
        L = nx.laplacian_matrix(G).astype(float)
        # Compute largest eigenvalues first? We need smallest > 0.
        # Use scipy.sparse.linalg.eigs with which='SM' is slow. Use numpy dense for small graphs.
        L_dense = L.toarray()
        eigvals = np.linalg.eigvalsh(L_dense)
        eigvals = np.sort(eigvals)
        # Filter out near-zero (numerical)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[:min(k, len(eigvals))]

    def hodge_1_laplacian_eigenvalues(self, G):
        """Compute eigenvalues of 1st-order Hodge Laplacian (edge Laplacian)."""
        # Build incidence matrix B (nodes x edges)
        edges = list(G.edges())
        if not edges:
            return []
        n_nodes = G.number_of_nodes()
        n_edges = len(edges)
        B = np.zeros((n_nodes, n_edges))
        for i, (u, v) in enumerate(edges):
            B[u, i] = 1.0
            B[v, i] = -1.0
        # Weighted? Use weights from graph
        W = np.diag([G[u][v]['weight'] for u,v in edges])
        L1 = B @ W @ B.T  # edge Laplacian (sometimes defined differently, but this is standard)
        # More common: L1 = B.T @ B (for unweighted). We'll use B.T @ B for simplicity.
        L1_simple = B.T @ B
        eigvals = np.linalg.eigvalsh(L1_simple)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[eigvals > 1e-6]
        return eigvals[:5]  # return top few

    def persistent_homology(self, adj):
        """Compute persistence barcodes for graph filtration."""
        # Rips complex from correlation matrix (weighted graph)
        # Use distance = 1 - corr (so that high correlation = small distance)
        dist = 1 - adj
        # Convert to lower triangular list for Rips
        # Create point cloud? For graph, we can use a Rips complex from distances between nodes.
        # Alternative: use graph filtration by threshold on edge weights.
        # We'll use gudhi's RipsComplex on the distance matrix.
        rips = gd.RipsComplex(distance_matrix=dist, max_edge_length=1.0)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        # Compute persistence
        persistence = simplex_tree.persistence()
        # Extract intervals for dimension 0 and 1
        barcode_0 = [interval for dim, interval in persistence if dim == 0]
        barcode_1 = [interval for dim, interval in persistence if dim == 1]
        # Count intervals that persist beyond threshold (birth < threshold)
        # We'll use birth and death; for infinite death we treat as > threshold
        count_0 = sum(1 for (b,d) in barcode_0 if b < 0.5 and (d == float('inf') or d > 0.5))
        count_1 = sum(1 for (b,d) in barcode_1 if b < 0.5 and (d == float('inf') or d > 0.5))
        return count_0, count_1

    def eigenvector_centrality(self, G):
        """Compute eigenvector centrality for each node."""
        try:
            cent = nx.eigenvector_centrality_numpy(G, weight='weight')
        except:
            cent = {n: 1.0/self.n_assets for n in G.nodes}
        return [cent[i] for i in range(self.n_assets)]

    def run(self):
        """Compute all spectral/topological features and return scores per ETF."""
        G, adj = self.compute_correlation_graph()
        # Graph Laplacian eigenvalues
        lap_eigs = self.graph_laplacian_eigenvalues(G, k=3)
        # Hodge 1-Laplacian eigenvalues
        hodge_eigs = self.hodge_1_laplacian_eigenvalues(G)
        # Persistence barcodes
        perc_0, perc_1 = self.persistent_homology(adj)
        # Eigenvector centrality
        cent = self.eigenvector_centrality(G)
        # Combine into a single score: higher centrality * (more persistent loops) 
        # and penalise high eigenvalue change (we'll use lap_eigs[0] as algebraic connectivity)
        # We'll compute a ranking per ETF directly.
        # For simplicity, we output raw centrality and metrics; ranking will be done by highest centrality.
        # But we want to emphasise topological features: we can weight centrality by (1 + perc_1) / (1 + lap_eigs[0])
        # to boost ETFs in graphs with many persistent cycles and low connectivity (more fragile).
        # Let's produce a score = centrality * (1 + perc_1) * (1 / (1 + lap_eigs[0]))   (if lap_eigs exists)
        if len(lap_eigs) > 0:
            connectivity_factor = 1.0 / (1.0 + lap_eigs[0] * 10)  # small lambda2 -> high factor
        else:
            connectivity_factor = 1.0
        loop_boost = 1 + perc_1 * 0.5
        scores = [c * connectivity_factor * loop_boost for c in cent]
        # Also include centrality as raw
        return {
            "scores": scores,
            "centrality": cent,
            "lap_eigs": lap_eigs.tolist() if len(lap_eigs) else [],
            "hodge_eigs": hodge_eigs,
            "persistence_0": perc_0,
            "persistence_1": perc_1,
            "assets": self.assets
        }
