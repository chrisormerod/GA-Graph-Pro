#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 13:38:24 2025

@author: cormerod
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, linalg
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class MultiLayerRandomWalk:
    """
    Multi-layer random walk implementation for heterogeneous biological networks.
    Supports both bidirectional and unidirectional walks with sparse matrix
    optimization.
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 walk_length: int = 3,
                 num_walks: int = 10,
                 p: float = 1.0,
                 q: float = 1.0,
                 inter_layer_prob: float = 0.1):
        """
        Initialize the multi-layer random walk system.

        Args:
            embedding_dim: Dimension of final embeddings
            walk_length: Maximum length of random walks (2-3 recommended)
            num_walks: Number of random walks per node
            p: Return parameter for biased walks
            q: In-out parameter for biased walks
            inter_layer_prob: Probability of transitioning between layers
        """
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.inter_layer_prob = inter_layer_prob

        self.networks = {}
        self.network_types = {}
        self.embeddings = {}
        self.combined_features = None

    def add_network(self,
                    name: str,
                    adjacency_matrix: Union[np.ndarray, sp.spmatrix],
                    network_type: str = 'bidirectional',
                    confidence_weights: Optional[np.ndarray] = None):
        """
        Add a biological network to the multi-layer system.

        Args:
            name: Network identifier (e.g., 'coprecipitation', 'phosphorylation')
            adjacency_matrix: Network adjacency matrix
            network_type: 'bidirectional' or 'unidirectional'
            confidence_weights: Optional edge confidence scores
        """
        # Convert to sparse CSR format for efficiency
        if not sp.issparse(adjacency_matrix):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        else:
            adjacency_matrix = adjacency_matrix.tocsr()

        # Apply confidence weighting if provided
        if confidence_weights is not None:
            adjacency_matrix = adjacency_matrix.multiply(confidence_weights)

        self.networks[name] = adjacency_matrix
        self.network_types[name] = network_type

        print(f"Added {name} network: {adjacency_matrix.shape}, "
              f"density: {adjacency_matrix.nnz / np.prod(adjacency_matrix.shape):.4f}")

    def _compute_transition_probabilities(self,
                                            network: sp.csr_matrix,
                                            network_type: str) -> sp.csr_matrix:
        """
        Compute transition probability matrix for random walks.

        Args:
            network: Sparse adjacency matrix
            network_type: 'bidirectional' or 'unidirectional'

        Returns:
            Sparse transition probability matrix
        """
        if network_type == 'bidirectional':
            # Symmetric normalization for undirected networks
            network = network + network.T
            network.data = network.data / 2.0  # Avoid double counting

        # Row normalization to get transition probabilities
        row_sums = np.array(network.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero

        # Create diagonal matrix for normalization
        inv_row_sums = sp.diags(1.0 / row_sums, format='csr')
        transition_matrix = inv_row_sums @ network

        return transition_matrix

    def _layer_specific_walks(self,
                                transition_matrix: sp.csr_matrix,
                                start_nodes: List[int],
                                layer_name: str) -> np.ndarray:
        """
        Perform random walks within a specific network layer.

        Args:
            transition_matrix: Sparse transition probability matrix
            start_nodes: List of starting node indices
            layer_name: Name of the network layer

        Returns:
            Walk embeddings for the layer
        """
        n_nodes = transition_matrix.shape[0]
        walk_features = []

        for start_node in start_nodes:
            node_walks = []

            for _ in range(self.num_walks):
                walk = [start_node]
                current_node = start_node

                for step in range(self.walk_length - 1):
                    # Get neighbors and their probabilities
                    neighbors = transition_matrix[current_node].indices
                    probabilities = transition_matrix[current_node].data

                    if len(neighbors) == 0:
                        break

                    # Choose next node based on probabilities
                    next_node = np.random.choice(neighbors,
                                                 p=probabilities / probabilities.sum())
                    walk.append(next_node)
                    current_node = next_node

                # Convert walk to feature vector (node visitation counts)
                walk_vector = np.zeros(n_nodes)
                for node in walk:
                    walk_vector[node] += 1

                node_walks.append(walk_vector)

            # Average walks for this starting node
            avg_walk = np.mean(node_walks, axis=0)
            walk_features.append(avg_walk)

        return np.array(walk_features)

    def _compute_k_hop_features(self,
                                  network: sp.csr_matrix,
                                  k: int = 3) -> np.ndarray:
        """
        Efficiently compute k-hop neighborhood features using sparse operations.

        Args:
            network: Sparse adjacency matrix
            k: Number of hops

        Returns:
            k-hop neighborhood features
        """
        k_hop_neighbors = network.copy()

        for i in range(k - 1):
            k_hop_neighbors = k_hop_neighbors @ network

        # Extract features: degree, clustering, centrality measures
        features = []

        # Degree centrality (normalized)
        degrees = np.array(k_hop_neighbors.sum(axis=1)).flatten()
        features.append(degrees / degrees.max() if degrees.max() > 0 else degrees)

        # k-hop connectivity
        k_hop_conn = np.array(k_hop_neighbors.sum(axis=1)).flatten()
        features.append(k_hop_conn / k_hop_conn.max() if k_hop_conn.max() > 0 else
                        k_hop_conn)

        return np.column_stack(features)

    def _spectral_embedding(self,
                             network: sp.csr_matrix,
                             dim: int = 50) -> np.ndarray:
        """
        Compute spectral embedding of the network using eigendecomposition.

        Args:
            network: Sparse adjacency matrix
            dim: Embedding dimension

        Returns:
            Spectral embedding matrix
        """
        # Compute normalized Laplacian
        degrees = np.array(network.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero

        # D^(-1/2) * A * D^(-1/2)
        inv_sqrt_degrees = sp.diags(1.0 / np.sqrt(degrees), format='csr')
        normalized_adj = inv_sqrt_degrees @ network @ inv_sqrt_degrees

        # Compute eigendecomposition
        try:
            eigenvals, eigenvecs = linalg.eigsh(normalized_adj, k=min(dim,
                                                                      network.shape[0] - 1))
            # Sort by eigenvalue magnitude
            idx = np.argsort(np.abs(eigenvals))[::-1]
            return eigenvecs[:, idx[:dim]]
        except:
            # Fallback to SVD if eigendecomposition fails
            U, s, Vt = linalg.svds(normalized_adj, k=min(dim, min(network.shape) - 1))
            return U

    def _attention_aggregation(self,
                                 layer_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate layer-specific embeddings using attention mechanism.

        Args:
            layer_embeddings: Dictionary of embeddings for each layer

        Returns:
            Aggregated embedding matrix
        """
        layer_names = list(layer_embeddings.keys())
        embeddings_list = [layer_embeddings[name] for name in layer_names]

        # Simple attention: compute attention weights based on embedding variance
        attention_weights = []
        for emb in embeddings_list:
            variance = np.var(emb, axis=0).mean()
            attention_weights.append(variance)

        # Normalize attention weights
        attention_weights = np.array(attention_weights)
        attention_weights = attention_weights / attention_weights.sum()

        # Weighted combination
        aggregated = np.zeros_like(embeddings_list[0])
        for i, emb in enumerate(embeddings_list):
            aggregated += attention_weights[i] * emb

        return aggregated

    def compute_embeddings(self, gene_ids: List[int]) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for all networks and aggregate them.

        Args:
            gene_ids: List of gene node indices

        Returns:
            Dictionary containing layer-specific and combined embeddings
        """
        layer_embeddings = {}

        # Process each network layer
        for network_name, network in self.networks.items():
            print(f"Processing {network_name} network...")

            # Compute transition probabilities
            transition_matrix = self._compute_transition_probabilities(
                network, self.network_types[network_name]
            )

            # Multiple embedding approaches
            embeddings_list = []

            # 1. Random walk embeddings
            if len(gene_ids) <= 1000:  # Only for manageable sizes
                walk_emb = self._layer_specific_walks(
                    transition_matrix, gene_ids, network_name
                )
                embeddings_list.append(walk_emb)

            # 2. k-hop neighborhood features
            khop_features = self._compute_k_hop_features(network, k=5)
            embeddings_list.append(khop_features[gene_ids])

            # 3. Spectral embeddings
            spectral_emb = self._spectral_embedding(network, dim=251)
            embeddings_list.append(spectral_emb[gene_ids])

            # Combine different embedding types
            combined_emb = np.hstack(embeddings_list)

            # Dimensionality reduction to target size
            if combined_emb.shape[1] > self.embedding_dim:
                pca = PCA(n_components=self.embedding_dim)
                combined_emb = pca.fit_transform(combined_emb)

            layer_embeddings[network_name] = combined_emb

        # Aggregate across layers using attention
        if len(layer_embeddings) > 1:
            aggregated_embedding = self._attention_aggregation(layer_embeddings)
        else:
            aggregated_embedding = list(layer_embeddings.values())[0]

        # Store results
        self.embeddings = layer_embeddings
        self.combined_features = aggregated_embedding

        return {
            'layer_embeddings': layer_embeddings,
            'combined_embedding': aggregated_embedding
        }

    def get_gene_features(self, gene_id: int) -> Dict[str, np.ndarray]:
        """
        Get features for a specific gene across all networks.

        Args:
            gene_id: Gene node index

        Returns:
            Dictionary of features for the gene
        """
        features = {}

        for network_name, embedding in self.embeddings.items():
            if gene_id < len(embedding):
                features[network_name] = embedding[gene_id]

        if self.combined_features is not None and gene_id < len(self.combined_features):
            features['combined'] = self.combined_features[gene_id]

        return features

    def save_embeddings(self, filepath: str):
        """Save computed embeddings to file."""
        np.savez_compressed(
            filepath,
            combined_features=self.combined_features,
            **self.embeddings
        )
        print(f"Embeddings saved to {filepath}")

    def load_embeddings(self, filepath: str):
        """Load embeddings from file."""
        data = np.load(filepath)
        self.combined_features = data['combined_features']
        self.embeddings = {key: data[key] for key in data.keys()
                           if key != 'combined_features'}
        print(f"Embeddings loaded from {filepath}")


class NetworkSampler:
    """
    Utility class for network sampling to reduce computational burden.
    """

    @staticmethod
    def importance_sampling(network: sp.csr_matrix,
                            disease_genes: List[int],
                            sample_ratio: float = 0.1) -> sp.csr_matrix:
        """
        Sample network based on biological importance (disease gene proximity).

        Args:
            network: Full network adjacency matrix
            disease_genes: List of known disease gene indices
            sample_ratio: Fraction of nodes to retain

        Returns:
            Sampled network
        """
        n_nodes = network.shape[0]
        n_sample = int(n_nodes * sample_ratio)

        # Compute distance to disease genes
        disease_mask = np.zeros(n_nodes, dtype=bool)
        disease_mask[disease_genes] = True

        # Include all disease genes
        important_nodes = set(disease_genes)

        # Add high-degree nodes (hubs)
        degrees = np.array(network.sum(axis=1)).flatten()
        hub_threshold = np.percentile(degrees, 90)
        hub_nodes = np.where(degrees >= hub_threshold)[0]
        important_nodes.update(hub_nodes)

        # Fill remaining slots randomly
        remaining_slots = n_sample - len(important_nodes)
        if remaining_slots > 0:
            all_nodes = set(range(n_nodes))
            remaining_nodes = list(all_nodes - important_nodes)
            random_nodes = np.random.choice(
                remaining_nodes,
                size=min(remaining_slots, len(remaining_nodes)),
                replace=False
            )
            important_nodes.update(random_nodes)

        # Extract subnetwork
        node_list = sorted(list(important_nodes))
        sampled_network = network[np.ix_(node_list, node_list)]

        return sampled_network, node_list

    @staticmethod
    def degree_based_sampling(network: sp.csr_matrix,
                              min_degree: int = 1) -> sp.csr_matrix:
        """
        Remove low-degree nodes to focus on well-connected genes.

        Args:
            network: Network adjacency matrix
            min_degree: Minimum degree threshold

        Returns:
            Filtered network
        """
        degrees = np.array(network.sum(axis=1)).flatten()
        keep_nodes = np.where(degrees >= min_degree)[0]

        filtered_network = network[np.ix_(keep_nodes, keep_nodes)]
        return filtered_network, keep_nodes

