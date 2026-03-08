"""
K-means clustering uses full-dimensional personality vectors (all dims).
2D coordinates here are for visualization only (PCA); clustering is never done in 2D.
"""

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from network import kmeans_auto_k


def get_2d_coordinates(vectors: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    For visualization only: reduce (n, dim) to (n, 2) via PCA.
    Clustering is always done in full dimension; this does not affect k-means.
    Returns: (n, 2) array of coordinates; row i corresponds to agent i.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[1] < 2:
        return np.hstack([vectors, np.zeros((vectors.shape[0], 1))])
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(vectors)


def shares_to_edges(shares: list[tuple[Any, list]]) -> list[tuple[Any, Any]]:
    """
    Convert shares from pipeline format to a list of directed edges.
    shares: list of (sharer_uid, [recipient_uid, ...])
    Returns: list of (start_uid, end_uid) for each share.
    """
    return [(sharer_uid, rec_uid) for sharer_uid, rec_list in shares for rec_uid in rec_list]

def get_clustering_output(
    uids: list,
    vectors: np.ndarray,
    labels: np.ndarray | None = None,
    shares: list[tuple[Any, list]] | None = None,
    pca_random_state: int = 42,
) -> tuple[np.ndarray, list[tuple[Any, Any]]]:
    """
    Compute 2D coordinates (for plotting) and share edges. K-means uses all dimensions.

    Clustering (if labels is None) is run on full vectors, not on 2D.
    2D is only for visualization.

    Args:
        uids: List of n agent UIDs (same order as vectors).
        vectors: (n, dim) full-dimensional personality vectors; k-means uses all dims.
        labels: (n,) cluster label per agent. If None, k-means is run on vectors (all dims).
        shares: List of (sharer_uid, [recipient_uid, ...]) from run_media_pipeline. Optional.
        pca_random_state: Seed for PCA (used only for 2D coords, not for clustering).

    Returns:
        coords: (n, 2) array for visualization; coords[i] is the 2D position for uids[i].
        edges: List of (start_uid, end_uid) for each share.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(uids)
    if vectors.shape[0] != n:
        raise ValueError("len(uids) must equal vectors.shape[0]")

    if labels is None:
        labels, _, _ = kmeans_auto_k(vectors)  # full-dimensional clustering

    coords = get_2d_coordinates(vectors, random_state=pca_random_state)  # 2D only for plotting
    edges = shares_to_edges(shares) if shares else []
    return coords, edges
