"""
Consolidated example and testing entry point for network.py and kmean_graph.py.
Run from project root: python testing/example_100_agents.py
Or from testing/: python example_100_agents.py (adds project root to path)
"""
import sys
from pathlib import Path

# Ensure project root is on path when run from testing/
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from network import (
    PERSONALITY_DIM,
    run_media_pipeline,
    kmeans_auto_k,
)

# Placeholder media (any object; only passed through to reaction_fn)
PLACEHOLDER_MEDIA = "placeholder_media"


REACTION_OPTIONS = [
    "I love this! So amazing and inspiring.",
    "This is fantastic, everyone should see it.",
    "Really great content, worth sharing.",
    "Pretty good, I enjoyed it.",
    "It's okay, nothing special.",
    "Not my thing, but whatever.",
    "Kind of boring to be honest.",
    "I did not like this at all.",
    "This is terrible and misleading.",
    "Awful. Do not watch.",
]


def generate_clustered_vectors(
    n_agents: int,
    dim: int,
    n_clusters: int = 5,
    cluster_std: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate personality vectors from a mixture of Gaussians so k-means finds
    real cluster structure. Each cluster has a random center; agents are
    assigned to clusters and get center + Gaussian noise.
    """
    rng = rng or np.random.default_rng()
    # Well-separated cluster centers in dim dimensions
    centers = rng.standard_normal((n_clusters, dim)) * 2.0
    # Assign each agent to a cluster (roughly balanced)
    assignment = rng.integers(0, n_clusters, size=n_agents)
    vectors = centers[assignment] + rng.standard_normal((n_agents, dim)) * cluster_std
    return vectors.astype(np.float64)


def main():
    n_agents = 100
    # No seed: different agents every run
    rng = np.random.default_rng()
    uids = list(range(n_agents))
    # Clustered data so k-means finds multiple groups
    vectors = generate_clustered_vectors(
        n_agents, PERSONALITY_DIM, n_clusters=5, cluster_std=0.4, rng=rng
    )

    # Shuffle which reaction text maps to which uid so reactions vary each run
    reaction_order = rng.permutation(len(REACTION_OPTIONS))

    def placeholder_reaction(uid, media):
        return REACTION_OPTIONS[reaction_order[uid % len(REACTION_OPTIONS)]]

    print("Running pipeline: 100 agents, placeholder media, placeholder reactions...")
    rep_uids, reactions, shares = run_media_pipeline(
        uids,
        vectors,
        media=PLACEHOLDER_MEDIA,
        reaction_fn=placeholder_reaction,
        seed=None,  # different share recipients etc. each run
    )

    print(f"Representatives: {len(rep_uids)} (10% of {n_agents})")
    print(f"Sample UIDs: {rep_uids[:8]} ...")
    print(f"Reactions (uid, text, like_value, action):")
    for r in reactions[:5]:
        print(f"  {r[0]}: like={r[2]:.3f} -> {r[3].name}")
    print(f"Shares (sharer -> recipients): {len(shares)} sharers")
    for sh in shares[:5]:
        print(f"  {sh[0]} -> {sh[1]}")

    # K-means labels for all agents (same vectors => same clustering as in pipeline)
    labels, _, k = kmeans_auto_k(vectors)
    print(f"\nK-means: k={k} clusters")

    # kmean_graph: coordinates and edges
    from kmean_graph import get_clustering_output

    coords, edges = get_clustering_output(uids, vectors, labels=labels, shares=shares)
    print(f"\nCoords shape: {coords.shape}")
    print("Edges (start_uid, end_uid):", edges)

    # Plot: 2D points by cluster + share arrows
    import matplotlib.pyplot as plt

    plt.rcParams["figure.figsize"] = (8, 6)
    fig, ax = plt.subplots()
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    n_clusters = int(labels.max()) + 1
    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=f"Cluster {c}",
            alpha=0.7,
            s=30,
        )
    for start_uid, end_uid in edges:
        i = uid_to_idx.get(start_uid)
        j = uid_to_idx.get(end_uid)
        if i is None or j is None:
            continue
        ax.annotate(
            "",
            xy=(coords[j, 0], coords[j, 1]),
            xytext=(coords[i, 0], coords[i, 1]),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-means clustering (2D PCA) and share directions")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    out_path = Path(__file__).resolve().parent / "kmeans_graph_100.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
