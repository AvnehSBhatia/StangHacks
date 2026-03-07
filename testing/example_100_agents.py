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


def placeholder_reaction(uid, media):
    """
    Placeholder reaction: returns sentences with varied sentiment so that
    some agents get share-related actions (like_share, dislike_share, etc.).
    """
    # Mix of reactions so VADER gives a spread; use uid for deterministic variety
    options = [
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
    return options[uid % len(options)]


def main():
    n_agents = 100
    rng = np.random.default_rng(42)
    uids = list(range(n_agents))
    vectors = rng.standard_normal((n_agents, PERSONALITY_DIM)).astype(np.float64)

    print("Running pipeline: 100 agents, placeholder media, placeholder reactions...")
    rep_uids, reactions, shares = run_media_pipeline(
        uids,
        vectors,
        media=PLACEHOLDER_MEDIA,
        reaction_fn=placeholder_reaction,
        seed=123,
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

    # kmean_graph: coordinates and edges only (no graph)
    from kmean_graph import get_clustering_output

    coords, edges = get_clustering_output(uids, vectors, labels=labels, shares=shares)
    print(f"\nCoords shape: {coords.shape}")
    print("Edges (start_uid, end_uid):", edges)


if __name__ == "__main__":
    main()
