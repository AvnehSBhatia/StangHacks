"""
Run the full pipeline (run_full_pipeline) and plot the k-means graph (kmean_graph).

Uses the same pattern as testing/example_100_agents.py but with real agents from
run_full_pipeline: 100 agents from descriptions → personality vectors → network →
2D PCA coordinates and share edges → matplotlib plot.

Run from project root: python plot_pipeline_kmeans.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kmean_graph import get_clustering_output
from network import kmeans_auto_k
from run_full_pipeline import main as run_full_pipeline_main


def plot_pipeline_kmeans(
    media: str | None = None,
    seed: int = 42,
    out_path: Path | str | None = None,
    show: bool = True,
) -> Path:
    """
    Run full pipeline, get 2D coords and share edges from kmean_graph, plot and save.

    Returns:
        Path to the saved figure.
    """
    # 1. Run full pipeline (100 agents, reactions, shares)
    run_kwargs: dict = {"seed": seed}
    if media is not None:
        run_kwargs["media"] = media
    agents, rep_uids, reactions, shares = run_full_pipeline_main(**run_kwargs)

    # 2. Build uids and personality vectors (same order as pipeline)
    uids = [a.uid for a in agents]
    vectors = np.stack([a.personality_vector for a in agents], axis=0).astype(np.float64)

    # 3. K-means labels (same as used inside run_media_pipeline for sharing)
    labels, _, k = kmeans_auto_k(vectors)

    # 4. 2D coordinates and share edges via kmean_graph
    coords, edges = get_clustering_output(
        uids,
        vectors,
        labels=labels,
        shares=shares,
        pca_random_state=42,
    )

    # 5. Plot: points by cluster + share arrows (same style as example_100_agents)
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
    ax.set_title("K-means clustering (2D PCA) and share directions — full pipeline")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()

    if out_path is None:
        out_path = Path(__file__).resolve().parent / "kmeans_graph_pipeline.png"
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Run full pipeline and plot k-means graph (2D PCA + share edges)"
    )
    p.add_argument("--media", type=str, default=None, help="Media text (default from run_full_pipeline)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=Path, default=None, help="Output image path (default: kmeans_graph_pipeline.png)")
    p.add_argument("--no-show", action="store_true", help="Do not show the plot window (only save file)")
    args = p.parse_args()

    plot_pipeline_kmeans(
        media=args.media,
        seed=args.seed,
        out_path=args.out,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
