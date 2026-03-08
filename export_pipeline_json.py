"""
Run the full pipeline and kmean_graph, then export results to JSON for the dashboard (index.html).

Usage: python export_pipeline_json.py [--out pipeline_results.json] [--seed 42]
Creates pipeline_results.json with agents, rep_uids, reactions, shares, graph (nodes + links),
action_counts, and chart-friendly series (neutral/skeptic/polar derived from like_value).
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from kmean_graph import get_clustering_output
from network import kmeans_auto_k
from run_full_pipeline import main as run_full_pipeline_main


def action_to_display_group(action_name: str) -> str:
    """Map pipeline Action (enum name) to chart bucket: neutral, skeptical, polarized."""
    if action_name in ("LIKE", "LIKE_SHARE", "LIKE_SHARE_COMMENT"):
        return "neutral"
    if action_name == "NOTHING":
        return "skeptical"
    return "polarized"  # DISLIKE, DISLIKE_SHARE, DISLIKE_SHARE_COMMENT


def export_pipeline_results(
    out_path: Path | str = Path("pipeline_results.json"),
    media: str | None = None,
    seed: int = 42,
) -> Path:
    """
    Run full pipeline, get 2D coords and edges from kmean_graph, write JSON for index.html.
    """
    out_path = Path(out_path)
    run_kwargs = {"seed": seed}
    if media is not None:
        run_kwargs["media"] = media

    agents, rep_uids, reactions, shares = run_full_pipeline_main(**run_kwargs)

    uids = [a.uid for a in agents]
    vectors = np.stack([a.personality_vector for a in agents], axis=0).astype(np.float64)
    labels, _, _ = kmeans_auto_k(vectors)
    coords, edges = get_clustering_output(
        uids, vectors, labels=labels, shares=shares, pca_random_state=42
    )

    # Action counts for stats and reaction bar
    action_counts = {}
    for a in agents:
        name = a.action.name if hasattr(a.action, "name") else str(a.action).split(".")[-1]
        action_counts[name] = action_counts.get(name, 0) + 1

    # Map to display categories for charts (neutral / skeptical / polarized)
    by_t = {"neutral": 0, "skeptical": 0, "polarized": 0}
    for a in agents:
        name = a.action.name if hasattr(a.action, "name") else str(a.action).split(".")[-1]
        by_t[action_to_display_group(name)] += 1
    total = len(agents)
    neutral_pct = [round(100 * by_t["neutral"] / total, 1)] * 5
    skeptic_pct = [round(100 * by_t["skeptical"] / total, 1)] * 5
    polar_pct = [round(100 * by_t["polarized"] / total, 1)] * 5

    # Graph nodes: id, x, y, cluster, action (for color), shared (bool)
    sharer_uids = {sh[0] for sh in shares}
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}
    nodes = []
    def node_display_action(action_name: str) -> str:
        if action_name == "LIKE":
            return "liked"
        if action_name in ("DISLIKE", "DISLIKE_SHARE", "DISLIKE_SHARE_COMMENT"):
            return "disliked"
        if action_name in ("LIKE_SHARE", "LIKE_SHARE_COMMENT", "DISLIKE_SHARE", "DISLIKE_SHARE_COMMENT"):
            return "shared"
        if action_name == "NOTHING":
            return "none"
        return "neutral"

    for i, uid in enumerate(uids):
        a = agents[i]
        action_name = a.action.name if hasattr(a.action, "name") else str(a.action).split(".")[-1]
        nodes.append({
            "id": uid,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": int(labels[i]),
            "action": action_name,
            "display_action": node_display_action(action_name),
            "like_value": float(a.like_value),
            "similarity_score": float(a.similarity_score),
            "shared": uid in sharer_uids,
            "description": a.description,
            "description_short": (a.description[:150] + "…") if len(a.description) > 150 else a.description,
            "response": a.response_to_media,
            "response_short": (a.response_to_media[:150] + "…") if len(a.response_to_media) > 150 else a.response_to_media,
        })

    links = [{"source": int(s), "target": int(t)} for s, t in edges]

    # Stats: impressions = 100, likes/dislikes/shares/comments from action_counts
    def get_count(*keys):
        return sum(action_counts.get(k, 0) for k in keys)

    stats = {
        "impressions": total,
        "likes": get_count("LIKE", "LIKE_SHARE", "LIKE_SHARE_COMMENT"),
        "dislikes": get_count("DISLIKE", "DISLIKE_SHARE", "DISLIKE_SHARE_COMMENT"),
        "shares": get_count("LIKE_SHARE", "LIKE_SHARE_COMMENT", "DISLIKE_SHARE", "DISLIKE_SHARE_COMMENT"),
        "comments": get_count("LIKE_SHARE_COMMENT", "DISLIKE_SHARE_COMMENT"),
        "nothing": get_count("NOTHING"),
    }

    # Reaction bar: mutually exclusive segments that sum to 100 (match actual agent counts)
    rxn_total = total
    rxn = {
        "liked": round(100 * get_count("LIKE") / rxn_total, 1),
        "disliked": round(100 * get_count("DISLIKE") / rxn_total, 1),
        "shared": round(100 * get_count("LIKE_SHARE", "DISLIKE_SHARE") / rxn_total, 1),
        "comment": round(100 * get_count("LIKE_SHARE_COMMENT", "DISLIKE_SHARE_COMMENT") / rxn_total, 1),
        "none": round(100 * get_count("NOTHING") / rxn_total, 1),
    }

    payload = {
        "agents": [
            {
                "uid": a.uid,
                "action": a.action.name if hasattr(a.action, "name") else str(a.action).split(".")[-1],
                "like_value": float(a.like_value),
                "similarity_score": float(a.similarity_score),
                "description": a.description,
                "response_to_media": a.response_to_media,
                "cluster": int(labels[i]),
            }
            for i, a in enumerate(agents)
        ],
        "rep_uids": rep_uids,
        "reactions": [
            [r[0], r[1], float(r[2]), r[3].name if hasattr(r[3], "name") else str(r[3]).split(".")[-1]]
            for r in reactions
        ],
        "shares": [[int(s[0]), [int(x) for x in s[1]]] for s in shares],
        "graph": {"nodes": nodes, "links": links},
        "action_counts": action_counts,
        "stats": stats,
        "reaction_bar": rxn,
        "chart": {
            "labels": ["T=0s", "T=5s", "T=10s", "Correction", "T=60s"],
            "neutral": neutral_pct,
            "skeptic": skeptic_pct,
            "polar": polar_pct,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Exported pipeline results to {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export pipeline + kmean graph to JSON for dashboard")
    p.add_argument("--out", type=Path, default=Path("pipeline_results.json"), help="Output JSON path")
    p.add_argument("--media", type=str, default=None, help="Media text (default from pipeline)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    export_pipeline_results(out_path=args.out, media=args.media, seed=args.seed)
