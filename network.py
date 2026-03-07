"""
Agent media reaction and sharing pipeline: k-means clustering, representative
sampling, VADER sentiment → actions, and similarity-based sharing.
"""

import random
from enum import Enum
from typing import Callable, Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Editable: personality vector dimension (e.g. 64 from CompressionModel or 384 from sentence embeddings)
PERSONALITY_DIM = 64

# K-means auto-k range
K_MIN = 2
K_MAX_FRAC = 5  # k_max = min(20, n // K_MAX_FRAC)

# Representative sample fraction
REPRESENTATIVE_FRAC = 0.10

# Similarity: top-K candidates for sharing, same-cluster bonus
TOP_K_SIMILAR = 10
SAME_CLUSTER_BONUS = 0.15

# VADER compound in [-1, 1]; 7 actions need 6 thresholds (equal-width bins)
NUM_ACTIONS = 7
_action_thresholds = np.linspace(-1.0, 1.0, num=NUM_ACTIONS, endpoint=False)[1:]  # 6 thresholds


class Action(Enum):
    """Seven actions from lowest to highest like value."""
    DISLIKE_SHARE_COMMENT = "dislike_share_comment"
    DISLIKE_SHARE = "dislike_share"
    DISLIKE = "dislike"
    NOTHING = "nothing"
    LIKE = "like"
    LIKE_SHARE = "like_share"
    LIKE_SHARE_COMMENT = "like_share_comment"


def _action_includes_share(action: Action) -> bool:
    return action in (
        Action.DISLIKE_SHARE_COMMENT,
        Action.DISLIKE_SHARE,
        Action.LIKE_SHARE,
        Action.LIKE_SHARE_COMMENT,
    )


def _like_value_to_action(compound: float) -> Action:
    """Map VADER compound score to one of 7 actions using 6 thresholds."""
    for i, t in enumerate(_action_thresholds):
        if compound < t:
            return list(Action)[i]
    return list(Action)[NUM_ACTIONS - 1]


def kmeans_auto_k(vectors: np.ndarray, k_range: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run k-means with automatic k selection via silhouette score.
    vectors: (n, dim)
    Returns: (labels shape (n,), centroids (k, dim), k).
    """
    n, _ = vectors.shape
    k_min = K_MIN
    k_max = min(20, max(k_min + 1, n // K_MAX_FRAC))
    if k_range is not None:
        k_min, k_max = k_range
    k_max = min(k_max, n - 1)
    if k_max <= k_min:
        k_best = k_min
        km = KMeans(n_clusters=k_best, random_state=42, n_init=10).fit(vectors)
        return km.labels_, km.cluster_centers_, k_best

    best_score = -2.0
    best_km = None
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)
        score = silhouette_score(vectors, km.labels_, metric="euclidean")
        if score > best_score:
            best_score = score
            best_km = km
    assert best_km is not None
    return best_km.labels_, best_km.cluster_centers_, best_km.n_clusters


def select_representatives(
    uids: list,
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    fraction: float = REPRESENTATIVE_FRAC,
) -> list:
    """
    Select ~fraction of agents that best represent each cluster (closest to centroid).
    Returns list of UIDs of selected representatives.
    """
    n = len(uids)
    k = centroids.shape[0]
    target_count = max(1, int(round(n * fraction)))
    # Per-cluster quota proportional to cluster size; at least 1 per cluster if k <= target_count
    cluster_sizes = np.bincount(labels, minlength=k)
    quotas = np.zeros(k, dtype=int)
    if target_count >= k:
        remaining = target_count
        for c in range(k):
            share = max(0, int(round(cluster_sizes[c] * fraction)))
            share = min(share, cluster_sizes[c], remaining)
            if share < 1 and cluster_sizes[c] >= 1 and remaining >= 1:
                share = 1
            quotas[c] = min(share, cluster_sizes[c])
            remaining -= quotas[c]
        if remaining > 0:
            for c in np.argsort(-cluster_sizes):
                if quotas[c] < cluster_sizes[c] and remaining > 0:
                    add = min(remaining, cluster_sizes[c] - quotas[c])
                    quotas[c] += add
                    remaining -= add
    else:
        quotas[np.argmax(cluster_sizes)] = target_count

    selected_uids = []
    for c in range(k):
        mask = labels == c
        indices = np.where(mask)[0]
        if len(indices) == 0 or quotas[c] == 0:
            continue
        pts = vectors[indices]
        cent = centroids[c : c + 1]
        dists = np.linalg.norm(pts - cent, axis=1)
        take = min(quotas[c], len(indices))
        nearest = indices[np.argsort(dists)[:take]]
        for i in nearest:
            selected_uids.append(uids[i])
    return selected_uids


def get_like_value_from_reaction(reaction_text: str) -> float:
    """VADER compound score in [-1, 1] as like value."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(reaction_text)
    return scores["compound"]


def reaction_to_action(reaction_text: str) -> tuple[float, Action]:
    """Get like value and action for a reaction sentence."""
    like = get_like_value_from_reaction(reaction_text)
    action = _like_value_to_action(like)
    return like, action


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities; (n, n)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def combined_similarity_scores(
    sharer_idx: int,
    vectors: np.ndarray,
    labels: np.ndarray,
    same_cluster_bonus: float = SAME_CLUSTER_BONUS,
) -> np.ndarray:
    """
    For sharer at index sharer_idx, compute combined similarity to all others (n,).
    Exclude self (set to -inf or -1 so it's never chosen). Others: cos_sim + bonus if same cluster.
    """
    n = vectors.shape[0]
    cos = cosine_similarity_matrix(vectors)
    row = cos[sharer_idx].copy()
    row[sharer_idx] = -2.0
    same = labels == labels[sharer_idx]
    same[sharer_idx] = False
    row[same] += same_cluster_bonus
    return row


def pick_recipients(
    sharer_idx: int,
    uids: list,
    vectors: np.ndarray,
    labels: np.ndarray,
    top_k: int = TOP_K_SIMILAR,
    rng: random.Random | None = None,
) -> list:
    """
    For the sharer at index sharer_idx, pick 1–3 recipients from top-K similar users.
    Returns list of recipient UIDs.
    """
    rng = rng or random.Random()
    n = len(uids)
    if n <= 1:
        return []
    scores = combined_similarity_scores(sharer_idx, vectors, labels)
    candidates = np.argsort(-scores)[:top_k]
    num_recipients = rng.randint(1, min(3, len(candidates)))
    chosen_indices = rng.sample(list(candidates), num_recipients)
    return [uids[i] for i in chosen_indices]


def run_media_pipeline(
    uids: list,
    vectors: np.ndarray,
    media: Any,
    reaction_fn: Callable[[Any, Any], str],
    n_clusters: int | None = None,
    fraction: float = REPRESENTATIVE_FRAC,
    seed: int | None = None,
) -> tuple[list, list[tuple[Any, str, float, Action]], list[tuple[Any, list]]]:
    """
    Main entry point.

    Args:
        uids: List of n agent UIDs.
        vectors: (n, PERSONALITY_DIM) array of personality vectors.
        media: Arbitrary object passed to reaction_fn(uid, media).
        reaction_fn: Callable (uid, media) -> reaction sentence (str).
        n_clusters: If set, use this k instead of auto k.
        fraction: Representative sample fraction (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        representative_uids: UIDs of the 10% representatives.
        reactions: List of (uid, reaction_text, like_value, action) for each representative.
        shares: List of (sharer_uid, [recipient_uid, ...]) for each representative who shared.
    """
    rng = random.Random(seed)
    n = len(uids)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[0] != n or vectors.ndim != 2:
        raise ValueError("vectors must be shape (n, dim) with n = len(uids)")

    # 1. K-means
    if n_clusters is not None:
        km = KMeans(n_clusters=min(n_clusters, n), random_state=42, n_init=10).fit(vectors)
        labels = km.labels_
        centroids = km.cluster_centers_
    else:
        labels, centroids, _ = kmeans_auto_k(vectors)

    # 2. Representatives
    rep_uids = select_representatives(uids, vectors, labels, centroids, fraction=fraction)
    uid_to_idx = {uid: i for i, uid in enumerate(uids)}

    # 3. Reactions and actions for representatives
    reactions = []
    for uid in rep_uids:
        text = reaction_fn(uid, media)
        like, action = reaction_to_action(text)
        reactions.append((uid, text, like, action))

    # 4. For each representative who shared, pick 1–3 recipients by similarity
    shares = []
    for uid in rep_uids:
        like, action = next((r[2], r[3]) for r in reactions if r[0] == uid)
        if not _action_includes_share(action):
            continue
        idx = uid_to_idx[uid]
        recipients = pick_recipients(idx, uids, vectors, labels, rng=rng)
        if recipients:
            shares.append((uid, recipients))

    return rep_uids, reactions, shares


# --- Example / testing: use example_100_agents.py ---
# def _stub_reaction(uid: Any, media: Any) -> str:
#     """Stub for testing: returns a placeholder reaction."""
#     return "I think this is okay."
#
# if __name__ == "__main__":
#     n_agents = 50
#     dim = PERSONALITY_DIM
#     rng = np.random.default_rng(42)
#     uids = list(range(n_agents))
#     vectors = rng.standard_normal((n_agents, dim)).astype(np.float64)
#     rep_uids, reactions, shares = run_media_pipeline(
#         uids, vectors, media="some_video", reaction_fn=_stub_reaction, seed=123
#     )
#     print("Representatives:", len(rep_uids), rep_uids[:5], "...")
#     print("Reactions sample:", reactions[0] if reactions else None)
#     print("Shares:", shares[:3] if shares else [])