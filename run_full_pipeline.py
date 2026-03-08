"""
Full pipeline: 100 agents from descriptions → 384D persona vectors → network.py.

Flow:
1. Load 100 agent descriptions (from hybrid_review_dataset or fallback).
2. Compute 384D personality vectors via preprocessor(embed(description)).
3. Run network.run_media_pipeline with 100 UIDs and vectors.
4. Reaction for each agent uses the answer predictor (description + media as question).

Prerequisites:
- preprocessor_checkpoint.pt (run preprocessor build or train pipeline)
- answer_predictor_checkpoint.pt (run train_answer_predictor.py)
- hybrid_review_dataset.json with at least 100 responses (for descriptions)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embedding_engine import EmbeddingEngine
from network import Action, get_like_value_from_reaction, kmeans_auto_k, like_value_to_action, run_media_pipeline
from preprocessor import PREPROCESSOR_CHECKPOINT, load_preprocessor
from train import get_device
from train_answer_predictor import (
    CHECKPOINT_PATH,
    batch_nearest_answer_texts,
    load_checkpoint,
    predict_answer_embeddings_batch,
)


@dataclass
class AgentResult:
    """Per-agent data returned by the full pipeline."""

    uid: int
    description: str
    description_vector: np.ndarray  # (384,) raw embedding
    personality_vector: np.ndarray  # (384,) after preprocessor
    response_to_media: str
    like_value: float  # VADER compound in [-1, 1]
    similarity_score: float  # cosine similarity to own cluster centroid [−1, 1]
    action: Action  # one of 7 actions (dislike, nothing, like, share, etc.)

# Paths (same as rest of project)
HYBRID_DATASET_PATH = Path("hybrid_review_dataset.json")
NUM_AGENTS = 100

# Default media shown to the initial 10% (reel/post content)
DEFAULT_MEDIA = (
    "More than transportation—an exercise in proportion, motion, and restraint. Every line exists for a reason."
    )


def get_100_agent_descriptions(
    dataset_path: Path = HYBRID_DATASET_PATH,
    seed: int | None = None,
) -> list[str]:
    """
    Load exactly 100 agent descriptions, chosen at random from the dataset (new draw each run).
    Uses hybrid_review_dataset responses; if fewer than 100 available, pads with
    archetype-based placeholders from personality_answers.
    seed: Random seed (None = different 100 agents every run; int = reproducible).
    """
    rng = random.Random(seed)
    descriptions: list[str] = []
    if dataset_path.exists():
        with dataset_path.open() as f:
            data = json.load(f)
        responses = data.get("responses", [])
        candidates = [r["description"].strip() for r in responses if "description" in r]
        if len(candidates) >= NUM_AGENTS:
            descriptions = rng.sample(candidates, NUM_AGENTS)
        else:
            descriptions = list(candidates)

    # Fallback: if we have fewer than 100, pad with simple placeholders
    if len(descriptions) < NUM_AGENTS:
        fallback_path = Path("personality_answers.json")
        if fallback_path.exists():
            with fallback_path.open() as f:
                fallback_data = json.load(f)
            arch_responses = fallback_data.get("responses", [])
            seen_arch = set()
            for r in arch_responses:
                if len(descriptions) >= NUM_AGENTS:
                    break
                arch = r.get("persona", {}).get("archetype", "unknown")
                if arch not in seen_arch:
                    seen_arch.add(arch)
                    qs = r.get("questions", [])
                    ans = r.get("answers", [])
                    desc = f"Archetype: {arch}."
                    if qs and ans:
                        desc += f" {qs[0]} {ans[0]}"
                    descriptions.append(desc)
        while len(descriptions) < NUM_AGENTS:
            descriptions.append(f"Agent personality placeholder {len(descriptions)}.")

    return descriptions[:NUM_AGENTS]


def get_100_vectors(
    descriptions: list[str],
    preprocessor_path: Path | str = PREPROCESSOR_CHECKPOINT,
    device: torch.device | None = None,
    engine: EmbeddingEngine | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute description vectors (raw embed) and personality vectors (after preprocessor).
    Returns (description_vectors, personality_vectors) each (100, 384) float64.
    If engine is provided, reuse it (avoids loading the embedding model twice).
    """
    device = device or get_device()
    if engine is None:
        engine = EmbeddingEngine()
    preprocessor = load_preprocessor(preprocessor_path, device=device)
    preprocessor.eval()

    desc_embeddings = torch.tensor(
        engine.encode(descriptions),
        dtype=torch.float32,
        device=device,
    )
    description_vectors = desc_embeddings.cpu().numpy().astype(np.float64)
    with torch.no_grad():
        persona_vectors = preprocessor(desc_embeddings)
    personality_vectors = persona_vectors.cpu().numpy().astype(np.float64)
    return description_vectors, personality_vectors


def compute_all_responses_batch(
    personality_vectors: np.ndarray,
    media: str,
    engine: EmbeddingEngine,
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    device: torch.device | None = None,
    answer_temperature: float | None = 0.5,
    seed: int | None = None,
) -> tuple[list[str], list[float], list[str]]:
    """
    Compute response text and like value for all agents in one batch.
    answer_temperature: if not None, sample from top answers by similarity (adds variety; avoids always dislike).
    Returns (response_texts, like_values, sentiment_labels) each of length N.
    """
    device = device or get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    question = f"What is your opinion on this content: {media}"
    question_embedding = torch.tensor(
        engine.encode([question]),
        dtype=torch.float32,
        device=device,
    )
    persona_tensor = torch.tensor(
        personality_vectors,
        dtype=torch.float32,
        device=device,
    )
    predicted = predict_answer_embeddings_batch(
        persona_tensor,
        question_embedding,
        model,
    )
    rng = random.Random(seed)
    results = batch_nearest_answer_texts(
        predicted,
        checkpoint,
        temperature=answer_temperature,
        top_k=25,
        rng=rng,
    )
    response_texts = [r[0] for r in results]
    like_values = [get_like_value_from_reaction(t) for t in response_texts]
    sentiment_labels = [r[1] for r in results]
    return response_texts, like_values, sentiment_labels


def main(
    media: str = DEFAULT_MEDIA,
    seed: int = 42,
    dataset_path: Path | None = None,
) -> tuple[list[AgentResult], list[Any], list[tuple[Any, str, float, Any]], list[tuple[Any, list]]]:
    """
    Run the full pipeline and return all agents with full per-agent data.

    Returns:
        agents: List of AgentResult (one per agent), each with uid, description,
                description_vector, personality_vector, response_to_media, like_value.
        rep_uids: UIDs of the 10% representatives who saw the media in the network run.
        reactions: List of (uid, reaction_text, like_value, action) for representatives.
        shares: List of (sharer_uid, [recipient_uids]) from the network.
    """
    dataset_path = dataset_path or HYBRID_DATASET_PATH
    device = get_device()

    # 1. Checkpoints
    if not Path(PREPROCESSOR_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_CHECKPOINT}. "
            "Build it (e.g. run preprocessor.py or the training pipeline) first."
        )
    if not Path(CHECKPOINT_PATH).exists():
        raise FileNotFoundError(
            f"Answer predictor not found: {CHECKPOINT_PATH}. "
            "Run train_answer_predictor.py first."
        )

    # 2. 100 agents: random draw each run (seed=None); pipeline seed still controls network/sharing
    descriptions = get_100_agent_descriptions(dataset_path)
    assert len(descriptions) == NUM_AGENTS, f"Expected {NUM_AGENTS} descriptions, got {len(descriptions)}"
    engine = EmbeddingEngine()
    description_vectors, personality_vectors = get_100_vectors(
        descriptions, device=device, engine=engine
    )
    uids = list(range(NUM_AGENTS))

    # 3. Batch compute all 100 responses (temperature sampling so we don't always get "dislike")
    response_texts, like_values, sentiment_labels = compute_all_responses_batch(
        personality_vectors,
        media,
        engine,
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        answer_temperature=0,
        seed=seed,
    )

    # 3b. Per-agent similarity score: cosine similarity to own cluster centroid
    labels, centroids, _ = kmeans_auto_k(personality_vectors)
    similarity_scores = np.zeros(NUM_AGENTS, dtype=np.float64)
    for i in range(NUM_AGENTS):
        vec = personality_vectors[i]
        cent = centroids[labels[i]]
        nv, nc = np.linalg.norm(vec), np.linalg.norm(cent)
        if nv > 0 and nc > 0:
            similarity_scores[i] = float(np.dot(vec, cent) / (nv * nc))
        else:
            similarity_scores[i] = 0.0

    agents = [
        AgentResult(
            uid=uid,
            description=descriptions[uid],
            description_vector=description_vectors[uid].copy(),
            personality_vector=personality_vectors[uid].copy(),
            response_to_media=response_texts[uid],
            like_value=like_values[uid],
            similarity_score=float(similarity_scores[uid]),
            action=like_value_to_action(like_values[uid]),
        )
        for uid in uids
    ]

    # 4. Reaction function for pipeline: lookup precomputed responses
    def reaction_fn(uid: int, _media: Any) -> str:
        if 0 <= uid < len(response_texts):
            return response_texts[uid]
        return "I have no strong opinion."

    # 5. Run media pipeline: 10% representatives, reactions, shares (for network simulation)
    rep_uids, reactions, shares = run_media_pipeline(
        uids,
        personality_vectors,
        media=media,
        reaction_fn=reaction_fn,
        fraction=0.10,
        seed=seed,
    )

    # 6. Print summary
    sentiment_counts = Counter(sentiment_labels)
    print("=" * 60)
    print("FULL PIPELINE: 100 agents → network")
    print("=" * 60)
    print(f"Agents: {NUM_AGENTS} (UIDs 0..{NUM_AGENTS - 1})")
    print(f"Media: {media[:80]}...")
    print(f"Response sentiment distribution: {dict(sentiment_counts)}")
    action_counts = Counter(a.action for a in agents)
    print(f"Action distribution: {dict((k.name, v) for k, v in action_counts.items())}")
    print(f"Representatives (10%): {len(rep_uids)}")
    print(f"Sample representative UIDs: {rep_uids[:10]}")
    print()
    print("Sample agent results (uid, action, similarity, like_value, description snippet, response snippet):")
    for a in agents[:4]:
        desc_snip = a.description[:50] + "..." if len(a.description) > 50 else a.description
        resp_snip = a.response_to_media[:50] + "..." if len(a.response_to_media) > 50 else a.response_to_media
        print(f"  UID {a.uid}: action={a.action.name} similarity={a.similarity_score:.3f} like={a.like_value:.3f}")
        print(f"    desc: {desc_snip}")
        print(f"    resp: {resp_snip}")
    print()
    print("Sample reactions (representatives):")
    for r in reactions[:4]:
        uid, text, like_val, action = r
        print(f"  UID {uid}: like={like_val:.3f} -> {action.name}")
    print()
    print(f"Shares (sharer -> recipients): {len(shares)}")
    for sh in shares[:8]:
        print(f"  {sh[0]} -> {sh[1]}")
    print("=" * 60)

    return agents, rep_uids, reactions, shares


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run full pipeline: 100 agents → network (10%% exposure, reactions, sharing)")
    p.add_argument("--media", type=str, default=DEFAULT_MEDIA, help="Media/reel content to show")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dataset", type=Path, default=HYBRID_DATASET_PATH, help="Path to hybrid_review_dataset.json")
    args = p.parse_args()
    agents, rep_uids, reactions, shares = main(media=args.media, seed=args.seed, dataset_path=args.dataset)
    for a in agents:
        print(a.uid, a.action.name, a.similarity_score, a.description[:80], a.response_to_media[:80], a.like_value)
    # print(f"\nReturned: {len(agents)} agents (each with description, description_vector, personality_vector, response_to_media, like_value)")
