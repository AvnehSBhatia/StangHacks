#!/usr/bin/env python3
"""Full pipeline test: text description + question -> predicted embedding -> sentiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from preprocessor import PREPROCESSOR_CHECKPOINT
from train import get_device
from train_answer_predictor import (
    CHECKPOINT_PATH,
    load_checkpoint,
    nearest_answer_text,
    predict_answer_embedding,
)

DATASET_PATH = Path("hybrid_review_dataset.json")

SENTIMENT_DISPLAY = {
    "strong_like": "Strong like",
    "like": "Like",
    "neutral": "Neutral",
    "dislike": "Dislike",
    "strong_dislike": "Strong dislike",
}

ESSAY_QUESTION = '''What is your opinion on this essay: The best aerospace moment I've had wasn't in a lab. Rather, it was at a middle school career fair, watching a kid hold a foamboard airplane over his head. I had just explained to him, as I had for hundreds of other students that day, how the Coandă effect and Newton's Third Law turn airflow over a wing into lift. He tilted the nose up, and sprinted down the MPR. His eyes went wide and his face had an expression of shock as he felt the plane begin to generate lift. Ten minutes earlier, lift was a foreign concept. Now, it was real, manifested in the very palms of his hands.
That moment, when the theories became tangible, is why Cluster 5 is my first choice. Mechanical and Aerospace Engineering is where physics gets huge. A principle you can note down on a napkin is the same principle that lifts a paper plane and hurls rockets past the atmosphere. From exploring everything from self-balancing robots that utilize PID control to analyzing a rocket's thrust profile in a wind tunnel, this cluster is where I can visualize that thrilling scale. This is the cluster where I can experience turning fundamental laws into systems you can feel.'''


def nearest_answers_topk(
    predicted_embedding: torch.Tensor,
    checkpoint: dict,
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return top-k nearest (answer_text, sentiment_label, score)."""
    bank = checkpoint["answer_embeddings"].to(predicted_embedding.device)
    sims = F.cosine_similarity(
        F.normalize(predicted_embedding.unsqueeze(0), dim=-1),
        F.normalize(bank, dim=-1),
        dim=-1,
    )
    k = min(top_k, sims.size(0))
    topk = sims.topk(k)
    return [
        (
            checkpoint["answer_texts"][idx],
            checkpoint["sentiment_labels"][idx],
            float(score),
        )
        for idx, score in zip(topk.indices.tolist(), topk.values.tolist())
    ]


def load_profile(dataset_path: Path, profile_id: int) -> dict:
    with dataset_path.open() as f:
        dataset = json.load(f)
    responses = dataset["responses"]
    if profile_id < 0 or profile_id >= len(responses):
        raise IndexError(f"profile_id must be in [0, {len(responses) - 1}]")
    return responses[profile_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: description + question -> sentiment")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--profile-id", type=int, default=0, help="Profile to use for description")
    parser.add_argument("--description", type=str, default=None, help="Override with custom description")
    parser.add_argument("--question", type=str, default=None, help="Override question (default: essay)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=5, help="Show top-k for sentiment distribution")
    args = parser.parse_args()

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    if args.description is not None:
        description = args.description
        print("Using custom description")
    else:
        profile = load_profile(args.dataset, args.profile_id)
        description = profile["description"]
        print(f"Profile {args.profile_id}: {profile['hybrid_name']}")

    question = args.question if args.question is not None else ESSAY_QUESTION

    print("\n" + "=" * 60)
    print("DESCRIPTION")
    print("=" * 60)
    print(description[:500] + ("..." if len(description) > 500 else ""))
    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question[:400] + ("..." if len(question) > 400 else ""))
    print("\n" + "=" * 60)

    predicted_embedding = predict_answer_embedding(
        description=description,
        question=question,
        model=model,
        preprocessor_path=args.preprocessor,
        device=device,
    )

    answer_text, sentiment_label, score = nearest_answer_text(predicted_embedding, checkpoint)
    sentiment_display = SENTIMENT_DISPLAY.get(sentiment_label, sentiment_label)

    print("PREDICTED ANSWER (nearest in bank)")
    print("=" * 60)
    print(answer_text)
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)
    print(f"Sentiment: {sentiment_display}")
    print(f"Similarity to nearest: {score:.4f}")

    top_results = nearest_answers_topk(predicted_embedding, checkpoint, top_k=args.top_k)
    counts = Counter(r[1] for r in top_results)
    print(f"\nTop-{args.top_k} sentiment distribution:")
    for label in ["strong_like", "like", "neutral", "dislike", "strong_dislike"]:
        c = counts.get(label, 0)
        if c > 0:
            print(f"  {SENTIMENT_DISPLAY.get(label, label)}: {c}")

    print("\nTop matches:")
    for i, (ans, lbl, s) in enumerate(top_results, 1):
        print(f"  [{i}] {SENTIMENT_DISPLAY.get(lbl, lbl)} ({s:.3f}): {ans[:80]}...")


if __name__ == "__main__":
    main()
