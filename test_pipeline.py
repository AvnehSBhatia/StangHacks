#!/usr/bin/env python3
"""Full pipeline test: text description + question -> predicted embedding -> sentiment."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
# Use default backend so plt.show() displays; Agg would save-only with no window
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from answer_bank import ANSWER_BANK_PATH, load_answer_bank
from embedding_engine import EmbeddingEngine
from preprocessor import PREPROCESSOR_CHECKPOINT, load_preprocessor
from train import get_device
from train_answer_predictor import (
    CHECKPOINT_PATH,
    load_checkpoint,
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

# One sentence per sentiment; voting = argmax cosine similarity to predicted embedding
SENTIMENT_PHRASES = [
    "I strongly agree and love this. It resonates deeply with me.",
    "I agree with this. It works well for me.",
    "I feel neutral. I can see both sides.",
    "I disagree with this. It doesn't work for me.",
    "I strongly disagree and dislike this. It feels wrong.",
]
SENTIMENT_ORDER = ["strong_like", "like", "neutral", "dislike", "strong_dislike"]

# When |max(like+strong_like) - max(dislike+strong_dislike)| < this, pick neutral
# Lower = fewer neutrals (only when like/dislike are very close)
NEUTRAL_TIE_THRESHOLD = 0.04

POSITIVE_SENTIMENTS = {"strong_like", "like"}
NEGATIVE_SENTIMENTS = {"dislike", "strong_dislike"}

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


def sample_top_result(
    top_results: list[tuple[str, str, float]],
) -> tuple[tuple[str, str, float], Counter]:
    """Sample one result from top-k, weighted by similarity."""
    if not top_results:
        return ("No matching answers found.", "neutral", 0.0), Counter()

    counts = Counter(label for _, label, _ in top_results)
    min_score = min(score for _, _, score in top_results)
    weights = [max(score - min_score, 1e-6) for _, _, score in top_results]
    sampled = random.choices(top_results, weights=weights, k=1)[0]
    return sampled, counts


def sample_top_result_with_neutral(
    top_results: list[tuple[str, str, float]],
    threshold: float = NEUTRAL_TIE_THRESHOLD,
) -> tuple[tuple[str, str, float], Counter]:
    """
    Return neutral when positive and negative top-k mass are close.
    Otherwise randomly sample one result from top-k weighted by similarity.
    """
    if not top_results:
        return ("No matching answers found.", "neutral", 0.0), Counter()

    counts = Counter(label for _, label, _ in top_results)
    pos_mass = sum(score for _, label, score in top_results if label in POSITIVE_SENTIMENTS)
    neg_mass = sum(score for _, label, score in top_results if label in NEGATIVE_SENTIMENTS)
    if abs(pos_mass - neg_mass) < threshold:
        return ("I feel neutral. I can see both sides.", "neutral", (pos_mass + neg_mass) / 2), counts

    sampled, _ = sample_top_result(top_results)
    return sampled, counts


def load_profile(dataset_path: Path, profile_id: int) -> dict:
    with dataset_path.open() as f:
        dataset = json.load(f)
    responses = dataset["responses"]
    if profile_id < 0 or profile_id >= len(responses):
        raise IndexError(f"profile_id must be in [0, {len(responses) - 1}]")
    return responses[profile_id]


def _filter_bank_no_neutral(bank: dict) -> dict:
    """Return bank with neutral answers excluded."""
    keep = [i for i, s in enumerate(bank["sentiment_labels"]) if s != "neutral"]
    if len(keep) == len(bank["sentiment_labels"]):
        return bank
    emb = bank["answer_embeddings"]
    return {
        "answer_embeddings": emb[keep] if isinstance(emb, torch.Tensor) else torch.tensor(emb)[keep],
        "answer_texts": [bank["answer_texts"][i] for i in keep],
        "sentiment_labels": [bank["sentiment_labels"][i] for i in keep],
    }


def resolve_sentiment(
    predicted_embedding: torch.Tensor,
    bank: dict,
    top_k: int = 5,
    threshold: float = NEUTRAL_TIE_THRESHOLD,
) -> tuple[str, str, float]:
    """Resolve sentiment from top-k nearest answers with neutral fallback."""
    top_results = nearest_answers_topk(predicted_embedding, bank, top_k=top_k)
    sampled, _ = sample_top_result_with_neutral(top_results, threshold=threshold)
    return sampled


def run_benchmark(
    n: int,
    dataset_path: Path,
    checkpoint_path: Path,
    preprocessor_path: Path,
    batch_size: int = 128,
    top_k: int = 5,
    use_phrase_sentiment: bool = False,
    use_large_bank: bool = True,
    bank_path: Path | None = None,
    neutral_threshold: float = NEUTRAL_TIE_THRESHOLD,
) -> None:
    """Load all models into RAM, run n inferences (encode + preprocessor + model) in multibatch, report speed."""
    device = get_device()
    print(f"Loading all models into RAM (device={device})...")

    # CuDNN benchmark finds faster algorithms (CUDA only)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    engine = EmbeddingEngine()
    engine.model.to(device)
    # Keep float32 for encoder so sentiment matches individual run (half precision caused all-neutral)
    engine.model.float()
    preprocessor = load_preprocessor(preprocessor_path, device=device)
    preprocessor.eval()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    # Use large answer bank for sentiment lookup (or checkpoint's small bank if --no-large-bank)
    if use_large_bank:
        bank = load_answer_bank(bank_path or ANSWER_BANK_PATH, device=device)
        print(f"Using large answer bank ({len(bank['answer_texts'])} answers)")
    else:
        bank = {
            "answer_embeddings": checkpoint["answer_embeddings"].to(device),
            "answer_texts": checkpoint["answer_texts"],
            "sentiment_labels": checkpoint["sentiment_labels"],
        }

    # Compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        preprocessor = torch.compile(preprocessor, mode="reduce-overhead")
        model = torch.compile(model, mode="reduce-overhead")

    with dataset_path.open() as f:
        data = json.load(f)
    responses = data["responses"]

    # Random descriptions, same question for all n runs
    question = ESSAY_QUESTION
    descriptions = [random.choice(responses)["description"] for _ in range(n)]

    encode_batch = min(batch_size, n)

    print(f"Running {n} full pipelines (random descriptions, same question), batch={encode_batch}...")
    start = time.perf_counter()
    with torch.inference_mode():
        desc_embs = engine.model.encode(
            descriptions,
            batch_size=encode_batch,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        desc_embs = desc_embs.float().to(device)
        # Encode question once, expand to n (same question for all runs)
        q_emb_single = engine.model.encode(
            [question],
            batch_size=1,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        q_embs = q_emb_single.float().to(device).repeat(n, 1)
        personas = preprocessor(desc_embs)
        preds = model(personas, q_embs)
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_norm = F.normalize(preds, dim=-1)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (encoding + inference)")
    print("=" * 60)
    print(f"  Runs: {n}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per run: {elapsed / n * 1000:.2f} ms")
    print(f"  Throughput: {n / elapsed:.1f} runs/s")

    # Sentiment: answer-bank top-k vote (matches single-run) or phrase-based
    if use_phrase_sentiment:
        phrase_embs = engine.model.encode(
            SENTIMENT_PHRASES,
            batch_size=5,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        phrase_norm = F.normalize(phrase_embs.float().to(device), dim=-1)
        phrase_sims = pred_norm @ phrase_norm.T  # (n, 5)
        winning_idx = phrase_sims.argmax(dim=-1)  # (n,)
        run_sentiments = [SENTIMENT_ORDER[i] for i in winning_idx.cpu().tolist()]
        method_label = "cosine sim to 5 phrases"
    else:
        bank_f = bank
        if not bank_f["answer_texts"]:
            run_sentiments = ["neutral"] * n
        else:
            bank_emb = F.normalize(bank_f["answer_embeddings"].to(device), dim=-1)
            sims = pred_norm @ bank_emb.T  # (n, bank_size)
            k = min(top_k, sims.size(1))
            topk = sims.topk(k, dim=1)
            run_sentiments = []
            for row_indices, row_scores in zip(topk.indices.tolist(), topk.values.tolist()):
                top_results = [
                    (
                        bank_f["answer_texts"][idx],
                        bank_f["sentiment_labels"][idx],
                        float(score),
                    )
                    for idx, score in zip(row_indices, row_scores)
                ]
                (_, sampled_label, _), _ = sample_top_result_with_neutral(
                    top_results,
                    threshold=neutral_threshold,
                )
                run_sentiments.append(sampled_label)
        method_label = f"random weighted pick from top-{top_k} with neutral fallback ({len(bank_f['answer_texts'])} answers)"

    counts = Counter(run_sentiments)
    order = ["strong_like", "like", "neutral", "dislike", "strong_dislike"]
    labels_display = [SENTIMENT_DISPLAY.get(l, l) for l in order]
    values = [counts.get(l, 0) for l in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c", "#c0392b"]
    bars = ax.bar(labels_display, values, color=colors)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title(f"Sentiment across {n} runs ({method_label})")
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    out_path = Path("benchmark_sentiments.png").resolve()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved sentiment graph to {out_path}")
    # Open in Cursor/VS Code (shows image in editor) or system viewer
    try:
        for cmd in [["cursor", str(out_path)], ["code", str(out_path)]]:
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                break
        else:
            if sys.platform == "darwin":
                subprocess.run(["open", str(out_path)], check=False)
            elif sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", str(out_path)], check=False)
            elif sys.platform == "win32":
                subprocess.run(["start", "", str(out_path)], check=False, shell=True)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: description + question -> sentiment")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--profile-id", type=int, default=0, help="Profile to use for description")
    parser.add_argument("--description", type=str, default=None, help="Override with custom description")
    parser.add_argument("--question", type=str, default=None, help="Override question (default: essay)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--preprocessor", type=Path, default=PREPROCESSOR_CHECKPOINT)
    parser.add_argument("--top-k", type=int, default=5, help="Show top-k for sentiment distribution")
    parser.add_argument("--benchmark", action="store_true", help="Run 100 batched inferences, models loaded once")
    parser.add_argument("--n", type=int, default=100, help="Number of runs for benchmark")
    parser.add_argument("--bank", type=Path, default=None, help="Path to answer bank (default: answer_bank.pt)")
    parser.add_argument("--no-large-bank", action="store_true", help="Use checkpoint's small bank instead of large bank")
    parser.add_argument("--neutral-threshold", type=float, default=NEUTRAL_TIE_THRESHOLD, help="Pick neutral when |like-dislike| < this (default: 0.04)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            n=args.n,
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            preprocessor_path=args.preprocessor,
            top_k=args.top_k,
            use_large_bank=not args.no_large_bank,
            bank_path=args.bank,
            neutral_threshold=args.neutral_threshold,
        )
        return

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    # Load answer bank for nearest-neighbor lookup (large by default)
    if args.no_large_bank:
        bank = checkpoint
    else:
        bank = load_answer_bank(args.bank or ANSWER_BANK_PATH, device=device)
        print(f"Using large answer bank ({len(bank['answer_texts'])} answers)")

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

    top_results = nearest_answers_topk(predicted_embedding, bank, top_k=args.top_k)
    (answer_text, sentiment_label, score), counts = sample_top_result_with_neutral(
        top_results,
        threshold=args.neutral_threshold,
    )
    sentiment_display = SENTIMENT_DISPLAY.get(sentiment_label, sentiment_label)

    print("PREDICTED ANSWER (nearest in bank)")
    print("=" * 60)
    print(answer_text)
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)
    print(f"Sentiment: {sentiment_display}")
    print(f"Similarity to nearest: {score:.4f}")

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
