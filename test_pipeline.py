#!/usr/bin/env python3
"""Full pipeline test: text description + question -> predicted embedding -> sentiment."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from embedding_engine import EmbeddingEngine
from preprocessor import PREPROCESSOR_CHECKPOINT, load_preprocessor
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


def run_benchmark(
    n: int,
    dataset_path: Path,
    checkpoint_path: Path,
    preprocessor_path: Path,
    batch_size: int = 128,
    top_k: int = 5,
) -> None:
    """Load all models into RAM, run n inferences (encode + preprocessor + model) in multibatch, report speed."""
    device = get_device()
    print(f"Loading all models into RAM (device={device})...")

    # CuDNN benchmark finds faster algorithms (CUDA only)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    engine = EmbeddingEngine()
    engine.model.to(device)
    # Half precision: ~2x faster encoder, less memory (encoder is the bottleneck)
    if device.type in ("cuda", "mps"):
        engine.model.half()
    preprocessor = load_preprocessor(preprocessor_path, device=device)
    preprocessor.eval()
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    answer_bank = checkpoint["answer_embeddings"].to(device)
    answer_bank_norm = F.normalize(answer_bank, dim=-1)

    # Compile for faster inference (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        preprocessor = torch.compile(preprocessor, mode="reduce-overhead")
        model = torch.compile(model, mode="reduce-overhead")

    with dataset_path.open() as f:
        data = json.load(f)
    responses = data["responses"]

    pairs: list[tuple[str, str]] = []
    for i in range(n):
        profile = responses[i % len(responses)]
        qa = profile["qa_pairs"][i % len(profile["qa_pairs"])]
        pairs.append((profile["description"], qa["question"]))

    descriptions = [p[0] for p in pairs]
    questions = [p[1] for p in pairs]

    # Use convert_to_tensor to keep on device, avoid numpy round-trip
    encode_batch = min(batch_size, n)

    print(f"Running {n} full pipelines (encode + preprocessor + model + nearest), batch={encode_batch}...")
    start = time.perf_counter()
    with torch.inference_mode():
        desc_embs = engine.model.encode(
            descriptions,
            batch_size=encode_batch,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        desc_embs = desc_embs.float().to(device)
        q_embs = engine.model.encode(
            questions,
            batch_size=encode_batch,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        q_embs = q_embs.float().to(device)
        personas = preprocessor(desc_embs)
        preds = model(personas, q_embs)
        if isinstance(preds, tuple):
            preds = preds[0]
        pred_norm = F.normalize(preds, dim=-1)
        sims = pred_norm @ answer_bank_norm.T
        best_indices = sims.argmax(dim=-1)
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (encoding + inference)")
    print("=" * 60)
    print(f"  Runs: {n}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per run: {elapsed / n * 1000:.2f} ms")
    print(f"  Throughput: {n / elapsed:.1f} runs/s")

    # Sentiment distribution from top-k answers per run (majority vote)
    k = min(top_k, sims.size(-1))
    topk_indices = sims.topk(k, dim=-1).indices  # (n, k)
    sentiment_labels = checkpoint["sentiment_labels"]

    run_sentiments: list[str] = []
    for i in range(n):
        labels = [sentiment_labels[idx] for idx in topk_indices[i].cpu().tolist()]
        vote = Counter(labels).most_common(1)[0][0]
        run_sentiments.append(vote)

    counts = Counter(run_sentiments)
    order = ["strong_like", "like", "neutral", "dislike", "strong_dislike"]
    labels_display = [SENTIMENT_DISPLAY.get(l, l) for l in order]
    values = [counts.get(l, 0) for l in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#3498db", "#95a5a6", "#e74c3c", "#c0392b"]
    bars = ax.bar(labels_display, values, color=colors)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_title(f"Sentiment distribution across {n} runs (top-{k} majority vote)")
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    out_path = Path("benchmark_sentiments.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved sentiment graph to {out_path}")
    plt.show()


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
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            n=args.n,
            dataset_path=args.dataset,
            checkpoint_path=args.checkpoint,
            preprocessor_path=args.preprocessor,
            top_k=args.top_k,
        )
        return

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
