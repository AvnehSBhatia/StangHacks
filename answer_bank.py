"""Build and load a large, expansive answer bank for sentiment lookup."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from embedding_engine import DEFAULT_MODEL_NAME, EmbeddingEngine

# Phrase-based sentiment inference for answers without gold labels
SENTIMENT_PHRASES = [
    "I strongly agree and love this. It resonates deeply with me.",
    "I agree with this. It works well for me.",
    "I feel neutral. I can see both sides.",
    "I disagree with this. It doesn't work for me.",
    "I strongly disagree and dislike this. It feels wrong.",
]
SENTIMENT_ORDER = ["strong_like", "like", "neutral", "dislike", "strong_dislike"]

ANSWER_BANK_PATH = Path("answer_bank.pt")
HYBRID_REVIEW_PATH = Path("hybrid_review_dataset.json")
PERSONALITY_ANSWERS_PATH = Path("personality_answers.json")


def _infer_sentiment(embedding: np.ndarray, phrase_embeddings: np.ndarray) -> str:
    """Infer sentiment by argmax cosine similarity to phrase embeddings."""
    emb = embedding.reshape(1, -1).astype(np.float32)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    sims = np.dot(emb_norm, phrase_embeddings.T)
    idx = int(np.argmax(sims[0]))
    return SENTIMENT_ORDER[idx]


def _load_hybrid_review(path: Path) -> list[tuple[str, str]]:
    """Load (answer, sentiment) pairs from hybrid_review_dataset."""
    with path.open() as f:
        data = json.load(f)
    pairs: list[tuple[str, str]] = []
    for response in data.get("responses", []):
        for qa in response.get("qa_pairs", []):
            pairs.append((qa["answer"], qa["sentiment_label"]))
    return pairs


def _load_personality_answers(path: Path) -> list[str]:
    """Load all unique answers from personality_answers (no sentiment)."""
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    answers: list[str] = []
    seen: set[str] = set()
    for response in data.get("responses", []):
        for ans in response.get("answers", []):
            if ans and ans not in seen:
                seen.add(ans)
                answers.append(ans)
    return answers


def build_answer_bank(
    hybrid_path: Path = HYBRID_REVIEW_PATH,
    personality_path: Path = PERSONALITY_ANSWERS_PATH,
    output_path: Path = ANSWER_BANK_PATH,
    batch_size: int = 256,
    device: str | None = None,
) -> dict:
    """
    Build a large answer bank from hybrid_review (gold sentiment) + personality_answers (inferred).
    Dedupes by text; hybrid_review wins on overlap. Saves embeddings to output_path.
    """
    engine = EmbeddingEngine()

    # 1. Load hybrid_review (gold labels) - these take precedence
    answer_to_sentiment: dict[str, str] = {}
    if hybrid_path.exists():
        for ans, sent in _load_hybrid_review(hybrid_path):
            answer_to_sentiment[ans] = sent
        print(f"Loaded {len(answer_to_sentiment)} answers from hybrid_review (gold sentiment)")
    else:
        print(f"Warning: {hybrid_path} not found, skipping")

    # 2. Load personality_answers - add new answers, infer sentiment for those without gold
    personality_answers = _load_personality_answers(personality_path)
    added = 0
    for ans in personality_answers:
        if ans not in answer_to_sentiment:
            answer_to_sentiment[ans] = ""  # placeholder, infer later
            added += 1
    print(f"Added {added} unique answers from personality_answers")

    # 3. Encode phrase embeddings for inference
    phrase_embs = engine.encode(SENTIMENT_PHRASES, batch_size=5)
    phrase_norm = phrase_embs / (np.linalg.norm(phrase_embs, axis=1, keepdims=True) + 1e-8)

    # 4. Encode all answers once
    answer_texts = list(answer_to_sentiment.keys())
    print(f"Encoding {len(answer_texts)} answers...")
    answer_embeddings = engine.encode(answer_texts, batch_size=batch_size)

    # 5. Infer sentiment for answers that need it (personality_answers)
    need_inference = [i for i, a in enumerate(answer_texts) if not answer_to_sentiment[a]]
    if need_inference:
        print(f"Inferring sentiment for {len(need_inference)} answers via phrase similarity...")
        for i in need_inference:
            answer_to_sentiment[answer_texts[i]] = _infer_sentiment(
                answer_embeddings[i], phrase_norm
            )

    sentiment_labels = [answer_to_sentiment[a] for a in answer_texts]

    # 5b. Exclude neutral (pick neutral only when like vs dislike are too close at inference)
    keep_idx = [i for i, s in enumerate(sentiment_labels) if s != "neutral"]
    answer_texts = [answer_texts[i] for i in keep_idx]
    sentiment_labels = [sentiment_labels[i] for i in keep_idx]
    answer_embeddings = answer_embeddings[keep_idx]

    # 6. Build bank
    answer_embeddings_t = torch.tensor(answer_embeddings, dtype=torch.float32)

    bank = {
        "answer_texts": answer_texts,
        "sentiment_labels": sentiment_labels,
        "answer_embeddings": answer_embeddings_t,
        "embedding_dim": int(answer_embeddings_t.size(-1)),
    }
    torch.save(bank, output_path)
    print(f"Saved answer bank ({len(answer_texts)} answers) -> {output_path}")
    return bank


def load_answer_bank(
    path: Path = ANSWER_BANK_PATH,
    device: torch.device | None = None,
    rebuild_if_missing: bool = True,
) -> dict:
    """
    Load the answer bank. If missing and rebuild_if_missing, builds it first.
    Returns dict with answer_embeddings, answer_texts, sentiment_labels, embedding_dim.
    """
    path = Path(path)
    if not path.exists():
        if rebuild_if_missing:
            print(f"Answer bank not found at {path}, building...")
            return build_answer_bank(output_path=path)
        raise FileNotFoundError(f"Answer bank not found: {path}")
    bank = torch.load(path, map_location="cpu", weights_only=False)
    if device is not None and "answer_embeddings" in bank:
        bank = dict(bank)
        bank["answer_embeddings"] = bank["answer_embeddings"].to(device)
    return bank


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the large answer bank")
    parser.add_argument("--output", type=Path, default=ANSWER_BANK_PATH)
    args = parser.parse_args()
    build_answer_bank(output_path=args.output)

