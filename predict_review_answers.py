"""Run question-by-question answer prediction for a hybrid review profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessor import PREPROCESSOR_CHECKPOINT
from train import get_device
from train_answer_predictor import (
    CHECKPOINT_PATH,
    load_checkpoint,
    nearest_answer_text,
    predict_answer_embedding,
)

DATASET_PATH = Path("hybrid_review_dataset.json")


def load_profile(dataset_path: str | Path, profile_id: int) -> dict:
    with Path(dataset_path).open() as f:
        dataset = json.load(f)
    responses = dataset["responses"]
    if profile_id < 0 or profile_id >= len(responses):
        raise IndexError(f"profile_id must be in [0, {len(responses) - 1}]")
    return responses[profile_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict answers one question at a time")
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--profile-id", type=int, default=0)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--preprocessor", default=str(PREPROCESSOR_CHECKPOINT))
    args = parser.parse_args()

    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    profile = load_profile(args.dataset, args.profile_id)
    description = profile["description"]

    print(f"profile_id={profile['id']} hybrid_name={profile['hybrid_name']}")
    print(f"description={description}\n")

    for idx, qa_pair in enumerate(profile["qa_pairs"][: args.max_questions], start=1):
        predicted_embedding = predict_answer_embedding(
            description=description,
            question=qa_pair["question"],
            model=model,
            preprocessor_path=args.preprocessor,
            device=device,
        )
        predicted_text, predicted_label, score = nearest_answer_text(predicted_embedding, checkpoint)
        print(f"[{idx}] question:")
        print(qa_pair["question"])
        print(f"predicted_label={predicted_label} similarity={score:.4f}")
        print(f"predicted_answer={predicted_text}")
        print(f"gold_label={qa_pair['sentiment_label']}")
        print(f"gold_answer={qa_pair['answer']}\n")


if __name__ == "__main__":
    main()
