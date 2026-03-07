"""Interactive demo: answer questions, get a 64D persona vector."""

from __future__ import annotations

import torch

from embedding_engine import EmbeddingEngine
from generate_personas import QUESTIONS
from train import get_device, load_checkpoint

CHECKPOINT_PATH = "persona_encoder_checkpoint.pt"


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    try:
        model, _ = load_checkpoint(CHECKPOINT_PATH, device=device)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Checkpoint {CHECKPOINT_PATH} not found. Run `python3 train.py` first."
        ) from exc
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    model.eval()
    engine = EmbeddingEngine()

    print("\n--- Answer these questions to generate your persona vector ---\n")
    answers: list[str] = []
    for i, q in enumerate(QUESTIONS, 1):
        ans = input(f"{i}. {q}\n   > ").strip()
        if not ans:
            ans = "(no answer)"
        answers.append(ans)

    question_embeddings = engine.encode(QUESTIONS)
    answer_embeddings = engine.encode(answers)

    q_t = torch.tensor(question_embeddings, dtype=torch.float32, device=device).unsqueeze(0)
    a_t = torch.tensor(answer_embeddings, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        persona = model.encode_persona(q_t, a_t)
    vector = persona.squeeze(0).cpu().tolist()

    print("\n--- Results ---\n")
    print("Persona vector:")
    print(vector)


if __name__ == "__main__":
    main()
