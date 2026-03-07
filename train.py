"""Train CompressionModel using data from personality_answers.json and niche_questions.json."""

import json
import random

import numpy as np
import torch
from response_modify import vectorize_pair
from compression_model import (
    CompressionModel,
    SimilarityConsistencyLoss,
    reconstruction_loss,
)

PERSONALITY_PATH = "personality_answers.json"
NICHE_PATH = "niche_questions.json"
SAVE_PATH = "compression_model.pt"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_data():
    with open(PERSONALITY_PATH) as f:
        personality = json.load(f)
    with open(NICHE_PATH) as f:
        niche = json.load(f)
    return personality, niche


def build_user_profile(personality_entry: dict, niche_pool: list, rng: random.Random) -> tuple[list[str], list[str]]:
    """
    Build a 10-question / 10-answer profile for one user.
    Q1-5: global personality questions with the user's answers.
    Q6-10: randomly sampled niche questions with their answers.
    """
    global_questions = [
        "What are your biggest motivations?",
        "What are your biggest weaknesses? Strengths?",
        "What activities do you do to handle stress or recharge?",
        "Do you think or act first?",
        "Do you work better alone or with a group?",
    ]
    ans = personality_entry["answers"]
    global_answers = [
        ans["q1_motivation"],
        ans["q2_weakness_strength"],
        ans["q3_stress_recharge"],
        ans["q4_think_or_act"],
        ans["q5_alone_or_group"],
    ]

    niche_sample = rng.sample(niche_pool, 5)
    niche_questions = [qa["question"] for qa in niche_sample]
    niche_answers = [qa["answer"] for qa in niche_sample]

    questions = global_questions + niche_questions
    answers = global_answers + niche_answers
    return questions, answers


def build_pairs(personality: dict, niche: dict, n_pairs: int, seed: int = 42) -> list:
    """
    Build (questions_1, answers_1, questions_2, answers_2) pairs from the JSON data.
    Each pair is two different users; Q1-5 are shared, Q6-10 differ per user.
    """
    rng = random.Random(seed)
    responses = personality["responses"]
    niche_pool = niche["qa_pairs"]
    pairs = []

    for _ in range(n_pairs):
        u1, u2 = rng.sample(responses, 2)
        q1, a1 = build_user_profile(u1, niche_pool, rng)
        q2, a2 = build_user_profile(u2, niche_pool, rng)
        pairs.append((q1, a1, q2, a2))

    return pairs


def collate_batch(pairs: list, device: torch.device) -> tuple:
    """Vectorize a batch of pairs and return (Q1, A1, Q2, A2) tensors."""
    Q1_list, A1_list, Q2_list, A2_list = [], [], [], []

    for q1, a1, q2, a2 in pairs:
        Q1, A1, Q2, A2 = vectorize_pair(q1, a1, q2, a2)
        Q1_list.append(Q1)
        A1_list.append(A1)
        Q2_list.append(Q2)
        A2_list.append(A2)

    Q1 = torch.tensor(np.stack(Q1_list), dtype=torch.float32, device=device)
    A1 = torch.tensor(np.stack(A1_list), dtype=torch.float32, device=device)
    Q2 = torch.tensor(np.stack(Q2_list), dtype=torch.float32, device=device)
    A2 = torch.tensor(np.stack(A2_list), dtype=torch.float32, device=device)

    return Q1, A1, Q2, A2


def save_checkpoint(model, loss_fn, epoch, loss, path=SAVE_PATH):
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "loss_fn_state_dict": loss_fn.state_dict(),
    }, path)


def load_checkpoint(path=SAVE_PATH, device=None):
    device = device or get_device()
    model = CompressionModel(n=384).to(device)
    loss_fn = SimilarityConsistencyLoss().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    loss_fn.load_state_dict(ckpt["loss_fn_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return model, loss_fn


def train(
    n_epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    n_pairs: int = 64,
    device: str = None,
    recon_weight: float = 0.5,
):
    device = device or get_device()
    if isinstance(device, str):
        device = torch.device(device)
    print(f"Using device: {device}")

    personality, niche = load_data()
    print(f"Loaded {len(personality['responses'])} personality profiles, {niche['total']} niche Q/A pairs")

    model = CompressionModel(n=384).to(device)
    loss_fn = SimilarityConsistencyLoss().to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    pairs = build_pairs(personality, niche, n_pairs=n_pairs)
    n_batches = (len(pairs) + batch_size - 1) // batch_size

    best_loss = float("inf")

    for epoch in range(n_epochs):
        total_loss = 0.0
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            Q1, A1, Q2, A2 = collate_batch(batch, device)

            opt.zero_grad()
            v1 = model(Q1, A1)
            v2 = model(Q2, A2)
            loss_sim = loss_fn(A1, A2, v1, v2)
            loss_recon = reconstruction_loss(model, Q1, A1)
            loss = loss_sim + recon_weight * loss_recon
            loss.backward()
            opt.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / n_batches
        w = torch.softmax(loss_fn.logits, dim=0)
        print(f"Epoch {epoch + 1}/{n_epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.6f}  weights=[{w[0]:.3f}, {w[1]:.3f}]")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, loss_fn, epoch + 1, avg_loss)

    print(f"\nBest loss: {best_loss:.4f} — saved to {SAVE_PATH}")
    return model, loss_fn


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_pairs", type=int, default=32)
    p.add_argument("--quick", action="store_true", help="Quick test: 2 epochs, 8 pairs")
    args = p.parse_args()
    if args.quick:
        train(n_epochs=2, batch_size=8, n_pairs=8)
    else:
        train(n_epochs=args.n_epochs, batch_size=args.batch_size, n_pairs=args.n_pairs)
