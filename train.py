"""Train the minimal persona encoder."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from compression_model import (
    CompressionModel,
    cosine_embedding_loss,
    smoke_test_shapes,
)
from embedding_engine import EmbeddedDataset, EmbeddingEngine
from generate_personas import DEFAULT_OUTPUT_PATH, ensure_dataset_exists

SAVE_PATH = "persona_encoder_checkpoint.pt"

# Training config
FULL_TRAIN_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DATASET_PATH = str(DEFAULT_OUTPUT_PATH)
MIN_PROFILES = 1000  # 100 archetypes x 10 profiles each
USE_CACHE = True


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def shuffle_indices(n_items: int) -> list[int]:
    indices = list(range(n_items))
    random.shuffle(indices)
    return indices


def slice_batch(dataset: EmbeddedDataset, indices: list[int]) -> tuple[torch.Tensor, ...]:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=dataset.questions.device)
    questions = dataset.questions.index_select(0, index_tensor)
    answers = dataset.answers.index_select(0, index_tensor)
    archetype_embeddings = dataset.archetype_embeddings.index_select(0, index_tensor)
    return questions, answers, archetype_embeddings


def save_checkpoint(
    model: CompressionModel,
    epoch: int,
    loss: float,
    dataset_path: str | Path,
    path: str | Path = SAVE_PATH,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "dataset_path": str(dataset_path),
        "model_state_dict": model.state_dict(),
        "model_embedding_dim": model.n,
        "model_latent_dim": model.latent_dim,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path = SAVE_PATH,
    device: torch.device | None = None,
) -> tuple[CompressionModel, dict]:
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_embedding_dim = int(checkpoint.get("model_embedding_dim", 64))
    model_latent_dim = int(checkpoint.get("model_latent_dim", model_embedding_dim))
    model = CompressionModel(
        n=model_embedding_dim,
        latent_dim=model_latent_dim,
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} does not match the minimal persona encoder. "
            "Retrain with train.py to create a fresh persona-only checkpoint."
        ) from exc
    return model, checkpoint


def train_model(
    model: CompressionModel,
    dataset: EmbeddedDataset,
    dataset_path: str | Path,
    lr: float,
    epochs: int,
    batch_size: int,
) -> CompressionModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_loss = float("inf")

    for epoch in range(epochs):
        indices = shuffle_indices(dataset.size())
        total_loss = 0.0
        batch_count = 0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            questions, answers, archetype_embeddings = slice_batch(dataset, batch_indices)

            optimizer.zero_grad()
            persona_summary = model.encode_persona(questions, answers)
            loss = cosine_embedding_loss(persona_summary, archetype_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batch_count)
        msg = (
            f"[train] epoch={epoch + 1}/{epochs} "
            f"loss={avg_loss:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )
        print(msg)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                epoch=epoch + 1,
                loss=avg_loss,
                dataset_path=dataset_path,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    return model


def tiny_overfit_test(
    model: CompressionModel,
    dataset: EmbeddedDataset,
    steps: int = 20,
    lr: float = 1e-3,
) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    indices = list(range(min(8, dataset.size())))

    final_loss = 0.0
    for _ in range(steps):
        questions, answers, archetype_embeddings = slice_batch(dataset, indices)
        optimizer.zero_grad()
        persona_summary = model.encode_persona(questions, answers)
        loss = cosine_embedding_loss(persona_summary, archetype_embeddings)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    return final_loss


def train(
    full_train_epochs: int = FULL_TRAIN_EPOCHS,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    dataset_path: str | Path = DEFAULT_OUTPUT_PATH,
    min_profiles: int = MIN_PROFILES,
    use_cache: bool = True,
) -> CompressionModel:
    device = device or get_device()
    if isinstance(device, str):
        device = torch.device(device)
    print(f"Using device: {device}")

    dataset_path = ensure_dataset_exists(dataset_path, min_profiles=min_profiles)
    embedding_engine = EmbeddingEngine(dataset_path=dataset_path)
    dataset = embedding_engine.embed_training_examples(
        dataset_path=dataset_path,
        device=device,
        use_cache=use_cache,
    )
    print(f"Loaded embedded dataset with {dataset.size()} persona examples from {dataset_path}")

    embedding_dim = embedding_engine.embedding_dim
    latent_dim = embedding_dim
    smoke_test_shapes(embedding_dim=embedding_dim, latent_dim=latent_dim)

    model = CompressionModel(n=embedding_dim, latent_dim=latent_dim).to(device)
    model = train_model(
        model=model,
        dataset=dataset,
        dataset_path=dataset_path,
        lr=lr,
        epochs=full_train_epochs,
        batch_size=batch_size,
    )

    overfit_loss = tiny_overfit_test(model, dataset, steps=15, lr=lr)
    print(f"[overfit-check] final_loss={overfit_loss:.4f}")
    print(f"Saved best checkpoint to {SAVE_PATH}")
    return model


if __name__ == "__main__":
    train(
        full_train_epochs=FULL_TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        dataset_path=DATASET_PATH,
        min_profiles=MIN_PROFILES,
        use_cache=USE_CACHE,
    )
