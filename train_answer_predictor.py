"""Train a description-conditioned answer predictor on hybrid review QA pairs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_engine import DEFAULT_MODEL_NAME, EmbeddingEngine
from preprocessor import PREPROCESSOR_CHECKPOINT, build_and_save_preprocessor, load_preprocessor
from train import get_device

DATASET_PATH = Path("hybrid_review_dataset.json")
CHECKPOINT_PATH = Path("answer_predictor_checkpoint.pt")
BATCH_SIZE = 16
EPOCHS = 250
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 1e-5
VAL_RATIO = 0.1
USE_CACHE = True
MODEL_TYPE = "attention"
ATTENTION_HIDDEN_DIM = 512
ATTENTION_NUM_HEADS = 8
ATTENTION_NUM_LAYERS = 2
ATTENTION_FF_MULTIPLIER = 4


@dataclass
class FlatReviewExample:
    description: str
    question: str
    answer: str
    sentiment_label: str


@dataclass
class ReviewTensorDataset:
    persona_vectors: torch.Tensor
    question_embeddings: torch.Tensor
    answer_embeddings: torch.Tensor
    answer_texts: list[str]
    sentiment_labels: list[str]
    embedding_dim: int

    def size(self) -> int:
        return int(self.question_embeddings.size(0))


class AttentionBlock(nn.Module):
    """Transformer-style block over the persona/question token pair."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn_output)
        ff_output = self.ff(tokens)
        return self.norm2(tokens + ff_output)


class LegacyAnswerPredictor(nn.Module):
    """Original MLP predictor kept for loading older checkpoints."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or (embedding_dim * 2)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        persona_vectors: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([persona_vectors, question_embeddings], dim=-1)
        return self.net(x)


class AnswerPredictor(nn.Module):
    """Attention-based predictor over persona and question tokens."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = ATTENTION_HIDDEN_DIM,
        num_heads: int = ATTENTION_NUM_HEADS,
        num_layers: int = ATTENTION_NUM_LAYERS,
        ff_multiplier: int = ATTENTION_FF_MULTIPLIER,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_multiplier = ff_multiplier

        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.token_type_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        persona_vectors: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        persona_token = self.input_projection(persona_vectors)
        question_token = self.input_projection(question_embeddings)
        tokens = torch.stack([persona_token, question_token], dim=1)
        tokens = tokens + self.token_type_embeddings.unsqueeze(0)

        for block in self.blocks:
            tokens = block(tokens)

        fused_question_token = tokens[:, 1, :]
        return self.readout(fused_question_token)


def cosine_embedding_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    return 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()


def load_flat_examples(dataset_path: str | Path) -> list[FlatReviewExample]:
    with Path(dataset_path).open() as f:
        dataset = json.load(f)

    examples: list[FlatReviewExample] = []
    for response in dataset["responses"]:
        description = response["description"]
        for qa_pair in response["qa_pairs"]:
            examples.append(
                FlatReviewExample(
                    description=description,
                    question=qa_pair["question"],
                    answer=qa_pair["answer"],
                    sentiment_label=qa_pair["sentiment_label"],
                )
            )
    return examples


def default_cache_path(dataset_path: str | Path, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    dataset_path = Path(dataset_path)
    safe_model = model_name.replace("/", "_")
    return dataset_path.with_suffix(f".{safe_model}.answer_prediction.embeddings.pt")


def build_tensor_dataset(
    dataset_path: str | Path,
    preprocessor_path: str | Path,
    device: torch.device,
    use_cache: bool = True,
) -> ReviewTensorDataset:
    dataset_path = Path(dataset_path)
    cache_path = default_cache_path(dataset_path)
    if use_cache and cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        return ReviewTensorDataset(
            persona_vectors=cache["persona_vectors"].to(device),
            question_embeddings=cache["question_embeddings"].to(device),
            answer_embeddings=cache["answer_embeddings"].to(device),
            answer_texts=list(cache["answer_texts"]),
            sentiment_labels=list(cache["sentiment_labels"]),
            embedding_dim=int(cache["embedding_dim"]),
        )

    if not Path(preprocessor_path).exists():
        build_and_save_preprocessor(output_path=preprocessor_path, device=device)

    examples = load_flat_examples(dataset_path)
    engine = EmbeddingEngine()
    preprocessor = load_preprocessor(preprocessor_path, device=device)
    preprocessor.eval()

    unique_descriptions = list(dict.fromkeys(example.description for example in examples))
    description_matrix = torch.tensor(
        engine.encode(unique_descriptions),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        persona_matrix = preprocessor(description_matrix)
    description_to_persona = {
        description: persona_matrix[idx]
        for idx, description in enumerate(unique_descriptions)
    }

    question_embeddings = torch.tensor(
        engine.encode([example.question for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    answer_embeddings = torch.tensor(
        engine.encode([example.answer for example in examples]),
        dtype=torch.float32,
        device=device,
    )
    persona_vectors = torch.stack([description_to_persona[example.description] for example in examples])

    dataset = ReviewTensorDataset(
        persona_vectors=persona_vectors,
        question_embeddings=question_embeddings,
        answer_embeddings=answer_embeddings,
        answer_texts=[example.answer for example in examples],
        sentiment_labels=[example.sentiment_label for example in examples],
        embedding_dim=engine.embedding_dim,
    )

    if use_cache:
        torch.save(
            {
                "persona_vectors": dataset.persona_vectors.cpu(),
                "question_embeddings": dataset.question_embeddings.cpu(),
                "answer_embeddings": dataset.answer_embeddings.cpu(),
                "answer_texts": dataset.answer_texts,
                "sentiment_labels": dataset.sentiment_labels,
                "embedding_dim": dataset.embedding_dim,
            },
            cache_path,
        )
    return dataset


def split_indices(n_items: int, val_ratio: float, seed: int = 42) -> tuple[list[int], list[int]]:
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(n_items * val_ratio))
    return indices[val_size:], indices[:val_size]


def select_batch(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(0, index_tensor)


def evaluate(
    model: AnswerPredictor,
    dataset: ReviewTensorDataset,
    indices: list[int],
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(
            select_batch(dataset.persona_vectors, indices),
            select_batch(dataset.question_embeddings, indices),
        )
        targets = select_batch(dataset.answer_embeddings, indices)
        return float(cosine_embedding_loss(predictions, targets).item())


def save_checkpoint(
    model: nn.Module,
    dataset: ReviewTensorDataset,
    dataset_path: str | Path,
    preprocessor_path: str | Path,
    val_loss: float,
    path: str | Path = CHECKPOINT_PATH,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "model_type": MODEL_TYPE,
            "num_heads": getattr(model, "num_heads", None),
            "num_layers": getattr(model, "num_layers", None),
            "ff_multiplier": getattr(model, "ff_multiplier", None),
            "dataset_path": str(dataset_path),
            "preprocessor_path": str(preprocessor_path),
            "val_loss": val_loss,
            "answer_texts": dataset.answer_texts,
            "answer_embeddings": dataset.answer_embeddings.cpu(),
            "sentiment_labels": dataset.sentiment_labels,
        },
        path,
    )


def load_checkpoint(
    path: str | Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "mlp")
    if model_type == "attention":
        model = AnswerPredictor(
            embedding_dim=int(checkpoint["embedding_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_heads=int(checkpoint.get("num_heads", ATTENTION_NUM_HEADS)),
            num_layers=int(checkpoint.get("num_layers", ATTENTION_NUM_LAYERS)),
            ff_multiplier=int(checkpoint.get("ff_multiplier", ATTENTION_FF_MULTIPLIER)),
        ).to(device)
    else:
        model = LegacyAnswerPredictor(
            embedding_dim=int(checkpoint["embedding_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_answer_embedding(
    description: str,
    question: str,
    model: nn.Module,
    preprocessor_path: str | Path = PREPROCESSOR_CHECKPOINT,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or get_device()
    engine = EmbeddingEngine()
    preprocessor = load_preprocessor(preprocessor_path, device=device)
    preprocessor.eval()

    description_embedding = torch.tensor(
        engine.encode([description]),
        dtype=torch.float32,
        device=device,
    )
    question_embedding = torch.tensor(
        engine.encode([question]),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        persona_vector = preprocessor(description_embedding)
        return model(persona_vector, question_embedding)[0]


def nearest_answer_text(predicted_embedding: torch.Tensor, checkpoint: dict) -> tuple[str, str, float]:
    answer_bank = checkpoint["answer_embeddings"].to(predicted_embedding.device)
    scores = F.cosine_similarity(
        F.normalize(predicted_embedding.unsqueeze(0), dim=-1),
        F.normalize(answer_bank, dim=-1),
        dim=-1,
    )
    best_idx = int(scores.argmax().item())
    return (
        checkpoint["answer_texts"][best_idx],
        checkpoint["sentiment_labels"][best_idx],
        float(scores[best_idx].item()),
    )


def train_answer_predictor(
    dataset_path: str | Path = DATASET_PATH,
    preprocessor_path: str | Path = PREPROCESSOR_CHECKPOINT,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    val_ratio: float = VAL_RATIO,
    use_cache: bool = USE_CACHE,
) -> AnswerPredictor:
    device = get_device()
    print(f"Using device: {device}")

    dataset = build_tensor_dataset(
        dataset_path=dataset_path,
        preprocessor_path=preprocessor_path,
        device=device,
        use_cache=use_cache,
    )
    print(f"Loaded {dataset.size()} question-answer training pairs from {dataset_path}")

    train_indices, val_indices = split_indices(dataset.size(), val_ratio=val_ratio)
    model = AnswerPredictor(
        embedding_dim=dataset.embedding_dim,
        hidden_dim=ATTENTION_HIDDEN_DIM,
        num_heads=ATTENTION_NUM_HEADS,
        num_layers=ATTENTION_NUM_LAYERS,
        ff_multiplier=ATTENTION_FF_MULTIPLIER,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=MIN_LEARNING_RATE,
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        shuffled = train_indices[:]
        random.shuffle(shuffled)
        total_loss = 0.0
        batch_count = 0

        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            persona_batch = select_batch(dataset.persona_vectors, batch_indices)
            question_batch = select_batch(dataset.question_embeddings, batch_indices)
            answer_batch = select_batch(dataset.answer_embeddings, batch_indices)

            optimizer.zero_grad()
            predictions = model(persona_batch, question_batch)
            loss = cosine_embedding_loss(predictions, answer_batch)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        train_loss = total_loss / max(1, batch_count)
        val_loss = evaluate(model, dataset, val_indices)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[answer-predictor] epoch={epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={current_lr:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                dataset=dataset,
                dataset_path=dataset_path,
                preprocessor_path=preprocessor_path,
                val_loss=val_loss,
                path=checkpoint_path,
            )
        scheduler.step()

    print(f"Saved best checkpoint to {checkpoint_path}")
    return model


if __name__ == "__main__":
    train_answer_predictor()
