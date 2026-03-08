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
VAL_RATIO = 0.2  # larger val set = more reliable early stopping
USE_CACHE = True
PATIENCE = 100  # early stop quickly
WEIGHT_DECAY = 0.01  # moderate L2 (0.1 was over-regularizing)
DROPOUT = 0.3  # moderate dropout for MLP
INPUT_NOISE = 0.03  # light Gaussian noise on inputs
MODEL_TYPE = "resnet"  # "linear" | "mlp" | "resnet" | "attention"
GRAD_CLIP = 1.0  # gradient clipping for stability
ATTENTION_HIDDEN_DIM = 512
ATTENTION_NUM_HEADS = 8
ATTENTION_NUM_LAYERS = 2
ATTENTION_FF_MULTIPLIER = 4
MLP_HIDDEN_DIM = 128  # more capacity for better learning


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

    def __init__(self, hidden_dim: int, num_heads: int, ff_multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ff_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ff_multiplier, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.drop(attn_output))
        return self.norm2(tokens + self.ff(tokens))


class LegacyAnswerPredictor(nn.Module):
    """MLP predictor: concat(persona, question) -> hidden -> output. Generalizes well on small data."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        num_layers: int = 2,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (embedding_dim * 2)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        layers: list[nn.Module] = []
        in_dim = embedding_dim * 2
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        persona_vectors: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([persona_vectors, question_embeddings], dim=-1)
        return self.net(x)


class LinearAnswerPredictor(nn.Module):
    """Minimal linear predictor: concat(persona, question) -> output."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim  # for checkpoint compat
        self.net = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        persona_vectors: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([persona_vectors, question_embeddings], dim=-1)
        return self.net(x)


class ResNetAnswerPredictor(nn.Module):
    """ResNet-style MLP: residual blocks with skip connections for better gradient flow."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = MLP_HIDDEN_DIM,
        num_blocks: int = 2,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout
        self.input_proj = nn.Linear(embedding_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        persona_vectors: torch.Tensor,
        question_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([persona_vectors, question_embeddings], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x) + x  # residual
        return self.output_proj(x)


class ResNetBlock(nn.Module):
    """Single residual block: Linear -> ReLU -> LayerNorm -> Dropout."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_multiplier = ff_multiplier
        self.dropout_rate = dropout

        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.token_type_embeddings = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
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
    return 1.0 - F.cosine_similarity(predicted, target, dim=-1).mean()


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
    return tensor.index_select(0, torch.tensor(indices, dtype=torch.long, device=tensor.device))


def evaluate(
    model: nn.Module,
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
            "num_blocks": getattr(model, "num_blocks", None),
            "ff_multiplier": getattr(model, "ff_multiplier", None),
            "dropout": getattr(model, "dropout_rate", DROPOUT),
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
            dropout=float(checkpoint.get("dropout", DROPOUT)),
        ).to(device)
    elif model_type == "linear":
        model = LinearAnswerPredictor(embedding_dim=int(checkpoint["embedding_dim"])).to(device)
    elif model_type == "resnet":
        model = ResNetAnswerPredictor(
            embedding_dim=int(checkpoint["embedding_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_blocks=int(checkpoint.get("num_blocks", 2)),
            dropout=float(checkpoint.get("dropout", DROPOUT)),
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
    print(f"device: {device}")

    dataset = build_tensor_dataset(
        dataset_path=dataset_path,
        preprocessor_path=preprocessor_path,
        device=device,
        use_cache=use_cache,
    )
    print(f"loaded {dataset.size()} QA pairs from {dataset_path}")

    train_indices, val_indices = split_indices(dataset.size(), val_ratio=val_ratio)
    if MODEL_TYPE == "linear":
        model = LinearAnswerPredictor(embedding_dim=dataset.embedding_dim).to(device)
    elif MODEL_TYPE == "mlp":
        model = LegacyAnswerPredictor(
            embedding_dim=dataset.embedding_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            dropout=DROPOUT,
        ).to(device)
    elif MODEL_TYPE == "resnet":
        model = ResNetAnswerPredictor(
            embedding_dim=dataset.embedding_dim,
            hidden_dim=MLP_HIDDEN_DIM,
            num_blocks=2,
            dropout=DROPOUT,
        ).to(device)
    else:
        model = AnswerPredictor(
            embedding_dim=dataset.embedding_dim,
            hidden_dim=ATTENTION_HIDDEN_DIM,
            num_heads=ATTENTION_NUM_HEADS,
            num_layers=ATTENTION_NUM_LAYERS,
            ff_multiplier=ATTENTION_FF_MULTIPLIER,
            dropout=DROPOUT,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=MIN_LEARNING_RATE,
    )

    best_val_loss = float("inf")
    no_improve = 0
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
            if INPUT_NOISE > 0:
                persona_batch = persona_batch + torch.randn_like(persona_batch, device=persona_batch.device) * INPUT_NOISE
                question_batch = question_batch + torch.randn_like(question_batch, device=question_batch.device) * INPUT_NOISE

            optimizer.zero_grad()
            predictions = model(persona_batch, question_batch)
            loss = cosine_embedding_loss(predictions, answer_batch)
            loss.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += float(loss.item())
            batch_count += 1

        train_loss = total_loss / max(1, batch_count)
        val_loss = evaluate(model, dataset, val_indices)
        current_lr = scheduler.get_last_lr()[0]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            save_checkpoint(
                model=model,
                dataset=dataset,
                dataset_path=dataset_path,
                preprocessor_path=preprocessor_path,
                val_loss=val_loss,
                path=checkpoint_path,
            )
        else:
            no_improve += 1
        print(f"epoch {epoch + 1}/{epochs} train={train_loss:.4f} val={val_loss:.4f} lr={current_lr:.6f}")
        scheduler.step(val_loss)
        if no_improve >= PATIENCE:
            print(f"early stop (no improvement for {PATIENCE} epochs)")
            break

    print(f"saved best checkpoint -> {checkpoint_path}")
    return model


if __name__ == "__main__":
    train_answer_predictor()
