"""Persona encoder: stacked Q/A pairs -> nD vector (n = embedding dim)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressionModel(nn.Module):
    """
    Stacked Q/A pipeline:
    1. Stack aligned Q/A pairs into 2×n matrices (2 rows, n cols) per slot
    2. Transform each 2×n via Linear(2*n, 10) -> 1×10 per slot
    3. Stack all 10 slots -> 10×10 matrix
    4. Transform via Linear(100, latent_dim) -> persona vector
    """

    def __init__(
        self,
        n: int = 64,
        num_slots: int = 10,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.n = n
        self.num_slots = num_slots
        self.latent_dim = latent_dim

        if latent_dim != n:
            raise ValueError("latent_dim must match embedding dim n")

        # Step 1->2: each 2×n -> 1×10
        self.pair_to_vec = nn.Linear(2 * n, num_slots)

        # Step 3->4: 10×10 (100) -> latent_dim
        self.matrix_to_latent = nn.Linear(num_slots * num_slots, latent_dim)

    def _validate_inputs(self, questions: torch.Tensor, answers: torch.Tensor) -> None:
        if questions.ndim != 3 or answers.ndim != 3:
            raise ValueError("questions and answers must be (B, 10, n)")
        if questions.shape != answers.shape:
            raise ValueError("questions and answers must match shape")
        if questions.size(1) != self.num_slots or questions.size(2) != self.n:
            raise ValueError(f"expected (B, {self.num_slots}, {self.n})")

    def encode_persona(self, questions: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        """
        questions: (B, 10, n)
        answers: (B, 10, n)
        returns: (B, latent_dim)
        """
        self._validate_inputs(questions, answers)
        B = questions.size(0)

        # 1. Stack Q/A pairs: each slot [Q[i]; A[i]] -> (2, n) -> flatten
        pairs = torch.stack([questions, answers], dim=2)  # (B, 10, 2, n)
        pairs_flat = pairs.reshape(B, self.num_slots, 2 * self.n)

        # 2. Transform each 2×n -> 1×10
        slot_vecs = self.pair_to_vec(pairs_flat)  # (B, 10, 10)

        # 3. Stack -> 10×10, flatten to (B, 100)
        matrix_flat = slot_vecs.reshape(B, -1)

        # 4. Transform 10×10 -> latent_dim
        return self.matrix_to_latent(matrix_flat)

    def forward(self, questions: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
        return self.encode_persona(questions, answers)

    encode_profile = encode_persona


def cosine_embedding_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean 1 - cosine_similarity over embedding pairs."""
    pred_norm = F.normalize(predicted, dim=-1)
    tgt_norm = F.normalize(target, dim=-1)
    cosine = (pred_norm * tgt_norm).sum(dim=-1)
    return (1.0 - cosine).mean()


def smoke_test_shapes(embedding_dim: int = 64, latent_dim: int = 64) -> None:
    batch = 3
    model = CompressionModel(n=embedding_dim, num_slots=10, latent_dim=latent_dim)
    questions = torch.randn(batch, 10, embedding_dim)
    answers = torch.randn(batch, 10, embedding_dim)
    persona = model.encode_persona(questions, answers)
    assert persona.shape == (batch, latent_dim)


if __name__ == "__main__":
    smoke_test_shapes()
    smoke_test_shapes(embedding_dim=384, latent_dim=384)  # all-MiniLM-L6-v2
    print("compression_model.py smoke test passed")
