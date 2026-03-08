"""Preprocessor: description embedding -> 384D persona via weighted blend of archetype personas."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEMPERATURE = 0.55
PREPROCESSOR_CHECKPOINT = "preprocessor_checkpoint.pt"


class PreprocessorModel(nn.Module):
    """
    Takes a 384D description embedding, computes cosine similarity with 100 archetype
    embeddings -> 100-dim vector, double softmax with temperature 0.55, then weighted
    sum of archetype personas -> 384D output.
    """

    def __init__(
        self,
        archetype_embeddings: torch.Tensor,
        archetype_personas: torch.Tensor,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        super().__init__()
        if archetype_embeddings.shape != archetype_personas.shape:
            raise ValueError("archetype_embeddings and archetype_personas must match shape")
        if archetype_embeddings.dim() != 2:
            raise ValueError("Expected (100, 384)")
        self.num_archetypes = archetype_embeddings.size(0)
        self.dim = archetype_embeddings.size(1)
        self.temperature = temperature
        self.register_buffer("archetype_embeddings", F.normalize(archetype_embeddings, dim=-1))
        self.register_buffer("archetype_personas", archetype_personas)

    def forward(self, description_embedding: torch.Tensor) -> torch.Tensor:
        """
        description_embedding: (B, 384)
        returns: (B, 384)
        """
        desc_norm = F.normalize(description_embedding, dim=-1)
        sims = desc_norm @ self.archetype_embeddings.T  # (B, 100)
        weights = F.softmax(sims / self.temperature, dim=-1)
        weights = F.softmax(weights / self.temperature, dim=-1)  # double softmax
        return weights @ self.archetype_personas  # (B, 384)


def build_preprocessor_artifacts(
    embedding_engine,
    persona_encoder,
    embedded_dataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build archetype_embeddings (100, 384) and archetype_personas (100, 384).
    - archetype_embeddings: from dataset archetypes (one per unique archetype)
    - archetype_personas: mean persona encoder output per archetype
    """
    archetypes = embedded_dataset.archetypes
    questions = embedded_dataset.questions
    answers = embedded_dataset.answers
    persona_encoder.eval()

    unique_archetypes = list(dict.fromkeys(archetypes))

    # Embed unique archetypes for cosine similarity
    arch_embs = embedding_engine.encode(unique_archetypes)
    archetype_embeddings = torch.tensor(arch_embs, dtype=torch.float32, device=device)

    # Compute mean persona per archetype
    with torch.no_grad():
        all_personas = persona_encoder.encode_persona(questions, answers)  # (N, 384)
    arch_to_indices: dict[str, list[int]] = {a: [] for a in unique_archetypes}
    for i, a in enumerate(archetypes):
        arch_to_indices[a].append(i)
    mean_personas = []
    for a in unique_archetypes:
        idx = arch_to_indices[a]
        mean_personas.append(all_personas[idx].mean(dim=0))
    archetype_personas = torch.stack(mean_personas)

    return archetype_embeddings, archetype_personas


def save_preprocessor(
    preprocessor: PreprocessorModel,
    path: str | Path = PREPROCESSOR_CHECKPOINT,
) -> None:
    torch.save(
        {
            "archetype_embeddings": preprocessor.archetype_embeddings,
            "archetype_personas": preprocessor.archetype_personas,
            "temperature": preprocessor.temperature,
            "num_archetypes": preprocessor.num_archetypes,
            "dim": preprocessor.dim,
        },
        path,
    )


def build_and_save_preprocessor(
    persona_encoder_path: str | Path = "persona_encoder_checkpoint.pt",
    dataset_path: str | Path = "personality_answers.json",
    output_path: str | Path = PREPROCESSOR_CHECKPOINT,
    device: torch.device | None = None,
) -> PreprocessorModel:
    """Build preprocessor from trained persona encoder and embedded dataset, save to disk."""
    from embedding_engine import EmbeddingEngine
    from train import get_device, load_checkpoint

    device = device or get_device()
    persona_encoder, _ = load_checkpoint(persona_encoder_path, device=device)
    persona_encoder.eval()
    engine = EmbeddingEngine(dataset_path=dataset_path)
    embedded = engine.embed_training_examples(dataset_path=dataset_path, device=device)
    archetype_embeddings, archetype_personas = build_preprocessor_artifacts(
        engine, persona_encoder, embedded, device
    )
    preprocessor = PreprocessorModel(
        archetype_embeddings=archetype_embeddings,
        archetype_personas=archetype_personas,
    ).to(device)
    save_preprocessor(preprocessor, output_path)
    return preprocessor


def load_preprocessor(
    path: str | Path = PREPROCESSOR_CHECKPOINT,
    device: torch.device | None = None,
    temperature: float | None = None,
) -> PreprocessorModel:
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PreprocessorModel(
        archetype_embeddings=ckpt["archetype_embeddings"],
        archetype_personas=ckpt["archetype_personas"],
        temperature=temperature if temperature is not None else ckpt.get("temperature", DEFAULT_TEMPERATURE),
    ).to(device)
    return model


if __name__ == "__main__":
    from generate_personas import DEFAULT_OUTPUT_PATH, ensure_dataset_exists
    from train import get_device

    ensure_dataset_exists(DEFAULT_OUTPUT_PATH)
    device = get_device()
    build_and_save_preprocessor(
        persona_encoder_path="persona_encoder_checkpoint.pt",
        dataset_path=str(DEFAULT_OUTPUT_PATH),
        output_path=PREPROCESSOR_CHECKPOINT,
        device=device,
    )
    print(f"Preprocessor saved to {PREPROCESSOR_CHECKPOINT}")
