"""Sentence embedding helpers for the persona encoder."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Reduce Hugging Face / transformers load report before any HF import
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Load .env from project root (directory of this file) so HF_TOKEN is set before any HF import
def _load_env_from_project_root() -> None:
    _root = Path(__file__).resolve().parent
    _env_file = _root / ".env"
    if not _env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
        if os.environ.get("HF_TOKEN") and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
        return
    except ImportError:
        pass
    # Fallback without python-dotenv: parse KEY=VALUE lines and set os.environ
    with open(_env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value.startswith(('"', "'")) and value[0] == value[-1]:
                value = value[1:-1]
            if key:
                if key not in os.environ:
                    os.environ[key] = value
                # So HF Hub sees the token (it checks HUGGING_FACE_HUB_TOKEN)
                if key == "HF_TOKEN" and "HUGGING_FACE_HUB_TOKEN" not in os.environ:
                    os.environ["HUGGING_FACE_HUB_TOKEN"] = value


_load_env_from_project_root()

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Suppress "UNEXPECTED" / "Loading weights" warnings when loading the sentence-transformers model
for _name in ("transformers", "transformers.modeling_utils", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.ERROR)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_hf_token(token: str | None = None) -> str | None:
    """Hugging Face token: argument > HF_TOKEN > HUGGING_FACE_HUB_TOKEN env."""
    if token is not None and token.strip():
        return token.strip()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


@dataclass
class TrainingExample:
    questions: list[str]
    answers: list[str]
    archetype: str


@dataclass
class EmbeddedDataset:
    questions: torch.Tensor
    answers: torch.Tensor
    archetypes: list[str]
    archetype_embeddings: torch.Tensor
    embedding_dim: int

    def size(self) -> int:
        return int(self.questions.size(0))


class EmbeddingEngine:
    """SentenceTransformer wrapper for persona datasets."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dataset_path: str | Path | None = None,
        token: str | None = None,
    ):
        self.model_name = model_name
        hf_token = _get_hf_token(token)
        # Suppress verbose load report during model load
        _prev = logging.getLogger("transformers").level
        logging.getLogger("transformers").setLevel(logging.ERROR)
        try:
            self.model = SentenceTransformer(model_name, token=hf_token)
        finally:
            logging.getLogger("transformers").setLevel(_prev)
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Embed a batch of strings into vectors for the configured sentence model."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text], batch_size=1)[0]

    def load_dataset(self, dataset_path: str | Path) -> dict:
        with Path(dataset_path).open() as f:
            return json.load(f)

    def get_answer_keys(self, dataset: dict, limit: int | None = None) -> list[str]:
        answer_keys = dataset.get("answer_keys")
        if answer_keys:
            return answer_keys[:limit] if limit is not None else answer_keys

        responses = dataset.get("responses", [])
        if not responses:
            raise ValueError("Dataset has no responses")
        keys = list(responses[0]["answers"].keys())
        return keys[:limit] if limit is not None else keys

    def build_training_examples(
        self,
        dataset_path: str | Path,
        num_turns: int = 10,
    ) -> list[TrainingExample]:
        dataset = self.load_dataset(dataset_path)
        examples: list[TrainingExample] = []
        for response in dataset["responses"]:
            if "questions" in response and "answers" in response:
                qs = response["questions"]
                ans = response["answers"]
            else:
                questions = dataset.get("questions", [])
                answer_keys = self.get_answer_keys(dataset, limit=num_turns)
                if len(questions) < num_turns or len(answer_keys) < num_turns:
                    raise ValueError(f"Dataset needs {num_turns} questions and answer_keys")
                ans_d = response.get("answers", {})
                qs = questions[:num_turns]
                ans = [ans_d[k] for k in answer_keys]
            if len(qs) < num_turns or len(ans) < num_turns:
                raise ValueError(f"Response needs {num_turns} questions and answers")
            examples.append(
                TrainingExample(
                    questions=qs[:num_turns],
                    answers=ans[:num_turns],
                    archetype=response["persona"]["archetype"],
                )
            )
        return examples

    def default_cache_path(self, dataset_path: str | Path) -> Path:
        dataset_path = Path(dataset_path)
        safe_model = self.model_name.replace("/", "_")
        return dataset_path.with_suffix(f".{safe_model}.persona_100x10.embeddings.pt")

    def embed_training_examples(
        self,
        dataset_path: str | Path,
        device: torch.device,
        num_turns: int = 10,
        batch_size: int = 64,
        cache_path: str | Path | None = None,
        use_cache: bool = True,
    ) -> EmbeddedDataset:
        dataset_path = Path(dataset_path)
        cache_path = Path(cache_path) if cache_path is not None else self.default_cache_path(dataset_path)

        if use_cache and cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            return EmbeddedDataset(
                questions=cache["questions"].to(device),
                answers=cache["answers"].to(device),
                archetypes=list(cache["archetypes"]),
                archetype_embeddings=cache["archetype_embeddings"].to(device),
                embedding_dim=int(cache["embedding_dim"]),
            )

        examples = self.build_training_examples(
            dataset_path=dataset_path,
            num_turns=num_turns,
        )

        flat_questions: list[str] = []
        flat_answers: list[str] = []
        archetypes: list[str] = []

        for example in examples:
            flat_questions.extend(example.questions)
            flat_answers.extend(example.answers)
            archetypes.append(example.archetype)

        question_embeddings = self.encode(flat_questions, batch_size=batch_size)
        answer_embeddings = self.encode(flat_answers, batch_size=batch_size)
        archetype_embeddings = self.encode(archetypes, batch_size=batch_size)

        num_examples = len(examples)
        embedding_dim = self.embedding_dim
        embedded = EmbeddedDataset(
            questions=torch.tensor(
                question_embeddings.reshape(num_examples, num_turns, embedding_dim),
                dtype=torch.float32,
                device=device,
            ),
            answers=torch.tensor(
                answer_embeddings.reshape(num_examples, num_turns, embedding_dim),
                dtype=torch.float32,
                device=device,
            ),
            archetypes=archetypes,
            archetype_embeddings=torch.tensor(
                archetype_embeddings,
                dtype=torch.float32,
                device=device,
            ),
            embedding_dim=embedding_dim,
        )

        if use_cache:
            torch.save(
                {
                    "questions": embedded.questions.cpu(),
                    "answers": embedded.answers.cpu(),
                    "archetypes": embedded.archetypes,
                    "archetype_embeddings": embedded.archetype_embeddings.cpu(),
                    "embedding_dim": embedded.embedding_dim,
                },
                cache_path,
            )
        return embedded


def sentence_to_vector(sentence: str) -> np.ndarray:
    return EmbeddingEngine().encode_one(sentence)


def to_matrix(strings: list[str], batch_size: int = 32) -> np.ndarray:
    return EmbeddingEngine().encode(strings, batch_size=batch_size)
