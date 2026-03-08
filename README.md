# Scalable Weighted Agent Reaction Model

Agents (personality vectors from descriptions + 100 archetypes) react to media; the network exposes content to a representative 10%, then propagates via similarity-based sharing.

## Pipeline overview

1. **Agents**: Description string → embedding → preprocessor (weights over 100 archetypes) → **384D personality vector** (`preprocessor.py`, `embedding_engine.py`).
2. **Network**: `network.py` takes `n` agents (UIDs + 384D vectors), runs k-means, selects ~10% representatives, gets reactions via `reaction_fn(uid, media)`, maps reactions to actions (VADER + 7 action levels), and for each sharer finds 1–3 similar users to share to (cosine + same-cluster bonus).

## Hugging Face token

To avoid `UNEXPECTED` warnings when loading the sentence-transformers model, set your Hugging Face token:

- **Option A:** Copy `.env.example` to `.env` and set `HF_TOKEN=your_token` (get one at https://huggingface.co/settings/tokens). If `python-dotenv` is installed, it will be loaded automatically.
- **Option B:** `export HF_TOKEN=your_token` (or `HUGGING_FACE_HUB_TOKEN`) before running.

You can also pass `token=...` into `EmbeddingEngine(..., token=...)`.  
**If `.env` was already committed:** run `git rm --cached .env` then commit so Git stops tracking it; `.gitignore` only applies to untracked files.

## How to run the full pipeline (100 agents → network)

**Prerequisites**

- `hybrid_review_dataset.json` (100 descriptions; or fallback from `personality_answers.json`).
- `preprocessor_checkpoint.pt`: build with `python preprocessor.py` (after persona encoder and dataset).
- `answer_predictor_checkpoint.pt`: train with `python train_answer_predictor.py`.

**Single entry point**

```bash
python run_full_pipeline.py
```

Optional args: `--media "Your reel text"` `--seed 42` `--dataset path/to/hybrid_review_dataset.json`.

This script:

- Loads 100 agent descriptions from the dataset.
- Computes 100×384 personality vectors via `preprocessor(embed(description))`.
- Calls `network.run_media_pipeline(uids, vectors, media, reaction_fn)` with a `reaction_fn` that uses the answer predictor (description + media as question → reaction text).
- Prints representatives, sample reactions, and shares.

## File roles

| File | Role |
|------|------|
| `embedding_engine.py` | Sentence embeddings (e.g. 384D); encodes descriptions and questions. |
| `preprocessor.py` | Description embedding → weights over 100 archetypes → 384D persona vector. |
| `train_answer_predictor.py` | Trains model: (persona, question) → answer embedding; provides `predict_answer_embedding`, `nearest_answer_text`. |
| `network.py` | K-means on agents, 10% representatives, VADER→actions, similarity-based sharing (1–3 recipients per sharer). |
| `run_full_pipeline.py` | **Entry point**: 100 agents from dataset → 384D vectors → `run_media_pipeline` with answer-predictor reaction fn. |
| `generate_personas.py` | Generates 100 archetypes × 10 profiles (Q&A) for persona encoder / preprocessor data. |
| `compression_model.py` | Persona encoder: 10 Q/A pairs → latent vector (used to build archetype personas in preprocessor). |
| `kmean_graph.py` | 2D coords (PCA of vectors) + share edges for visualization. |

## Quick test (no checkpoints)

To test the network with synthetic 384D vectors and placeholder reactions:

```bash
# From project root: use 384-dim in example (edit PERSONALITY_DIM in network.py is already 384)
python testing/example_100_agents.py
```

That uses random clustered vectors and a stub reaction function; for real agents and answer-predictor reactions, use `run_full_pipeline.py` after building the preprocessor and training the answer predictor.
