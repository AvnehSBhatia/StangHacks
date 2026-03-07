#!/bin/bash
# Delete embedding caches and checkpoint to force fresh re-embed with current model
cd "$(dirname "$0")"
rm -f personality_answers*.embeddings.pt
rm -f personality_answers*.pt
rm -f archetype_descriptions*.embeddings.pt
rm -f archetype_descriptions*.pt
rm -f persona_encoder_checkpoint.pt
echo "Caches and checkpoint deleted. Run 'python3 train.py' to re-embed and retrain."
