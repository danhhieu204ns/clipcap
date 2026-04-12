#!/usr/bin/env bash
set -euo pipefail

# Edit these values to match your checkpoint and image.
IMAGE_PATH="./Images/img.jpg"
CHECKPOINT_PATH="./checkpoints/flickr30k_transformer_finetune/flickr30k_transformer_finetune-009.pt"
OUTPUT_DIR="./visualizations/token_focus"
OUTPUT_PREFIX="transformer_token_focus"

# Model config.
MAPPING_TYPE="transformer"
PREFIX_LENGTH=10
PREFIX_LENGTH_CLIP=10
NUM_LAYERS=8
CLIP_MODEL_TYPE="ViT-B/32"
MAX_STEPS=20
TEMPERATURE=1.0
DEVICE="cuda:0"

# Set to true if the checkpoint was trained with GPT-2 frozen.
ONLY_PREFIX=false

CMD=(
  python ./visualize_transformer_token_focus.py
  --image "$IMAGE_PATH"
  --checkpoint "$CHECKPOINT_PATH"
  --mapping_type "$MAPPING_TYPE"
  --prefix_length "$PREFIX_LENGTH"
  --prefix_length_clip "$PREFIX_LENGTH_CLIP"
  --num_layers "$NUM_LAYERS"
  --clip_model_type "$CLIP_MODEL_TYPE"
  --max_steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --device "$DEVICE"
  --out_dir "$OUTPUT_DIR"
  --output_prefix "$OUTPUT_PREFIX"
)

if [ "$ONLY_PREFIX" = true ]; then
  CMD+=(--only_prefix)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
