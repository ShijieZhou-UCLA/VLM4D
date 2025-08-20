#!/bin/bash

# Environment variable setup
export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# Common parameters
MAX_NUM=-1
DATA_PATHS=(
  "data/real_mc.json"
  "data/synthetic_mc.json"
)
OPTIONS="--overwrite"

# Model configurations (model name and corresponding total_frames)
MODELS=(
  "gpt-4o:32"
  "claude-sonnet-4-20250514:20"
  "grok-2-vision-latest:16"
)

PROMPTS=(
    "cot"
    "direct-output"
)

# Execute the script for each model
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for ENTRY in "${MODELS[@]}"; do
      IFS=":" read -r MODEL TOTAL_FRAMES <<< "$ENTRY"
      python main.py --model "$MODEL" \
                     --prompt "$PROMPT" \
                     --total_frames "$TOTAL_FRAMES" \
                     --max_num "$MAX_NUM" \
                     --data_path "$DATA_PATH" \
                     $OPTIONS
    done
  done
done