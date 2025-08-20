#!/bin/bash

# Environment variable setup
export VLLM_CONFIGURE_LOGGING=0
export VLLM_LOGGING_LEVEL=ERROR
export PYTHONWARNINGS="ignore::UserWarning"
export TOKENIZERS_PARALLELISM=false

# Common parameters
TOTAL_FRAMES=-1
MAX_NUM=-1
DATA_PATHS=(
  "data/real_mc.json"
  "data/synthetic_mc.json"
)
OPTIONS="--overwrite"

# Models to run
MODELS=(
  "gemini-2.5-pro-preview-06-05"
  # "gemini-2.5-flash-preview-05-20"
)

PROMPTS=(
    "cot"
    "direct-output"
)

# Execute the script for each model
for DATA_PATH in "${DATA_PATHS[@]}"; do
  for PROMPT in "${PROMPTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      python main.py --model "$MODEL" \
                    --prompt "$PROMPT" \
                    --total_frames "$TOTAL_FRAMES" \
                    --max_num "$MAX_NUM" \
                    --data_path "$DATA_PATH" \
                    $OPTIONS
    done
  done
done