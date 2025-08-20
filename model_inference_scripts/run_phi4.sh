#!/bin/bash
export VLLM_LOGGING_LEVEL=ERROR

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Common parameters
MAX_NUM=-1
TOTAL_FRAMES=16
DATA_PATHS=(
  "data/real_mc.json"
  "data/synthetic_mc.json"
)

OPTIONS="--overwrite"

# Model configurations (conda environment and model name)
# videollama2 requirements: 
# internvideo requirements:
MODELS=(
  "phi4:microsoft/Phi-4-multimodal-instruct" ### transformers==4.48.2, vllm==0.8.2
)

PROMPTS=(
    "cot"
    "direct-output"
)

for DATA_PATH in "${DATA_PATHS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do
        for ENTRY in "${MODELS[@]}"; do
            IFS=":" read -r ENVIRONMENT MODEL <<< "$ENTRY"
            conda activate "$ENVIRONMENT"
            python main.py --model "$MODEL" \
                        --prompt "$PROMPT" \
                        --max_num "$MAX_NUM" \
                        --total_frames "$TOTAL_FRAMES" \
                        --data_path "$DATA_PATH" \
                        $OPTIONS
        done
    done
done