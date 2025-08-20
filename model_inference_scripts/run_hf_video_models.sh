#!/bin/bash
export VLLM_LOGGING_LEVEL=ERROR

# Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Common parameters
MAX_NUM=-1
TOTAL_FRAMES=-1
DATA_PATHS=(
  "data/real_mc.json"
  "data/synthetic_mc.json"
)

OPTIONS="--overwrite"

# Model configurations (conda environment and model name)
# videollama3 requirements: 
# internvideo requirements:
MODELS=(
  "internvideo2:OpenGVLab/InternVideo2-Chat-8B"
  "internvideo25:OpenGVLab/InternVideo2_5_Chat_8B"
  "videollama3:DAMO-NLP-SG/VideoLLaMA3-7B"
  "videollama3:DAMO-NLP-SG/VideoLLaMA3-2B"
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