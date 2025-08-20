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
  "meta-llama/Llama-4-Scout-17B-16E-Instruct:10"
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8:10"
  "microsoft/Phi-3.5-vision-instruct:16" 
  "llava-hf/LLaVA-NeXT-Video-34B-hf:8"
  "mistral-community/pixtral-12b:8"
  "rhymes-ai/Aria-Chat:8" ### transformers==4.45.2, vllm==0.6.6
  "OpenGVLab/InternVL2_5-8B:4"
  "OpenGVLab/InternVL2_5-38B:4"
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