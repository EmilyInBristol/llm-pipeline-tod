#!/bin/bash

mkdir -p logs

BASE_MODEL="Qwen/Qwen3-4B"
# BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
LORA_MODEL="./qwen3_lora_output"
# INFER_MODE="domain"  # 可选：domain 或 state
INFER_MODE="state"

BASE_TAG=${BASE_MODEL//\//_}

if [ "$INFER_MODE" = "domain" ]; then
    PROMPT_ARG="--prompt_file domain_recognition_prompts.json"
    LOG_PREFIX="domain"
elif [ "$INFER_MODE" = "state" ]; then
    PROMPT_ARG="--state_prompt_file state_extraction_prompts.json"
    LOG_PREFIX="state"
elif [ "$INFER_MODE" = "response" ]; then
    PROMPT_ARG="--response_prompt_file response_generation_prompts.json"
    LOG_PREFIX="response"
else
    echo "未知INFER_MODE: $INFER_MODE"
    exit 1
fi

# 1. 只用基础模型
LORA_TAG="none"
python3 infer_llm.py --base_model $BASE_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_base.log 2>&1

# # 2. 用基础模型 + 8bit
# LORA_TAG="none"
# python3 infer_llm.py --base_model $BASE_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --load_in_8bit --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_base_8bit.log 2>&1

# # 3. 用基础模型 + 4bit
# LORA_TAG="none"
# python3 infer_llm.py --base_model $BASE_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --load_in_4bit --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_base_4bit.log 2>&1

# # 4. 用lora
# LORA_TAG=${LORA_MODEL//\//_}
# python3 infer_llm.py --base_model $BASE_MODEL --lora_model $LORA_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --use_lora --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_lora.log 2>&1

# # 5. 用lora + 8bit
# LORA_TAG=${LORA_MODEL//\//_}
# python3 infer_llm.py --base_model $BASE_MODEL --lora_model $LORA_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --use_lora --load_in_8bit --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_lora_8bit.log 2>&1

# # 6. 用lora + 4bit
# LORA_TAG=${LORA_MODEL//\//_}
# python3 infer_llm.py --base_model $BASE_MODEL --lora_model $LORA_MODEL $PROMPT_ARG --infer_mode $INFER_MODE --use_lora --load_in_4bit --max_samples 100 > logs/${LOG_PREFIX}_${BASE_TAG}_${LORA_TAG}_lora_4bit.log 2>&1 