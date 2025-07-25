#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6
export NCCL_P2P_LEVEL="NVL"
export MODEL_PATH='/data/shared/users/luojing/Llama-3.1-8b'
export DS_SKIP_CUDA_CHECK=1
config_file="/home/luojing/accelerator_config_zero2.yaml"

# Directory where the 11 JSON files are stored
DATA_DIR="/home/luojing/ProjectFile/PersonaMath/persona_class_split"

# Base directory for saving model outputs
BASE_SAVE_PATH="/data/shared/users/luojing/comp_test/Llama-3.1-8b-persona"

# Loop through all 11 persona_class files
for i in {0..10}; do
    DATA_PATH="${DATA_DIR}/persona_class_${i}.json"
    SAVE_PATH="${BASE_SAVE_PATH}/persona_class_${i}"

    echo "=============================="
    echo " Training on persona_class_${i}.json"
    echo " Saving model to ${SAVE_PATH}"
    echo "=============================="

    accelerate launch --num_processes 6 --main_process_port $((12345 + i)) --config_file $config_file /home/luojing/ProjectCode/PersonaMath/Train.py \
        --model_name_or_path $MODEL_PATH \
        --data_path $DATA_PATH \
        --data_length 10000000 \
        --output_dir $SAVE_PATH \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1
done
