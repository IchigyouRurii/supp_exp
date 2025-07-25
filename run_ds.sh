export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6
export NCCL_P2P_LEVEL="NVL" 
export MODEL_PATH='/data/shared/users/luojing/Llama-3.1-8b'
export SAVE_PATH='/data/shared/users/luojing/comp_test/Llama-3.1-8b-persona'
export DS_SKIP_CUDA_CHECK=1

config_file="/home/luojing/accelerator_config_zero2.yaml"

accelerate launch --num_processes 6 --main_process_port 12345 --config_file $config_file /home/luojing/ProjectCode/PersonaMath/Train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path "/home/luojing/ProjectFile/PersonaMath/compare_test_persona.json" \
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
    --logging_steps 1 \