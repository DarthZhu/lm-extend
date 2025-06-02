#!/bin/bash
# ========================================================
# Complete QwenVL Training Launch Script with Full Parameter Documentation
# ========================================================

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"  # [Required] Master node IP for multi-GPU training
MASTER_PORT=$(shuf -i 20000-29999 -n 1)  # Random port to avoid conflicts
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detect available GPUs

# ======================
# Path Configuration
# ======================
MODEL_PATH="data/local_llms/Qwen2-VL-7B-Instruct-Qwen2-merged-weighted-all"  # [ModelArguments] Pretrained model path
OUTPUT_DIR="output/checkpoints/mm"  # Directory for saving checkpoints
CACHE_DIR="./cache"  # [TrainingArguments] Cache directory for models

# ======================
# Model Configuration
# ======================
DATASETS="training_mm_dataset"  # [DataArguments] Dataset with sampling rate

# ======================
# Training Launch
# ======================
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_llm True \
         --tune_mm_vision False \
         --tune_mm_mlp False \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --cache_dir $CACHE_DIR \
         --bf16 \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps 16 \
         --torch_empty_cache_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 2048 \
         --data_flatten True \
         --max_pixels $((288*28*28)) \
         --min_pixels $((16*28*28)) \
         --base_interval 2 \
         --video_max_frames 8 \
         --video_min_frames 4 \
         --video_max_frame_pixels $((1664*28*28)) \
         --video_min_frame_pixels $((256*28*28)) \
         --num_train_epochs 1 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 100 \
         --save_total_limit 10 \
         --deepspeed zero3.json



# 10/20 step eval
# lower lr