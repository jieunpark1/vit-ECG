#!/bin/bash

# =============================
# Training Script for ViT on PTB-XL (MI only)
# =============================

# 환경 설정
export DATASET_DIR="./data/ptbxl_mi"
export OUTPUT_DIR="./checkpoints/vit_ptbxl_mi"
export CONFIG_PATH="./configs/vit_config.yaml"
export LOG_DIR="./logs"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# 훈련 실행
python train.py \
  --dataset_dir $DATASET_DIR \
  --config $CONFIG_PATH \
  --output_dir $OUTPUT_DIR \
  --epochs 50 \
  --batch_size 8 \
  --lr 3e-4 \
  --num_workers 1 \
  --device cuda \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

