#!/bin/bash

# ============================================================
# 训练 DPO 模型 - 实验记录
# ============================================================
# 【参数变更记录】
# 2026-08-20: lr 从 5e-8 调整到 5e-6、beta从0.1调整到0.3
# 2026-08-21：实验开始从几何中点出发，保持beta=0.3不变，lr调整到5e-7
# ============================================================

export CUDA_VISIBLE_DEVICES=0

python train_dpo.py \
    --processed_dir /home/user/data/2025/wen/train_llm/data/processed \
    --checkpoint_dir /home/user/data/2025/wen/train_llm/checkpoints \
    --runs_dir /home/user/data/2025/wen/train_llm/runs \
    --sft_ckpt /home/user/data/2025/wen/train_llm/checkpoints/sft_h768_l8_lr2e-05_bs32_20260815_205446_final.pth \
    --epochs 2 \
    --lr 5e-7 \
    --beta 0.3 \
    --micro_batch_size 4 \
    --effective_batch_size 16 \
    --seed 42 \
    --log_every 100 \
    --eval_every 500 \
    --warmup_ratio 0.03 \
    --val_ratio 0.02 \
    --keep_last_n_ckpt 3 \
    --vocab_size 6400 \
    --hidden_size 768 \
    --num_layers 8 \
    --num_heads 8