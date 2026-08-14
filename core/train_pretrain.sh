#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_pretrain.py \
    --processed_dir /home/user/data/2025/wen/train_llm/data/processed \
    --checkpoint_dir /home/user/data/2025/wen/train_llm/checkpoints \
    --runs_dir /home/user/data/2025/wen/train_llm/runs \
    --epochs 2 \
    --lr 5e-4 \
    --micro_batch_size 4 \
    --effective_batch_size 48 \
    --seed 42 \
    --log_every 10 \
    --vocab_size 6400 \
    --hidden_size 768 \
    --num_layers 8 \
    --num_heads 8 \
    --keep_last_n_ckpt 1 \
    --max_eval_batches 200