#!/bin/bash
# compare_sft_dpo.sh

export CUDA_VISIBLE_DEVICES=0

python compare_sft_dpo.py \
    --sft_ckpt /home/user/data/2025/wen/train_llm/checkpoints/sft_h768_l8_lr2e-05_bs32_20260815_205446_final.pth \
    --dpo_ckpt /home/user/data/2025/wen/train_llm/checkpoints/dpo_h768_l8_lr4e-08_beta0.1_20260819_213034_final.pth 