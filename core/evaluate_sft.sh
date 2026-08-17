#!/bin/bash


python evaluate_sft.py \
    --ckpt '/home/user/data/2025/wen/train_llm/checkpoints/sft_h768_l8_lr2e-05_bs32_20260815_205446_final.pth' \
    --repetition_penalty 1.2
