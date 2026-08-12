# preprocess_pretrain.py
"""
预训练数据预处理：jsonl → .bin

"""

import json
import time
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from rich import print as rprint

from config import RAW_DATA_DIR,PROCESSED_DATA_DIR

def preprocess_pretrain(
    input_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
    add_bos: bool = False,
    add_eos: bool = False,
):
    """
    将 jsonl 预训练数据转换为 .bin 文件
    
    Args:
        input_path: 原始 jsonl 文件路径
        output_dir: 输出目录
        tokenizer_path: tokenizer.json 路径
        max_seq_len: 序列长度
        add_bos: 是否加 BOS
        add_eos: 是否加 EOS
    """
    if Path.exists(PROCESSED_DATA_DIR):
        pass
    else:
        PROCESSED_DATA_DIR.mkdir(parents=True,exist_ok=True)

    # ---------- 加载 tokenizer ----------
    rprint("[cyan] 加载 tokenizer...[/cyan]")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.token_to_id("<s>") or 1
    eos_id = tokenizer.token_to_id("</s>") or 2
    rprint(f"  vocab_size={vocab_size}, bos={bos_id}, eos={eos_id}")

    # ---------- 第一步：统计并预分配 ----------
    rprint("[cyan] 第一遍扫描：统计信息...[/cyan]")
    t0 = time.time()
    
    total_chars = 0
    num_lines = 0
    total_raw_tokens = 0

    with open(input_path,'r',encoding='utf-8') as f:
        for line in f:
            text = json.loads(line.strip()).get('text','')
            if not text.strip():
                continue
            num_lines += 1
            total_chars += len(text)
            ids = tokenizer.encode(text).ids
            total_raw_tokens += len(ids) + add_bos + add_eos

    rprint(f"  行数: {num_lines:,}")
    rprint(f"  总字符数: {total_chars:,}")
    rprint(f"  原始 token 数: {total_raw_tokens:,}")
    rprint(f"  耗时: {time.time()-t0:.1f}s")

    # ---------- 第二步：tokenize 并写入 .bin ----------
    rprint("[cyan] 第二遍：tokenize 并写入...[/cyan]")
    start_time = time.time()
    all_tokens = np.memmap(
        PROCESSED_DATA_DIR / "pretrain_tokens.bin",
        dtype=np.uint16 if vocab_size <= 65535 else np.uint32,
        mode='w+',
        shape=(total_raw_tokens,)
    )
