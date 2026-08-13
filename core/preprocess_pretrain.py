# preprocess_pretrain.py
"""
预训练数据预处理：jsonl → .bin

"""
import os
import json
import time
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from rich import print as rprint

from config import RAW_DATA_DIR,PROCESSED_DATA_DIR,MODEL_DIR

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
    rprint("[cyan]🔤 加载 tokenizer...[/cyan]")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.token_to_id("<s>") or 1
    eos_id = tokenizer.token_to_id("</s>") or 2
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    # ---------- 第一步：统计 ----------
    rprint("[cyan] 第一遍扫描：统计信息...[/cyan]")
    t0 = time.time()
    
    num_lines = 0
    total_raw_tokens = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = json.loads(line.strip()).get('text', '')
            if not text.strip():
                continue
            num_lines += 1
            total_raw_tokens += len(tokenizer.encode(text).ids) + int(add_bos) + int(add_eos)


    #有效的Token数(末尾的token不再计算)
    total_valid = (total_raw_tokens // max_seq_len) * max_seq_len

    rprint(f"  行数: {num_lines:,}, 原始 tokens: {total_raw_tokens:,}")
    rprint(f"  有效 tokens: {total_valid:,} (丢弃 {total_raw_tokens - total_valid})")

    # ---------- 第二步：tokenize 并写入 ----------
    rprint("[cyan] 第二遍：tokenize 并写入...[/cyan]")
    start_time = time.time()

    all_tokens = np.memmap(
        output_dir / "pretrain_tokens.bin",#type:ignore
        dtype=dtype,
        mode='w+',
        shape=(total_valid,)
    )

    write_pos = 0
    for line in open(input_path,'r',encoding='utf-8'):
        text = json.loads(line.strip()).get('text','')
        if not text.strip():
            continue

        ids = tokenizer.encode(text).ids

        # BOS
        if add_bos + write_pos < total_valid:
            all_tokens[write_pos] = bos_id
            write_pos += 1

        # 正文截断
        space = total_valid - write_pos
        chunk = ids[:space]
        all_tokens[write_pos:write_pos+len(chunk)] = chunk
        write_pos += len(chunk)

        # EOS
        if add_eos + write_pos < total_valid:
            all_tokens[write_pos] = eos_id
            write_pos += 1

        if write_pos >= total_valid:
            break

    all_tokens.flush()

    # ====== 保存元信息 ======
    config = {
        "total_tokens": total_valid,
        "num_sequences": total_valid // max_seq_len,
        "seq_len": max_seq_len,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
    }
    with open(output_dir / "pretrain_config.json", 'w') as f:   # type:ignore
        json.dump(config, f, indent=2)

    rprint(f"[green]✅ 完成！序列数: {total_valid // max_seq_len:,}, 文件大小: {(output_dir/'pretrain_tokens.bin').stat().st_size/1024**2:.1f} MB[/green]")