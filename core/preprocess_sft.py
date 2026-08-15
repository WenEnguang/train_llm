"""
preprocess_sft.py
SFT 数据预处理：对话 JSONL → tokenize + loss mask → .bin
（修复：marker 动态生成 + reasoning_content 原生透传 + 空think剔除）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
import numpy as np
from transformers import AutoTokenizer  # type:ignore
from tqdm import tqdm
from rich import print as rprint

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR

# ═══════════════════════════════════════════
# 1. message 归一化：reasoning_content 原样透传，None 时删除该key
# ═══════════════════════════════════════════

def _normalize_message(message:dict) -> dict:
    normalized = {
        "role": message["role"],
        "content": message["content"],
    }
    if message.get('reasoning_content'):
        normalized["reasoning_content"] = message["reasoning_content"]
    return normalized

def _postprocess_prompt(prompt: str) -> str:
    """只精确剔除空think块，非空think（真实reasoning）原样保留"""
    return prompt.replace("<think>\n\n</think>\n\n", "")

def _render_sft_prompt(tokenizer, conversations: list) -> str:
    messages = [_normalize_message(m) for m in conversations]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return _postprocess_prompt(prompt)

# ═══════════════════════════════════════════
# 2. marker 动态生成：不再手写 <bos>/<eos> 字符串
# ═══════════════════════════════════════════

def get_assistant_markers(tokenizer):
    bos_ids = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    ).input_ids
    eos_ids = tokenizer(
        f"{tokenizer.eos_token}\n", add_special_tokens=False
    ).input_ids
    return bos_ids, eos_ids

def make_sft_labels(
    input_ids: list,
    max_seq_len: int,
    assistant_start_marker: list,
    assistant_end_marker: list,
) -> list:
    """在 token id 序列上做匹配（而不是原来在字符串prompt_ids上匹配文本）"""
    labels = [-100] * len(input_ids)
    start_len = len(assistant_start_marker)
    end_len = len(assistant_end_marker)
    i=0
    n=len(input_ids)

    while i < n:
        if input_ids[i:i + start_len] == assistant_start_marker:
            content_start = i + start_len
            j = content_start
            while j < n:
                if input_ids[j:j + end_len] == assistant_end_marker:
                    break
                j += 1
            content_end = j + end_len if j < n else n
            for k in range(content_start, min(content_end, max_seq_len)):
                labels[k] = input_ids[k]
            i = content_end
        else:
            i += 1
    return labels[:max_seq_len]

# ═══════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════
INPUT_IDS_DTYPE = np.uint16   # 无负数，0~65535 足够覆盖 vocab_size=6400
LABELS_DTYPE = np.int16       # 含 -100 哨兵值，必须有符号

def preprocess_sft(
    input_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
):
    """
    将 SFT 对话数据转换为 .bin 文件
    
    Args:
        input_path: 原始 jsonl 文件路径
        output_dir: 输出目录
        tokenizer_path: tokenizer 目录
        max_seq_len: 每条对话的最大长度
    """
    output_dir = Path(output_dir)   # type:ignore
    if output_dir.exists(): #type:ignore
        pass
    else:
        output_dir.mkdir(parents=True,exist_ok=True)    # type:ignore

    # ---------- 加载 tokenizer ----------
    rprint("[cyan]🔤 加载 tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    assert vocab_size <= np.iinfo(INPUT_IDS_DTYPE).max, (
        f"tokenizer 词表大小 {vocab_size} 超出 {INPUT_IDS_DTYPE} 的表示范围 "
        f"{np.iinfo(INPUT_IDS_DTYPE).max}，请改用更大的 dtype（如 np.int32）"
    )

    # 获取 assistant 起止标记（用于定位 assistant 回复区间）
    bos_ids, eos_ids = get_assistant_markers(tokenizer)
    rprint(f"[dim]assistant_start token ids: {bos_ids}[/dim]")
    rprint(f"[dim]assistant_end   token ids: {eos_ids}[/dim]")
    rprint(f"[dim]input_ids dtype: {INPUT_IDS_DTYPE.__name__}, labels dtype: {LABELS_DTYPE.__name__}[/dim]")

    # ----------- 统计行数 -----------------
    rprint("[cyan]第一遍扫描：统计信息...[/cyan]")
    total_file_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    num_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_file_lines, desc="统计进度"):
            if line.strip():
                num_lines += 1
    
    # ------------- 预分配memmap并tokenize写入------------------
    rprint("[cyan]第二遍：tokenize 并写入 memmap...[/cyan]")
    start_time = time.time()
    input_ids_mmap = np.memmap(
        str(output_dir / "sft_input_ids.bin"), dtype=INPUT_IDS_DTYPE, mode="w+",  # type:ignore
        shape=(num_lines, max_seq_len),
    )
    labels_mmap = np.memmap(
        str(output_dir / "sft_labels.bin"), dtype=LABELS_DTYPE, mode="w+",     # type:ignore
        shape=(num_lines, max_seq_len),
    )

    skipped = 0  # 跳过的超长样本数
    empty_label_count = 0  # ✅ 新增：统计全-100样本数，做防御性检查

    with open(input_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(tqdm(f, total=num_lines, desc="Tokenizing")):
            line = line.strip()
            if not line:
                input_ids_mmap[row_idx] = np.full(max_seq_len, pad_id, dtype=INPUT_IDS_DTYPE)
                labels_mmap[row_idx] = np.full(max_seq_len, -100, dtype=LABELS_DTYPE)
                skipped += 1
                continue

            # 解析JSON
            sample = json.loads(line)
            prompt_text = _render_sft_prompt(tokenizer, sample["conversations"])
            input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            if len(input_ids) > max_seq_len:
                input_ids_mmap[row_idx] = np.full(max_seq_len, pad_id, dtype=INPUT_IDS_DTYPE)
                labels_mmap[row_idx] = np.full(max_seq_len, -100, dtype=LABELS_DTYPE)
                skipped += 1
                continue

            padded = input_ids + [pad_id] * (max_seq_len - len(input_ids))
            labels = make_sft_labels(
                input_ids=padded,
                max_seq_len=max_seq_len,
                assistant_start_marker=bos_ids,
                assistant_end_marker=eos_ids,
            )

            # ✅ 防御性检查：这条样本是否一个有效label都没匹配到
            if all(l == -100 for l in labels):
                empty_label_count += 1

            input_ids_mmap[row_idx] = np.array(padded, dtype=INPUT_IDS_DTYPE)
            labels_mmap[row_idx] = np.array(labels, dtype=LABELS_DTYPE)

    input_ids_mmap.flush()
    labels_mmap.flush()

    # ✅ 关键防御：如果全-100样本占比异常高，直接中断，不让坏数据流入训练
    empty_ratio = empty_label_count / max(1, num_lines - skipped)
    if empty_ratio > 0.05:
        raise RuntimeError(
            f"❌ 严重异常：{empty_label_count} 条样本（{empty_ratio:.1%}）没有匹配到任何 "
            f"assistant 区间，labels 全为 -100。这通常意味着 marker 生成逻辑或 chat_template "
            f"存在不匹配，请先排查，不要继续训练。"
        )

    meta = {
        "stage": "sft",
        "count": num_lines,
        "seq_len": max_seq_len,
        "input_ids_dtype": INPUT_IDS_DTYPE.__name__,   # ✅ 分开记录两个dtype
        "labels_dtype": LABELS_DTYPE.__name__,
        "skipped_long_samples": skipped,
        "empty_label_samples": empty_label_count,
        "files": {
            "input_ids": "sft_input_ids.bin",
            "labels": "sft_labels.bin",
        },
    }

    with open(output_dir / "sft_meta.json", "w", encoding="utf-8") as f:    # type:ignore
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    rprint(
        f"[green]SFT阶段数据预处理完成！对话数：{num_lines}，"
        f"跳过超长：{skipped}，全空label：{empty_label_count}，"
        f"序列长度:{max_seq_len}，耗时{elapsed:.2f}s[/green]"
    )

            

if __name__ == "__main__":
    preprocess_sft(
        input_path=os.path.join(RAW_DATA_DIR,'sft_t2t_mini.jsonl'),
        output_dir=str(PROCESSED_DATA_DIR),
        tokenizer_path=str(MODEL_DIR),
    )
