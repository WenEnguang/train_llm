"""
preprocess_sft.py
SFT 数据预处理：对话 JSONL → tokenize + loss mask → .bin
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer  # type:ignore
from tqdm import tqdm
from rich import print as rprint

from config import RAW_DATA_DIR,PROCESSED_DATA_DIR,MODEL_DIR

def make_sft_labels(
    prompt_ids: list,
    max_seq_len: int,
    assistant_start_marker: list,
    assistant_end_marker: list,
    pad_id: int,
) -> list:
    """
    生成 SFT 的 labels：只有 assistant 回复参与 loss 计算
    
    Args:
        prompt_ids: 完整 prompt 的 token ids
        max_seq_len: 最大长度
        assistant_start_marker: assistant 起始标记的 token ids
        assistant_end_marker: assistant 结束标记的 token ids
        pad_id: pad token id
    
    Returns:
        labels 列表，长度 = max_seq_len
    """
    # 初始全部 -100（默认不计算 loss）
    labels = [-100] * max_seq_len
    # 滑动窗口：查找所有 assistant 回复区间
    i = 0
    n = len(prompt_ids)
    start_len = len(assistant_start_marker)
    end_len = len(assistant_end_marker)

    while i < n:
        # 匹配 assistant 起始标记
        if prompt_ids[i:i+start_len] == assistant_start_marker:
            content_start = i + start_len
            
            # 向后查找结束标记
            j = content_start
            while j < n:
                if prompt_ids[j:j+end_len] == assistant_end_marker:
                    break
                j += 1
            
            # assistant 回复区间：[content_start, j+end_len)
            content_end = j + end_len if j < n else n
            
            # 把这个区间的 labels 设为原 token id
            for k in range(content_start, min(content_end, max_seq_len)):
                labels[k] = prompt_ids[k]
            
            # 跳到这个区间的后面继续找
            i = content_end
        else:
            i += 1
    
    return labels

def preprocess_sft(
    input_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 2048,
    keep_reasoning: bool = False,
):
    """
    将 SFT 对话数据转换为 .bin 文件
    
    Args:
        input_path: 原始 jsonl 文件路径
        output_dir: 输出目录
        tokenizer_path: tokenizer 目录
        max_seq_len: 每条对话的最大长度
        keep_reasoning: 是否保留 reasoning_content（默认丢弃）
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
    dtype = np.int32

    # 获取 assistant 起止标记（用于定位 assistant 回复区间）,这个标记取决于tokenizer的chat_template
    assistant_start_marker = tokenizer.encode(
        "<bos>assistant\n",add_special_tokens=False
    )
    assistant_end_marker = tokenizer.encode(
        "<eos>\n",add_special_tokens=False
    )

    # ----------- 统计行数 -----------------
    rprint("[cyan] 第一遍扫描：统计信息...[/cyan]")
    t0 = time.time()
    
    num_lines = 0

    total_file_lines = sum(1 for line in open(input_path, 'r', encoding='utf-8'))   # 先统计总行数（用于进度条）

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f,total=total_file_lines,desc='统计进度'):
            if not line.strip():
                continue
            num_lines += 1
    
    # ------------- 预分配memmap并tokenize写入------------------
    rprint("[cyan] 第二遍：预分配memmap并tokenize 并写入...[/cyan]")
    start_time = time.time()
    input_ids_mmap = np.memmap(
        str(output_dir / "sft_input_ids.bin"),  # type:ignore
        dtype=dtype,
        mode='w+',
        shape=(num_lines, max_seq_len),
    )
    labels_mmap = np.memmap(
        str(output_dir / "sft_labels.bin"), # type:ignore
        dtype=dtype,
        mode='w+',
        shape=(num_lines, max_seq_len),
    )

    skipped = 0  # 跳过的超长样本数
    with open(input_path, 'r', encoding='utf-8') as f:
        for row_idx, line in enumerate(tqdm(f, total=num_lines, desc="Tokenizing")):
            line = line.strip()
            if not line:
                # 空行：填入空数据
                input_ids_mmap[row_idx] = np.zeros(max_seq_len,dtype=dtype)
                labels_mmap[row_idx] = np.full(max_seq_len,-100,dtype=dtype)
                skipped += 1
                continue
            # 解析JSON
            sample = json.loads(line) # type:ignore
            conversations = sample["conversations"]
            # 清理reasoning_content 
            cleaned=[]
            for msg in conversations:
                new_msg = {
                    'role':msg['role'],
                    'content':msg['content'],
                }
                # 如果保留 reasoning，拼接到 content 前面
                if keep_reasoning and msg.get("reasoning_content"):
                    new_msg["content"] = msg["reasoning_content"] + "\n\n" + msg["content"]

                cleaned.append(new_msg)
            # 渲染chat template
            prompt_text = tokenizer.apply_chat_template(
                cleaned,
                tokenize=False,
                add_generation_prompt=False,
            )
            # tokenize 整段 prompt
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # 超长处理：跳过
            if len(prompt_ids) > max_seq_len:
                input_ids_mmap[row_idx] = np.zeros(max_seq_len, dtype=dtype)
                labels_mmap[row_idx] = np.full(max_seq_len, -100, dtype=dtype)
                skipped += 1
                continue
            # pad 到 max_seq_len
            padded = prompt_ids + [pad_id] * (max_seq_len - len(prompt_ids))
            # 生成loss mask
            labels = make_sft_labels(
                prompt_ids=prompt_ids,
                max_seq_len=max_seq_len,
                assistant_start_marker=assistant_start_marker,
                assistant_end_marker=assistant_end_marker,
                pad_id=pad_id,
            )
            # 写入memap
            input_ids_mmap[row_idx] = np.array(padded, dtype=dtype)
            labels_mmap[row_idx] = np.array(labels, dtype=dtype)

    # 保存元信息
    input_ids_mmap.flush()
    labels_mmap.flush()

    meta = {
        "stage": "sft",
        "count": num_lines,
        "seq_len": max_seq_len,
        "dtype": dtype.__name__,
        "skipped_long_samples": skipped,
        "files": {
            "input_ids": "sft_input_ids.bin",
            "labels": "sft_labels.bin",
        },
    }
    with open(output_dir / "sft_meta.json", 'w', encoding='utf-8') as f:    # type:ignore
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    rprint(f"[green]SFT阶段数据预处理完成！对话数：{num_lines}，跳过的超长文本:{skipped}，序列长度:{max_seq_len},耗时{elapsed:.2f}")

if __name__ == "__main__":
    preprocess_sft(
        input_path=os.path.join(RAW_DATA_DIR,'sft_t2t_mini.jsonl'),
        output_dir=str(PROCESSED_DATA_DIR),
        tokenizer_path=str(MODEL_DIR),
    )
