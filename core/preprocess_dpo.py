"""
preprocess_dpo.py
DPO 数据预处理：chosen/rejected 对话 → tokenize + next-token shift + loss mask → .bin
 
复用 SFT 阶段验证过的经验：
  - marker 从 tokenizer.bos_token/eos_token 动态生成，不手写字符串（坑5的教训）
  - 空 think 块用精确字符串匹配剔除，非空think天然保留
  - 预处理阶段就做防御性检查（全零mask占比超阈值直接中断），不等训练时才发现（坑5的教训）
  - dtype按实际取值范围选型：mask只有0/1两个值，用uint8（比SFT的labels还能再省空间）
"""
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
import json
import time
import numpy as np
from transformers import AutoTokenizer  # type:ignore
from tqdm import tqdm
from rich import print as rprint
 
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR


# ═══════════════════════════════════════════
# 1. message 归一化 + chat_template渲染（和SFT阶段完全一致的逻辑，直接复用）
# ═══════════════════════════════════════════
 
def _normalize_message(message: dict) -> dict:
    normalized = {
        "role": message["role"],
        "content": message["content"],
    }
    # DPO数据目前没观察到reasoning_content字段，但保留这个判断，
    # 万一以后换了带reasoning的DPO数据集，不需要再改这里
    if message.get("reasoning_content"):
        normalized["reasoning_content"] = message["reasoning_content"]
    return normalized
 
 
def _postprocess_prompt(prompt: str) -> str:
    """只精确剔除空think块，非空think（真实reasoning）原样保留"""
    return prompt.replace("<think>\n\n</think>\n\n", "")
 
 
def _render_prompt(tokenizer, conversations: list) -> str:
    messages = [_normalize_message(m) for m in conversations]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return _postprocess_prompt(prompt)
 
 
def get_assistant_markers(tokenizer):
    bos_ids = tokenizer(
        f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
    ).input_ids
    eos_ids = tokenizer(
        f"{tokenizer.eos_token}\n", add_special_tokens=False
    ).input_ids
    return bos_ids, eos_ids

# ═══════════════════════════════════════════
# 2. loss mask生成：在token id序列上匹配，输出0/1掩码（不是-100，这是和SFT最大的结构差异）
# ═══════════════════════════════════════════

def make_dpo_mask(
    input_ids:list,
    max_seq_len:int,
    assistant_start_marker: list,
    assistant_end_marker: list,
) -> list:
    mask = [0] * len(input_ids)
    start_len = len(assistant_start_marker)
    end_len = len(assistant_end_marker)
    i = 0
    n = len(input_ids)

    while i < n:
        if input_ids[i:i+start_len] == assistant_start_marker:
            content_start = i + start_len
            j = content_start
            while j < n:
                if input_ids[j:j + end_len] == assistant_end_marker:
                    break
                j += 1
            # 找不到结束标记（大概率是被截断），mask一直标到序列末尾，
            # 不因为没找到<|im_end|>就放弃这部分训练信号
            content_end = j + end_len if j < n else n
            for k in range(content_start, min(content_end, max_seq_len)):
                mask[k] = 1
            i = content_end
        else:
            i += 1
    return mask[:max_seq_len]

# ═══════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════
 
# mask只有0/1两个值，uint8足够（比SFT的labels dtype=int16更省，因为不需要负数哨兵值）
IDS_DTYPE = np.uint16     # x/y 序列，无负数，vocab_size=6400远小于65536
MASK_DTYPE = np.uint8     # mask 只有0/1

def preprocess_dpo(
    input_path: str,
    output_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 768,
):
    output_dir = Path(output_dir)   # type:ignore
    output_dir.mkdir(parents=True, exist_ok=True)   # type:ignore

    rprint("[cyan]🔤 加载 tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    assert vocab_size <= np.iinfo(IDS_DTYPE).max, (
        f"tokenizer 词表大小 {vocab_size} 超出 {IDS_DTYPE} 的表示范围，请改用更大的 dtype"
    )

    bos_ids, eos_ids = get_assistant_markers(tokenizer)
    rprint(f"[dim]assistant_start token ids: {bos_ids}[/dim]")
    rprint(f"[dim]assistant_end   token ids: {eos_ids}[/dim]")
    rprint(f"[dim]ids dtype: {IDS_DTYPE.__name__}, mask dtype: {MASK_DTYPE.__name__}[/dim]")

    rprint("[cyan]第一遍扫描：统计信息...[/cyan]")
    total_file_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    num_lines = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_file_lines, desc="统计进度"):
            if line.strip():
                num_lines += 1

    # 存储的是shift之后的x/y，长度比max_seq_len少1（next-token prediction的标准做法）
    stored_len = max_seq_len - 1

    rprint("[cyan]第二遍：tokenize 并写入 memmap...[/cyan]")
    start_time = time.time()
 
    shape = (num_lines, stored_len)
    x_chosen_mm = np.memmap(output_dir / "dpo_x_chosen.bin", dtype=IDS_DTYPE, mode="w+", shape=shape)   # type:ignore
    y_chosen_mm = np.memmap(output_dir / "dpo_y_chosen.bin", dtype=IDS_DTYPE, mode="w+", shape=shape)   # type:ignore
    mask_chosen_mm = np.memmap(output_dir / "dpo_mask_chosen.bin", dtype=MASK_DTYPE, mode="w+", shape=shape)    # type:ignore
    x_rejected_mm = np.memmap(output_dir / "dpo_x_rejected.bin", dtype=IDS_DTYPE, mode="w+", shape=shape)   # type:ignore
    y_rejected_mm = np.memmap(output_dir / "dpo_y_rejected.bin", dtype=IDS_DTYPE, mode="w+", shape=shape)   # type:ignore
    mask_rejected_mm = np.memmap(output_dir / "dpo_mask_rejected.bin", dtype=MASK_DTYPE, mode="w+", shape=shape)    # type:ignore

    empty_mask_count = 0  # chosen或rejected任一出现全零mask就计数一次

    def process_one_side(conversations: list) -> tuple:
        prompt_text = _render_prompt(tokenizer, conversations)
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = input_ids[:max_seq_len]  # 截断，不跳过（DPO希望尽量保留数据量）
        input_ids = input_ids + [pad_id] * (max_seq_len - len(input_ids))
 
        mask_full = make_dpo_mask(input_ids, max_seq_len, bos_ids, eos_ids)
 
        # next-token shift：x是[:-1]，y是[1:]，mask同步右移对齐
        x = input_ids[:-1]
        y = input_ids[1:]
        mask = mask_full[1:]
        return x, y, mask

    with open(input_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(tqdm(f, total=num_lines, desc="Tokenizing")):
            line = line.strip()
            if not line:
                x_chosen_mm[row_idx] = np.full(stored_len, pad_id, dtype=IDS_DTYPE)
                y_chosen_mm[row_idx] = np.full(stored_len, pad_id, dtype=IDS_DTYPE)
                mask_chosen_mm[row_idx] = np.zeros(stored_len, dtype=MASK_DTYPE)
                x_rejected_mm[row_idx] = np.full(stored_len, pad_id, dtype=IDS_DTYPE)
                y_rejected_mm[row_idx] = np.full(stored_len, pad_id, dtype=IDS_DTYPE)
                mask_rejected_mm[row_idx] = np.zeros(stored_len, dtype=MASK_DTYPE)
                continue
 
            sample = json.loads(line)
 
            x_c, y_c, mask_c = process_one_side(sample["chosen"])
            x_r, y_r, mask_r = process_one_side(sample["rejected"])
 
            if sum(mask_c) == 0 or sum(mask_r) == 0:
                empty_mask_count += 1
 
            x_chosen_mm[row_idx] = np.array(x_c, dtype=IDS_DTYPE)
            y_chosen_mm[row_idx] = np.array(y_c, dtype=IDS_DTYPE)
            mask_chosen_mm[row_idx] = np.array(mask_c, dtype=MASK_DTYPE)
            x_rejected_mm[row_idx] = np.array(x_r, dtype=IDS_DTYPE)
            y_rejected_mm[row_idx] = np.array(y_r, dtype=IDS_DTYPE)
            mask_rejected_mm[row_idx] = np.array(mask_r, dtype=MASK_DTYPE)
 
    for mm in (x_chosen_mm, y_chosen_mm, mask_chosen_mm, x_rejected_mm, y_rejected_mm, mask_rejected_mm):
        mm.flush()
 
    # ✅ 关键防御：全零mask占比超过阈值，直接中断，不让坏数据流入训练
    empty_ratio = empty_mask_count / max(1, num_lines)
    if empty_ratio > 0.05:
        raise RuntimeError(
            f"❌ 严重异常：{empty_mask_count} 条样本（{empty_ratio:.1%}）的 chosen 或 rejected "
            f"没有匹配到任何 assistant 区间，mask 全为 0。这通常意味着 marker 生成逻辑或数据格式转换"
            f"存在问题（比如之前convert_and_merge_dpo.py转换出的角色映射是否正确），请先排查，不要继续训练。"
        )
 
    meta = {
        "stage": "dpo",
        "count": num_lines,
        "seq_len": stored_len,  # 实际存储的序列长度（已经shift过）
        "max_seq_len_before_shift": max_seq_len,
        "ids_dtype": IDS_DTYPE.__name__,
        "mask_dtype": MASK_DTYPE.__name__,
        "empty_mask_samples": empty_mask_count,
        "files": {
            "x_chosen": "dpo_x_chosen.bin",
            "y_chosen": "dpo_y_chosen.bin",
            "mask_chosen": "dpo_mask_chosen.bin",
            "x_rejected": "dpo_x_rejected.bin",
            "y_rejected": "dpo_y_rejected.bin",
            "mask_rejected": "dpo_mask_rejected.bin",
        },
    }
    with open(output_dir / "dpo_meta.json", "w", encoding="utf-8") as f:    # type:ignore
        json.dump(meta, f, indent=2, ensure_ascii=False)
 
    elapsed = time.time() - start_time
    rprint(
        f"[green]DPO阶段数据预处理完成！样本数：{num_lines}，"
        f"全零mask异常样本：{empty_mask_count}，"
        f"seq_len（shift后）:{stored_len}，耗时{elapsed:.2f}s[/green]"
    )

if __name__ == "__main__":
    preprocess_dpo(
        input_path=str(RAW_DATA_DIR / "dpo.jsonl"),
        output_dir=str(PROCESSED_DATA_DIR),
        tokenizer_path=str(MODEL_DIR),
        max_seq_len=768,
    )