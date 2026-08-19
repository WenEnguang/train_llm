"""
check_dpo_data.py
DPO数据集诊断：
  1. 语言构成占比（中文 vs 英文 vs 混合），判断"英文数据"是个例还是普遍现象
  2. 按语言分组统计token长度分布，看词表mismatch对英文样本的膨胀程度
  3. 顺手统计一下pretrain/sft语料的语言构成，作为对照
     （如果pretrain阶段中文占绝对多数，DPO里大量英文数据就是真实的能力mismatch风险）
"""

import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
import numpy as np
from transformers import AutoTokenizer  # type:ignore
 
from core.config import RAW_DATA_DIR, MODEL_DIR

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")

def classify_language(text: str) -> str:
    """粗粒度语言分类：统计中文字符数 vs 英文字母数的比例"""
    cjk_count = len(CJK_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))
    total = cjk_count + latin_count
    if total == 0:
        return "其他/无法判断"
    cjk_ratio = cjk_count / total
    if cjk_ratio > 0.7:
        return "中文为主"
    elif cjk_ratio < 0.3:
        return "英文为主"
    else:
        return "中英混合"

def analyze_dpo_language_and_length(dpo_path: Path, tokenizer, sample_limit: int = 5000):
    print("=" * 80)
    print("【1. DPO数据语言构成 + 按语言分组的token长度分布】")
    print("=" * 80)
 
    total_lines = sum(1 for line in open(dpo_path, "r", encoding="utf-8") if line.strip())
    print(f"DPO数据集总行数: {total_lines}\n")
 
    lang_counter = {"中文为主": 0, "英文为主": 0, "中英混合": 0, "其他/无法判断": 0}
    lengths_by_lang = {"中文为主": [], "英文为主": [], "中英混合": []}
    checked = 0
 
    with open(dpo_path, "r", encoding="utf-8") as f:
        for line in f:
            if checked >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
 
            # 用chosen的user prompt判断整条样本的语言（prompt在chosen/rejected间共享）
            user_text = ""
            for msg in sample["chosen"]:
                if msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break
 
            lang = classify_language(user_text)
            lang_counter[lang] += 1
 
            if lang in lengths_by_lang:
                chosen_text = tokenizer.apply_chat_template(
                    sample["chosen"], tokenize=False, add_generation_prompt=False
                )
                length = len(tokenizer.encode(chosen_text, add_special_tokens=False))
                lengths_by_lang[lang].append(length)
 
            checked += 1
 
    print(f"抽样检查: {checked} 条\n")
    print("语言构成占比:")
    for lang, count in lang_counter.items():
        print(f"  {lang}: {count} 条 ({count/checked:.1%})")
 
    print("\n按语言分组的 token 长度分布（这里能看出词表对英文的膨胀程度）:")
    for lang, lengths in lengths_by_lang.items():
        if not lengths:
            continue
        lengths = np.array(lengths)
        print(f"  [{lang}] 均值/P50/P90/P99/max: "
              f"{lengths.mean():.0f} / {np.percentile(lengths,50):.0f} / "
              f"{np.percentile(lengths,90):.0f} / {np.percentile(lengths,99):.0f} / {lengths.max()}")
 
    # 整体（不分语言）的长度分布，用于最终定max_seq_len
    all_lengths = [l for lst in lengths_by_lang.values() for l in lst]
    if all_lengths:
        all_lengths = np.array(all_lengths)
        print(f"\n  [全部] 均值/P50/P90/P99/max: "
              f"{all_lengths.mean():.0f} / {np.percentile(all_lengths,50):.0f} / "
              f"{np.percentile(all_lengths,90):.0f} / {np.percentile(all_lengths,99):.0f} / {all_lengths.max()}")

def analyze_reference_corpus_language(path: Path, text_field: str, label: str, sample_limit: int = 5000):
    """对照组：看pretrain或sft语料的语言构成，判断模型底子里的语言能力分布"""
    if not path.exists():
        print(f"\n（跳过对照：找不到 {path}）")
        return

    print(f"\n{'=' * 80}")
    print(f"【2. 对照组：{label} 语料的语言构成】")
    print("=" * 80)
 
    lang_counter = {"中文为主": 0, "英文为主": 0, "中英混合": 0, "其他/无法判断": 0}
    checked = 0
 
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if checked >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
 
            if text_field == "text":
                text = sample.get("text", "")
            else:
                # sft格式：取第一个user消息
                text = ""
                for msg in sample.get("conversations", []):
                    if msg.get("role") == "user":
                        text = msg.get("content", "")
                        break
 
            lang = classify_language(text)
            lang_counter[lang] += 1
            checked += 1
 
    print(f"抽样检查: {checked} 条")
    for lang, count in lang_counter.items():
        if checked > 0:
            print(f"  {lang}: {count} 条 ({count/checked:.1%})")
 
 
def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
 
    dpo_path = RAW_DATA_DIR / "dpo.jsonl"  # 换成你的真实DPO文件名
    if not dpo_path.exists():
        print(f"找不到 {dpo_path}，请修改脚本里的文件名后重跑")
        return
 
    analyze_dpo_language_and_length(dpo_path, tokenizer)
 
    # 对照组：你的pretrain/sft语料主要是什么语言（判断模型的语言能力底子）
    pretrain_path = RAW_DATA_DIR / "pretrain_t2t_mini.jsonl"  # 换成你的真实pretrain文件名
    analyze_reference_corpus_language(pretrain_path, text_field="text", label="Pretrain")
 
    sft_path = RAW_DATA_DIR / "sft_t2t_mini.jsonl"
    analyze_reference_corpus_language(sft_path, text_field="conversations", label="SFT")


if __name__ == "__main__":
    main()