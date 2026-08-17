"""
diagnose_think_duplication.py
诊断"生成时出现孤立</think>+内容重复一遍"这个畸形结构的根因

分三步排查：
  A. 统计原始数据里 content 和 reasoning_content 的文本重合度
     （验证"数据本身就有重复模式"这个假设）
  B. 用受控样例测试 chat_template 渲染本身会不会引入重复
     （验证"模板bug"这个假设）
  C. 抽样解码已经写入 .bin 的真实训练目标，看训练目标本身有没有重复
     （验证"预处理代码引入重复"这个假设，这一步最直接）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import difflib
import numpy as np
from transformers import AutoTokenizer  # type:ignore

from core.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR


def text_overlap_ratio(a: str, b: str) -> float:
    """用最长公共子序列比例衡量两段文本的相似度，0=完全不同，1=完全相同"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ═══════════════════════════════════════════
# A. 统计原始数据：content 和 reasoning_content 是否本身就高度重复
# ═══════════════════════════════════════════

def check_data_duplication(input_path: Path, sample_limit: int = 20000):
    print("=" * 80)
    print("【A. 检查原始数据：content 和 reasoning_content 的文本重合度】")
    print("=" * 80)

    ratios = []
    high_overlap_examples = []  # 收集几个重合度高的真实例子，方便肉眼看
    checked = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if checked >= sample_limit:
                break
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            for msg in sample["conversations"]:
                if msg.get("role") != "assistant":
                    continue
                reasoning = msg.get("reasoning_content")
                content = msg.get("content", "")
                if not reasoning:
                    continue
                ratio = text_overlap_ratio(reasoning, content)
                ratios.append(ratio)
                if ratio > 0.5 and len(high_overlap_examples) < 5:
                    high_overlap_examples.append((ratio, reasoning[:150], content[:150]))
                checked += 1

    if not ratios:
        print("没有找到任何带 reasoning_content 的样本，无法统计。")
        return

    ratios = np.array(ratios)
    print(f"共检查 {len(ratios)} 条带 reasoning_content 的 assistant 回复")
    print(f"重合度 均值/P50/P90/P99: {ratios.mean():.3f} / "
          f"{np.percentile(ratios, 50):.3f} / {np.percentile(ratios, 90):.3f} / "
          f"{np.percentile(ratios, 99):.3f}")
    print(f"重合度 > 0.5 的样本占比: {(ratios > 0.5).mean():.1%}")
    print(f"重合度 > 0.8 的样本占比: {(ratios > 0.8).mean():.1%}")

    if high_overlap_examples:
        print("\n重合度较高的真实样本举例：")
        for ratio, r, c in high_overlap_examples:
            print(f"  [重合度 {ratio:.2f}]")
            print(f"    reasoning_content: {r}...")
            print(f"    content:           {c}...")
            print()
    else:
        print("\n没有发现重合度明显偏高的样本 —— 说明假设A（数据本身重复）不成立，"
              "问题大概率出在模板渲染或预处理环节，重点看下面 B、C 两步。")


# ═══════════════════════════════════════════
# B. 受控测试：chat_template 渲染本身会不会引入重复
# ═══════════════════════════════════════════

def check_template_duplication(tokenizer):
    print("\n" + "=" * 80)
    print("【B. 受控测试：用完全不同的 reasoning_content 和 content，看模板渲染是否重复】")
    print("=" * 80)

    control_sample = [
        {"role": "user", "content": "这是一个测试问题ABC"},
        {
            "role": "assistant",
            "content": "这是最终答案XYZ，和思考内容完全不一样，用来判断模板是否会重复渲染",
            "reasoning_content": "这是思考过程123，故意写得和content完全不同",
        },
    ]
    rendered = tokenizer.apply_chat_template(
        control_sample, tokenize=False, add_generation_prompt=False
    )
    print("渲染结果（repr形式，能看到不可见字符）：")
    print(repr(rendered))

    # 自动判断：content 的关键片段（"XYZ"）在渲染结果里出现了几次
    occur_count = rendered.count("XYZ")
    reasoning_occur_count = rendered.count("123")
    print(f"\ncontent 关键片段'XYZ'在渲染结果中出现次数: {occur_count}"
          f"{'  ⚠️ 异常，应该只出现1次，模板可能有重复渲染的bug！' if occur_count > 1 else '  ✅ 正常'}")
    print(f"reasoning_content 关键片段'123'在渲染结果中出现次数: {reasoning_occur_count}"
          f"{'  ⚠️ 异常，应该只出现1次！' if reasoning_occur_count > 1 else '  ✅ 正常'}")


# ═══════════════════════════════════════════
# C. 抽样解码 .bin 里真实的训练目标，看写入的数据本身有没有重复
# ═══════════════════════════════════════════

def check_bin_duplication(tokenizer, processed_dir: Path, sample_count: int = 30):
    print("\n" + "=" * 80)
    print("【C. 抽样解码 .bin 训练目标，检查预处理阶段写入的数据本身是否重复】")
    print("=" * 80)

    meta_path = processed_dir / "sft_meta.json"
    if not meta_path.exists():
        print(f"找不到 {meta_path}，跳过这一步。")
        return
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    dtype_map = {"int16": np.int16, "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32}
    input_dtype = dtype_map[meta["input_ids_dtype"]]

    input_ids_mm = np.memmap(
        processed_dir / meta["files"]["input_ids"], dtype=input_dtype,
        mode="r", shape=(meta["count"], meta["seq_len"]),
    )

    # 找一些包含 "<think>" 且思考内容非空的样本（用token id搜索think开标签更准确，
    # 这里图简单先解码再用文本搜索，样本量不大够用）
    rng = np.random.default_rng(42)
    candidate_idx = rng.choice(meta["count"], min(3000, meta["count"]), replace=False)

    found = 0
    for idx in candidate_idx:
        if found >= sample_count:
            break
        row = input_ids_mm[idx].astype(np.int64)
        text = tokenizer.decode(row, skip_special_tokens=False)
        if "<think>\n\n</think>" in text:
            continue  # 跳过空think的（我们只关心有真实reasoning内容的）
        if "<think>" not in text:
            continue
        found += 1

        # 提取 think 内容和 think 之后的最终回答，检查是否有明显重复片段
        try:
            think_part = text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            after_think = text.split("</think>", 1)[1]
        except IndexError:
            continue

        overlap = text_overlap_ratio(think_part[:200], after_think[:200])
        if overlap > 0.4:
            print(f"[行 {idx}] think内容与之后正文重合度: {overlap:.2f}  ⚠️ 偏高")
            print(f"  think内容前200字: {think_part[:200]}")
            print(f"  think后正文前200字: {after_think[:200]}")
            print()

    print(f"共检查了 {found} 条含真实reasoning的训练样本的实际写入内容")
    print("（如果上面没有打印任何'重合度偏高'的记录，说明 .bin 里写入的训练目标本身没有重复问题，")
    print(" 那就能排除假设C，问题更可能出在模型生成时对'两种assistant格式混合训练'的行为选择上，")
    print(" 而不是数据/预处理层面的bug —— 这种情况下，短期内比较难通过改预处理代码解决，")
    print(" 需要考虑增加更多样的多轮/无reasoning数据配比，或者接受这是当前数据配比下的正常代价）")


def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    input_path = RAW_DATA_DIR / "sft_t2t_mini.jsonl"
    processed_dir = Path(PROCESSED_DATA_DIR)

    check_data_duplication(input_path)
    check_template_duplication(tokenizer)
    check_bin_duplication(tokenizer, processed_dir)


if __name__ == "__main__":
    main()