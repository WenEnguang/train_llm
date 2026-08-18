"""
convert_and_merge_dpo.py
把 dpo_zh.json（from/value字段，chosen/rejected为独立对象）
转换成和英文 dpo_en.jsonl 一致的结构（role/content字段，chosen/rejected为完整对话数组），
然后合并两份数据并随机打散，输出最终用于预处理的中英混合数据集。
"""

import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
import json
import random
 
from core.config import RAW_DATA_DIR

"""
convert_and_merge_dpo.py
把 dpo_zh.json（from/value字段，chosen/rejected为独立对象）
转换成和英文 dpo.jsonl 一致的结构（role/content字段，chosen/rejected为完整对话数组），
然后合并两份数据并随机打散，输出最终用于预处理的中英混合数据集。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random

from core.config import RAW_DATA_DIR

FROM_TO_ROLE = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}

RANDOM_SEED = 42


def load_zh_data(path: Path) -> list:
    """dpo_zh.json 可能是整体一个JSON数组，也可能是jsonl逐行格式，两种都兼容"""
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        return json.loads(raw)
    # 按jsonl逐行解析
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def convert_zh_sample(sample: dict) -> dict | None:
    """
    把单条 dpo_zh 样本转换成 {"chosen": [...], "rejected": [...]} 结构，
    chosen/rejected 各自是完整对话数组（历史轮次 + 最后一轮assistant回复）
    """
    conversations = sample.get("conversations")
    chosen_turn = sample.get("chosen")
    rejected_turn = sample.get("rejected")

    if not conversations or not chosen_turn or not rejected_turn:
        return None

    # 转换历史轮次（human/system -> user/system）
    base_turns = []
    for turn in conversations:
        role = FROM_TO_ROLE.get(turn.get("from"))
        if role is None:
            return None  # 出现未知角色，直接跳过这条，不做猜测性映射
        base_turns.append({"role": role, "content": turn.get("value", "")})

    chosen_role = FROM_TO_ROLE.get(chosen_turn.get("from"), "assistant")
    rejected_role = FROM_TO_ROLE.get(rejected_turn.get("from"), "assistant")

    chosen_full = base_turns + [{"role": chosen_role, "content": chosen_turn.get("value", "")}]
    rejected_full = base_turns + [{"role": rejected_role, "content": rejected_turn.get("value", "")}]

    return {"chosen": chosen_full, "rejected": rejected_full}


def load_en_data(path: Path) -> list:
    """英文dpo.jsonl，逐行jsonl格式，结构已经是{"chosen":[...], "rejected":[...]}"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    zh_path = RAW_DATA_DIR / "dpo_zh.json"
    en_path = RAW_DATA_DIR / "dpo_en.jsonl"
    output_path = RAW_DATA_DIR / "dpo.jsonl"

    print(f"读取中文数据: {zh_path}")
    zh_raw = load_zh_data(zh_path)
    print(f"  原始条数: {len(zh_raw)}")

    zh_converted = []
    zh_skipped = 0
    for sample in zh_raw:
        converted = convert_zh_sample(sample)
        if converted is None:
            zh_skipped += 1
            continue
        zh_converted.append(converted)
    print(f"  转换成功: {len(zh_converted)}，跳过（字段缺失/未知角色）: {zh_skipped}")

    print(f"\n读取英文数据: {en_path}")
    en_data = load_en_data(en_path)
    print(f"  条数: {len(en_data)}")

    merged = zh_converted + en_data
    print(f"\n合并后总条数: {len(merged)}（中文 {len(zh_converted)} + 英文 {len(en_data)}）")

    # 随机打散，避免训练时先吃一整段单一语言
    random.seed(RANDOM_SEED)
    random.shuffle(merged)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n已写入: {output_path}")
    print(f"打散使用固定随机种子 seed={RANDOM_SEED}，如需复现相同顺序可保持不变")

    # 抽查前3条，确认转换后的结构正确
    print("\n抽查转换后的前3条数据结构（确认chosen/rejected都是完整对话数组）：")
    for item in merged[100:105]:
        print(json.dumps(item, ensure_ascii=False, indent=2)[:500])
        print("...\n")


if __name__ == "__main__":
    main()