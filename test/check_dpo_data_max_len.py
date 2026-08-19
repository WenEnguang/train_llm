import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import json
import numpy as np
from transformers import AutoTokenizer  # type:ignore
from core.config import PROCESSED_DATA_DIR, MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
meta = json.load(open(PROCESSED_DATA_DIR / "dpo_meta.json"))

mask_chosen = np.memmap(PROCESSED_DATA_DIR / meta["files"]["mask_chosen"], dtype=np.uint8,
                         mode="r", shape=(meta["count"], meta["seq_len"]))
x_chosen = np.memmap(PROCESSED_DATA_DIR / meta["files"]["x_chosen"], dtype=np.uint16,
                      mode="r", shape=(meta["count"], meta["seq_len"]))

empty_idx = np.where(mask_chosen.sum(axis=1) == 0)[0][:5]  # 抽5条看看
for idx in empty_idx:
    text = tokenizer.decode(x_chosen[idx].astype(np.int64), skip_special_tokens=False)
    print(f"[行 {idx}] 长度768被截断后的内容（后200字）:")
    print(text[-200:])
    print("-" * 60)