"""
generate_test.py
SFT 训练完成后的定性生成质量检查脚本
（因为本次训练没有验证集 loss、也只保留了 final checkpoint，
  无法用曲线判断"横盘"是收敛还是欠训练，只能用真实生成结果人工判断）

用法：
    python generate_test.py --ckpt /path/to/xxx_final.pth
    python generate_test.py --ckpt /path/to/xxx_final.pth --greedy   # 贪心解码，结果稳定可复现，便于反复对比
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
from transformers import AutoTokenizer  # type:ignore

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from config import MODEL_DIR


# ═══════════════════════════════════════════
# 测试用例：覆盖"训练集里大概率出现过的类型" + "没见过的新问题" + "多轮对话"
# 三类问题分别检查：格式是否学会 / 是否会泛化 / 是否理解上下文
# ═══════════════════════════════════════════

SINGLE_TURN_PROMPTS = [
    # 训练数据里大概率出现过的类型：检查基本格式和记忆
    "你是谁开发的？",
    "你的训练数据来源是什么？",
    # 常规知识问答：检查泛化能力
    "请解释一下什么是光合作用。",
    "1加1等于几？",
    "帮我写一句关于春天的诗。",
    # 边界/新颖问题：检查是否会崩溃、复读、答非所问
    "如果地球没有月亮会怎么样？",
    "用一句话总结三国演义的故事。",
]

MULTI_TURN_TEST = [
    {"role": "user", "content": "我想学做红烧肉，需要准备什么食材？"},
    {"role": "assistant", "content": "红烧肉主要需要五花肉、生抽、老抽、冰糖、料酒、姜片和八角。"},
    {"role": "user", "content": "那大概要炖多久？"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="SFT模型定性生成质量检查")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint .pth 完整路径")
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--greedy", action="store_true",
        help="用贪心解码代替采样，结果确定可复现，适合反复跑同样的prompt做对比"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0,
        help="重复惩罚系数，>1.0时会降低已出现过的token被再次选中的概率，用于抑制复读。常用范围1.1~1.3"
    )
    return parser.parse_args()


def load_model(args, device):
    model = MiniMindForCausalLM(config=MiniMindConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    ))
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)  # type:ignore
    model.eval()
    return model


@torch.no_grad()
def generate_reply(model, tokenizer, messages, args, device) -> str:
    # add_generation_prompt=True：只加 "<|im_start|>assistant\n"，不会插入think占位，
    # 这跟训练时"空think被剔除"的处理是一致的格式，不会产生训练/推理格式不匹配
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=args.repetition_penalty,
    )
    if args.greedy:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)

    output_ids = model.generate(input_ids, **gen_kwargs)  # type:ignore
    new_tokens = output_ids[0][input_ids.shape[1]:]
    # 故意不 skip_special_tokens，方便肉眼检查是否在 <|im_end|> 正确停止
    reply = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return reply


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载 tokenizer: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"加载模型 checkpoint: {args.ckpt}")
    model = load_model(args, device)

    decode_mode = "贪心(greedy)" if args.greedy else f"采样(T={args.temperature}, top_p={args.top_p})"
    print(f"解码方式: {decode_mode} | repetition_penalty={args.repetition_penalty}\n")
    print("=" * 80)

    # ── 单轮测试 ──
    for prompt in SINGLE_TURN_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        reply = generate_reply(model, tokenizer, messages, args, device)
        print(f"【问】{prompt}")
        print(f"【答】{reply}")
        print("-" * 80)

    # ── 多轮对话测试 ──
    reply = generate_reply(model, tokenizer, MULTI_TURN_TEST, args, device)
    print("【多轮对话测试】")
    for msg in MULTI_TURN_TEST:
        print(f"  [{msg['role']}] {msg['content']}")
    print(f"  [assistant，新生成] {reply}")
    print("=" * 80)

    print("\n【人工检查清单，逐条对照上面的生成结果打勾】")
    print("1. 回复是否在 <|im_end|> 处正确停止？（如果 max_new_tokens 用满还没停，说明ChatML格式没学到位）")
    print("2. 有没有出现逐词/逐句重复的复读机现象？（小模型常见问题，出现说明还需要继续训练或调整解码参数）")
    print("3. 多轮对话里，第二轮回复是否体现出对第一轮上下文的理解（这里期望提到'红烧肉'或'炖'相关内容）？")
    print("4. 训练数据里大概率出现过的问题（如'你是谁开发的'），回答是否和SFT数据风格高度一致？")
    print("5. 没见过的新问题，回复是否语句通顺、内容基本合理（哪怕简短或不够精确，但不能是乱码/无关内容）？")


if __name__ == "__main__":
    main()