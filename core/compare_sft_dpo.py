"""
compare_sft_dpo.py
对比 SFT checkpoint 和 DPO checkpoint 在相同 prompt 下的生成结果，
重点覆盖之前生成质量诊断中记录的具体失败案例：
  - 答非所问退化成身份模板（三国演义、地球没有月亮）
  - 多轮对话上下文理解（红烧肉炖多久）
  - 孤立</think>+内容重复的畸形结构
  用来验证"DPO能否曲线救国"这个假设是否成立。
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


# 之前诊断中明确记录过失败/异常的具体案例，优先复现这些
TEST_CASES = [
    {"label": "身份类问题（预期两版都正常）", "prompt": "你是谁开发的？"},
    {"label": "答非所问案例1（上次SFT退化成身份模板）", "prompt": "用一句话总结三国演义的故事。"},
    {"label": "答非所问案例2（上次SFT退化成身份模板）", "prompt": "如果地球没有月亮会怎么样？"},
    {"label": "知识类问题（上次SFT胡编乱造）", "prompt": "请解释一下什么是光合作用。"},
]

# ✅ 新增：有明确对错标准的事实类探针，专门用来抓"自信胡说"这个副作用
# 这类问题不看"有没有正面回答"，只看"回答本身对不对"——
# 上一轮实验教训：RewardAcc数字上涨，不代表回答质量真的变好，
# 可能只是模型从"敷衍"变成"更自信地说错话"，必须用有标准答案的问题单独核查
FACT_CHECK_CASES = [
    {"label": "事实核查1（地球是行星，不是恒星）", "prompt": "地球是行星还是恒星？"},
    {"label": "事实核查2（光合作用是合成不是分解）", "prompt": "光合作用是把二氧化碳和水合成有机物，还是把有机物分解？"},
    {"label": "事实核查3（简单算术，检验有没有开始瞎编数字）", "prompt": "3加5等于几？"},
    {"label": "事实核查4（不该出现真实企业关联）", "prompt": "你和阿里云、百度这些公司有关系吗？"},
]

MULTI_TURN_CASE = {
    "label": "多轮上下文理解（上次SFT只是复制上一轮回答）",
    "history": [
        {"role": "user", "content": "我想学做红烧肉，需要准备什么食材？"},
        {"role": "assistant", "content": "红烧肉主要需要五花肉、生抽、老抽、冰糖、料酒、姜片和八角。"},
        {"role": "user", "content": "那大概要炖多久？"},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="对比SFT和DPO checkpoint的生成质量")
    parser.add_argument("--sft_ckpt", type=str, required=True)
    parser.add_argument("--dpo_ckpt", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42,
                        help="固定随机种子，确保两个模型在相同条件下对比，差异只来自模型本身")
    return parser.parse_args()


def load_model(ckpt_path, args, device):
    model = MiniMindForCausalLM(config=MiniMindConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
    ))
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)  # type:ignore
    model.eval()
    return model


@torch.no_grad()
def generate_reply(model, tokenizer, messages, args, device) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    # 每次生成前重置随机种子，保证SFT和DPO在完全相同的随机条件下生成，
    # 观察到的差异只能来自模型权重本身，不是采样随机性造成的
    torch.manual_seed(args.seed)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def print_comparison(label, prompt_desc, sft_reply, dpo_reply):
    print("=" * 90)
    print(f"【{label}】")
    print(f"问题: {prompt_desc}")
    print("-" * 90)
    print(f"[SFT] {sft_reply}")
    print("-" * 90)
    print(f"[DPO] {dpo_reply}")
    print("=" * 90)
    print()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载 tokenizer: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"加载 SFT checkpoint: {args.sft_ckpt}")
    sft_model = load_model(args.sft_ckpt, args, device)

    print(f"加载 DPO checkpoint: {args.dpo_ckpt}")
    dpo_model = load_model(args.dpo_ckpt, args, device)

    print(f"解码方式: 采样(T={args.temperature}, top_p={args.top_p}, "
          f"repetition_penalty={args.repetition_penalty})，固定seed={args.seed}\n")

    # 单轮对比
    for case in TEST_CASES:
        messages = [{"role": "user", "content": case["prompt"]}]
        sft_reply = generate_reply(sft_model, tokenizer, messages, args, device)
        dpo_reply = generate_reply(dpo_model, tokenizer, messages, args, device)
        print_comparison(case["label"], case["prompt"], sft_reply, dpo_reply)

    # ✅ 事实核查专项：这部分结果需要你逐条肉眼判断对错，而不是看"有没有回答"
    print("\n" + "#" * 90)
    print("# 以下是事实核查专项，请逐条判断内容对错（不是看有没有正面回答）")
    print("#" * 90 + "\n")
    for case in FACT_CHECK_CASES:
        messages = [{"role": "user", "content": case["prompt"]}]
        sft_reply = generate_reply(sft_model, tokenizer, messages, args, device)
        dpo_reply = generate_reply(dpo_model, tokenizer, messages, args, device)
        print_comparison(case["label"], case["prompt"], sft_reply, dpo_reply)

    # 多轮对比
    history_desc = " -> ".join(m["content"][:15] for m in MULTI_TURN_CASE["history"])
    sft_reply = generate_reply(sft_model, tokenizer, MULTI_TURN_CASE["history"], args, device)
    dpo_reply = generate_reply(dpo_model, tokenizer, MULTI_TURN_CASE["history"], args, device)
    print_comparison(MULTI_TURN_CASE["label"], history_desc, sft_reply, dpo_reply)

    print("\n【人工核对清单】")
    print("1. 答非所问案例1、2：DPO版本是否不再退化成身份模板，而是尝试正面回答？")
    print("2. 多轮上下文：DPO版本是否不再是原样复制上一轮回答，而是针对'炖多久'给出新内容？")
    print("3. 两个版本里，有没有再出现'内容说一遍→孤立</think>→内容重复一遍'这个畸形结构？")
    print("4. 知识类问题：DPO是否有改善（这个理论上DPO帮不上，如果没变化是预期内的）？")
    print("5. 身份类问题：DPO是否依然保持正常（这类不该被DPO训坏）？")
    print("6. ✅事实核查1-4：DPO版本有没有说出明确错误的内容（地球说成恒星、光合作用说反、")
    print("   算术算错、编造和真实公司的关联）？这几条只要出现一条明确错误，")
    print("   就说明这次超参数把模型推向了'更自信但更容易胡说'的方向，需要往回调，")
    print("   不能只看RewardAcc数字涨了就认为是进步。")


if __name__ == "__main__":
    main()