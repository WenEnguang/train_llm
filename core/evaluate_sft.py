"""
evaluate_sft_generation.py
SFT 评估：用固定 prompt 测试模型对话能力(人工审核)
"""

import json
import torch
from transformers import AutoTokenizer  # type:ignore
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig


def load_model(ckpt_path, device):
    model = MiniMindForCausalLM(config=MiniMindConfig(
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
    ))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, prompt, device, max_new_tokens=100):
    # 构造对话格式
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取 assistant 回复部分
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("./model_hub/minimind")
    
    test_prompts = [
        "给我写一首关于春天的诗",
        "1+1等于几？",
        "你背后的模型是哪个版本？",
        "什么是机器学习？",
    ]
    
    # 对比 pretrain 和 sft 的生成效果
    models = {
        "pretrain": "./checkpoints/pretrain_xxx_final.pth",
        "sft": "./checkpoints/sft_xxx_final.pth",
    }
    
    for model_name, ckpt_path in models.items():
        model = load_model(ckpt_path, device)
        print(f"\n{'='*50}")
        print(f"模型: {model_name}")
        print(f"{'='*50}")
        
        for prompt in test_prompts:
            response = generate(model, tokenizer, prompt, device)
            print(f"\n用户: {prompt}")
            print(f"回复: {response}")