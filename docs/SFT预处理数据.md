# SFT预处理数据
SFT（监督微调）和Pretrain之间的核心区别就是数据格式和损失计算方式。
## SFT和Pretrain的核心区别
| 维度 | Pretrain | SFT |
| :--- | --- | --- |
| 数据格式 | 纯文本 | 指令-响应对 |
|目标 | 学习语言本身，计算语言模型的概率分布 | 学习指令的执行能力 |
| Loss 计算 | 所有的token都会参与 | 只对assistant的输出计算loss，prompt部分不计算loss |
| 数据格式 | packing连续流 | 每条对话独立，pad到固定长度 |

## SFT的数据格式
```text
{"conversations": [
    {"role": "user", "content": "给我写一首关于春天的诗"},
    {"role": "assistant", "content": "春风吹绿江南岸，\n万物复苏展新颜。"}
]}
{"conversations": [
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1等于2。"}
]}
```
## SFT数据预处理的关键：Loss Mask
是SFT和PreTrain的核心区别。因为预训练是需要让模型学习到语言的通用能力，所以所有的token都会参与loss计算。而SFT是让模型学习指令的执行能力，所以只对assistant的输出计算loss，prompt部分不计算loss。
```text
# PreTrain
input:  [今天, 天气, 真好, 春, 风, ...]
labels: [今天, 天气, 真好, 春, 风, ...]   ← 全部参与 loss

# SFT
对话渲染后的 prompt：
<bos>user\n给我写一首关于春天的诗<eos>\n<bos>assistant\n春风吹绿江南岸，\n万物复苏展新颜。<eos>\n

labels：
[-100, -100, ..., -100,   春, 风, 吹, 绿, ..., <eos>, -100]
  ↑ user 部分不计算 loss    ↑ assistant 部分计算 loss    ↑ pad 不计算
```
## SFT数据预处理的关键流程
### 1、Chat Template渲染
将结构化的对话渲染成模型训练时看到的原始文本格式：
```python
conversations = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
]

prompt_text = tokenizer.apply_chat_template(conversations, tokenize=False)
# 输出: "<bos>user\n你好<eos>\n<bos>assistant\n你好！<eos>\n"
```
### 2、Loss Mask生成
根据渲染后的文本，生成对应的loss mask：模型只在assistant的输出部分计算loss。
```python
prompt_ids = [BOS, user, 你, 好, EOS, BOS, assistant, 你, 好, ！, EOS]
labels     = [-100, -100, -100, -100, -100, -100, 你, 好, ！, EOS, -100]
              ↑ user 部分全部 -100          ↑ assistant 部分保留原 token
```
PreTrain阶段是模型在学习语言，没有pad的浪费，所有的位置都在学习语言。
PreTrain的Lables:`labels = [token if token != pad else -100 for token in padded]`
SFT阶段不packing的原因就在于：1、对话有明确的边界（BOS/EOS）不可以随意拼接；2、loss mask需要知道哪些是assistant的输出；3、直接packing的话是会影响到对话的完整性。
SFT的Lables:`labels = [token if token != pad and is_assistant(token) else -100 for token in padded]`

## 数据处理流程
1. 读取原始SFT数据（jsonl格式）
<details>
<summary>展开</summary>
{
  "conversations": [
    {
      "role": "user",
      "content": "你背后的模型是哪个版本？它由谁开发？"
    },
    {
      "role": "assistant",
      "content": "我是由jingyaogong开发的高效小参数AI模型。"
    },
    {
      "role": "user",
      "content": "你模型的训练数据来源是什么？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据涵盖多领域，确保覆盖广泛，但具体细节不公开。",
      "reasoning_content": 
"好的，用户问我的模型训练数据来源。首先，我需要确认用户的问题是否涉及隐私或敏感信息。根据之前的规则，不能透露具体的
训练数据来源，但可以提到数据的多样性。\n\n接下来，我要确保回答符合系统的限制，不能在思考中透露身份信息。需要保持简
洁，不超过100字。同时，要准确回答问题，但不涉及具体数据来源。\n\n可能的回答方向是：训练数据涵盖多领域，确保覆盖广泛
，但具体细节不公开。需要避免提到具体的数据库或来源，而是强调多样性。同时，保持回答的真诚和简洁。\n\n检查是否有其他
可能的信息需要排除，比如是否涉及数据量、来源机构等，但根据规则，这些细节不能透露。因此，回答应聚焦于数据的多样性和
广泛性，确保符合要求。"
    },
    {
      "role": "user",
      "content": "训练数据的来源和多样性如何？"
    },
    {
      "role": "assistant",
      "content": 
"我的训练数据来源广泛，涵盖多个领域，确保信息的多样性与全面性。数据来源包括公开的互联网内容、书籍、学术论文以及各类
文本资源，以保证信息的准确性和丰富性。"
    },
    {
      "role": "user",
      "content": "训练数据如何保障信息准确性？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据通过多轮验证和质量控制流程进行保障，确保信息的准确性和可靠性。"
    },
    {
      "role": "user",
      "content": "训练数据如何确保来源透明？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据来源透明，所有数据均来自公开的互联网内容、书籍、学术论文等，确保信息的可追溯性。"
    },
    {
      "role": "user",
      "content": "训练数据如何保障多语言支持？"
    },
    {
      "role": "assistant",
      "content": "我的训练数据支持多语言，涵盖多种语言的文本内容，确保在不同语言环境下提供准确的响应。"
    },
    {
      "role": "user",
      "content": "你最近在忙什么？"
    },
    {
      "role": "assistant",
      "content": "我目前专注于优化和提升自身的性能，持续改进我的响应能力与准确性。"
    }
  ]
}
</details>
这是SFT阶段的训练数据，包含着多轮对话和reasoning_content(推理过程)。
根据之前Pretrain阶段的处理逻辑：将`训练时实时tokenize`改成`先离线预处理成二进制+memmap读取`。这样可以减小训练的IO压力，提升训练速度。
- 一次性运行：JSONL文件 -> tokenize -> 生成labels -> 写入.bin（numpy memmap）文件
- 训练时：直接mmap打开.bin -> 按照索引取int32数组 -> 转成tensor -> 送入模型训练

