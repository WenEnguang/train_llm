### AttributeError: 'str' object has no attribute 'read'
`python core/preprocess_sft.py`出现属性错误问题，详情如下：
<details>
<summary>展开</summary>

```text
🔤 加载 tokenizer...
 第一遍扫描：统计信息...
统计进度: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 905718/905718 [00:02<00:00, 430341.19it/s]
 第二遍：预分配memmap并tokenize 并写入...
Tokenizing:   0%|                                                                                                                              | 0/905718 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/user/data/2025/wen/train_llm/core/preprocess_sft.py", line 220, in <module>
    preprocess_sft(
  File "/home/user/data/2025/wen/train_llm/core/preprocess_sft.py", line 155, in preprocess_sft
    sample = json.load(line) # type:ignore
  File "/home/user/anaconda3/envs/nlpV2_wen/lib/python3.10/json/__init__.py", line 293, in load
    return loads(fp.read(),
AttributeError: 'str' object has no attribute 'read'
```
</details>
错误原因是`json.load()`函数的参数应该是一个文件对象，而不是一个字符串。`line`是一个字符串，而不是一个文件对象，因此会出现`AttributeError: 'str' object has no attribute 'read'`错误。

### OverflowError: Python integer -100 out of bounds for uint16
结果显示的很明显，是发生了溢出错误，Python整数-100超出了uint16的范围。uint16的取值范围是0到65535，而-100显然不在这个范围内。
`dtype = np.uint16 if vocab_size <= 65535 else np.uint32`因为在SFT中，labels中有-100的值，而uint16无法表示负数，所以会导致溢出错误。解决方法是labels 改用 int16（有符号，范围 -32768~32767，vocab_size=6400 完全够用）；input_ids 因为不含负数，可以用 uint16。
- 此种方式也解决了数据空间的文件，充分利用数据范围，最大化利用数据空间。

### torch.OutOfMemoryError: CUDA out of memory.
很明显，GPU显存不足，无法分配所需的内存。解决方法有以下几种：
1. 减小batch size：可以尝试减小训练时的batch size，以减少每次训练所需的显存。
2. 使用梯度累积（Gradient Accumulation）：通过累积多个小批次的梯度来模拟更大的batch size，从而减少显存占用。
解决方案：通过减少梯度累积的步数减少内存的使用，将micro_batch_size从8改为4。

### KeyError: 'loss'
<details>
<summary>展开</summary>
```text
Traceback (most recent call last):
  File "/home/user/data/2025/wen/train_llm/core/train_sft.py", line 316, in <module>
    train(args)
  File "/home/user/data/2025/wen/train_llm/core/train_sft.py", line 246, in train
    raw_loss = outputs["loss"]
  File "/home/user/anaconda3/envs/nlpV2_wen/lib/python3.10/site-packages/transformers/utils/generic.py", line 445, in __getitem__
    return inner_dict[k]
KeyError: 'loss'
```
</details>
错误原因是`outputs`字典中没有`'loss'`键。接下来去查看一下outputs的内容。

```python
# ── 调试代码开始 ──
print(f"outputs 类型: {type(outputs)}")
print(f"outputs 内容: {outputs}")
# 看看 outputs 是什么类型
if hasattr(outputs, "loss"):
    print(f"outputs.loss: {outputs.loss}")
elif hasattr(outputs, "logits"):
    print(f"outputs.logits shape: {outputs.logits.shape}")
else:
    print(f"outputs 的所有属性: {dir(outputs)}")
# ── 调试代码结束 ──
```
<details>
<summary>展开</summary>
```text
outputs 类型: <class 'transformers.modeling_outputs.MoeCausalLMOutputWithPast'>
outputs 内容: MoeCausalLMOutputWithPast(loss=None, aux_loss=tensor(0., device='cuda:0', dtype=torch.float16), logits=tensor([[[全为nan]]], device='cuda:0', dtype=torch.float16, grad_fn=<UnsafeViewBackward0>), past_key_values=[None, None, None, None, None, None, None, None], hidden_states=tensor([[[全为nan]]], device='cuda:0', dtype=torch.float16), attentions=None, router_logits=None)
```
</details>
根据结果显示；

1. loss全是nan，说明模型没有接收到 labels，所以没有计算损失。
2. logits全是nan，说明模型的输出不正常，模型加载后前向传播异常。
进入模型的源码中进行查看：

```python
def forward(self, input_ids, attention_mask=None, ..., labels=None, **kwargs):
    hidden_states, past_key_values, aux_loss = self.model(input_ids, ...)
    logits = self.lm_head(hidden_states[:, slice_indices, :])
    loss = None
    if labels is not None:
        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
        loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
    return MoeCausalLMOutputWithPast(loss=loss, ...)
```

可以看到在定义的模型的`forward`方法中，模型的第二个参数是`attention_mask`，但是我再调用模型的`forward`方法时，`outputs = model(input_ids,labels)`,就导致我的`labels`被传递到了`attention_mask`的位置，而`labels`是一个tensor，所以模型的前向传播就会出现异常，导致loss和logits全为nan。

- 修复：通过标签值去传递值。`outputs = model(input_ids,labels=labels)`，这样就可以正确的传递labels参数，模型就可以正常计算loss和logits了。

### loss : nan(时间最长和最难排查的bug)
在修复完成上述的任务之后，就可以正常运行了:`bash core/train_sft.sh`,但是发现：

<details>
<summary>展开</summary>
```text
2026-08-15 16:42:30.628 | INFO     | __main__:train:224 - 
Epoch 1/3 | Step 100/226430 | Loss: nan |9.2s                                                                                                        
Epoch 1/3 | Step 200/226430 | Loss: nan |18.2s          
```
</details>
截取部分的训练日志，发现loss为nan。

#### 1、时间线排除
首先这个nan是第一次打印(step:100)时就已经出现了，而不是在训练过程中逐渐发散的。这时候的optimizer.step()大概只走了12次左右（100/accum=8 = 12.5),这时候的warmup才是刚刚起步，lr并不是十分地大，甚至还是接近于零。如果是"学习率太大训练发散"，不可能这么快就砸出 nan。所以我们先把"超参不当导致数值发散"这个假设排除掉，重点转向：数据或 loss 计算本身从结构上就是错的。

#### 2、验证chat_template的渲染结果
```python
from transformers import AutoTokenizer  # type:ignore
from core.config import MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
sample = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好，有什么可以帮你的？"},
]

rendered = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)

print("【渲染出的原始文本，repr形式（能看到不可见字符）】")
print(repr(rendered))
>>>output:
【渲染出的原始文本，repr形式（能看到不可见字符）】
'<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n你好，有什么可以帮你的？<|im_end|>\n'
```
结合之前的`preprocess_sft.py`文件中的数据处理方式:
```python
assistant_start_marker = tokenizer.encode(
        "<bos>assistant\n",add_special_tokens=False
    )
    assistant_end_marker = tokenizer.encode(
        "<eos>\n",add_special_tokens=False
    )
```
可以发现在真实的文本中是不会出现的，这个tokenizer采用的ChatML（<|im_start|> / <|im_end|>）。结果就是：每一条样本的 labels 从头到尾全部是 -100，一个 token 都没被标记为需要计算 loss 的 assistant 回复。
- marker 一次都匹配不上 → 所有样本的 labels 从头到尾全是 -100 → CrossEntropyLoss(ignore_index=-100) 在"整个 batch 都被忽略"的情况下算出 0/0 = nan——不报错，安静地输出 nan，与观察到的现象完全吻合。用抽样脚本实测验证了这个假设（抽样统计 valid_ratio == 0 的样本占比）。
- 顺手排查了另一个可能性：input_ids 有没有越界（vocab_size 配置值和 tokenizer 真实词表是否一致）。这次确认是一致的（6400 == 6400），排除。但这提醒了一件事：原代码只 assert 了 labels.max() < vocab_size，没有对 input_ids 做同样检查——如果两者不一致，input_ids 越界会导致 embedding 查表异步越界，同样表现为"不报错但全 nan"，这类 bug 必须主动去查，不会自己暴露。

但是还有一个问题就是assistant 回复里的 <think>\n\n</think>\n\n 这部分如果不处理，还是会被纳入assistance的回复区间，即模型会被训练去"学习生成一个固定的空think的块儿。"
- 如果 SFT 目标就是训练一个不做推理、直接回答的模型，这个空 think 块被学进去问题不大，顶多是每次回复都固定带一段空模板，稍微浪费一点 token。
- 如果你后续还想做支持推理开关的模型（比如某些轮次带真实思考、某些轮次不带），现在把"空 think"训成固定模式，会让模型学到错误的先验——以后即便传入真实 reasoning_content，模型也可能倾向于把 think 内容压缩/忽略，因为它在训练早期已经学到"assistant 开头就是空 think"这个捷径。

确认 reasoning_content 的真实分布：抽样统计发现 125 万条 assistant 轮次里，24.7% 带真实 reasoning_content，75.3% 不带——这是一个混合数据集，不能用非黑即白的处理方式。
验证 tokenizer 对 reasoning_content 的原生支持：把 reasoning_content 原样放进 message dict（不手动拼接进 content）交给模板渲染，确认模板会自动把内容包进 <think>...</think>；不传则渲染空 think。这说明原代码"手动拼接 content 字段"的做法是错的，应该让 tokenizer 原生处理。
最终方案：_postprocess_prompt 用精确字符串匹配只剔除空的 <think>\n\n</think>\n\n，非空的（带真实思考内容的）think 块因为字符串不完全相等，天然不会被误删，无需额外写"判断 think 是否为空"的逻辑分支。

**教训（最重要的一条）：任何"手写字面字符串去匹配模型/框架内部生成的文本"的写法都是高风险的，因为字面字符串和实际渲染结果是两条独立维护的路径，没有任何机制保证它们同步。凡是能从对象本身的属性（如 tokenizer.bos_token）动态获取的值，就不要手写字面量。**

### dtype空间优化问题
labels/input_ids 全部用 int32 存储，但 labels 里大部分是 -100 占位、input_ids 值域上限是 vocab_size=6400，远远用不满 32 位，直接使用int32进行存储，就会导致数据空间的大量浪费，并且存储的文件也会变大，导致存储空间的浪费。
1. labels 含负数哨兵值，不能用无符号类型：uint16 存 -100 会被 silent wrap 成 65436，导致ignore_index=-100 再也匹配不到，所有位置（包括 pad 和 prompt）都会被当成有效 label 参与 loss
2. dtype 是否够用取决于 tokenizer 真实词表，不是配置文件里写的数字：实测确认 tokenizer.vocab_size == len(tokenizer) == 配置的 vocab_size == 6400，三者一致才能确认 uint16/int16 安全，这一步不能凭"配置文件说是 6400 所以肯定没问题"就跳过。
3. 写入和读取的 dtype 必须严格同步，不能各自硬编码一遍：preprocess_sft.py 里改了写入 dtype，如果 train_sft.py 的 SFTDataset 还硬编码 dtype=np.int32 去读同一份文件，np.memmap 不会报错，只会按错误的字节对齐解析出乱码数据，训练能跑但 loss 是随机噪声，这种问题几乎无法从报错信息定位。解法：把 dtype 写进 meta.json，SFTDataset 从 meta 动态读取，保证读写只有一个真源（single source of truth）。

**涉及二进制序列化格式（.bin、memmap、自定义 dtype）的改动，"写入端"和"读取端"永远要当成一个整体来改，任何一边改了另一边没同步，都不会在改动当下报错，而是在未来某次训练里以"数据看起来正常但训练效果诡异"的形式出现。**

#### 静默错误类是通用排查四步法：
1. **先用时间线/数量级排除缩小范围**：nan 在第几步出现？如果是训练一开始就出现（而非逐渐发散），基本可以排除"超参导致数值发散"这类需要时间累积的假设，转向"结构性错误"（数据本身、loss 计算路径本身）。
2. **用最小可复现样例隔离问题**：不在 90 万条真实数据上调试，而是构造 1-2 条最简单的样例（一问一答），跑通整条链路的每一步中间产物（渲染文本 → token ids → labels），逐层核对。
3. **对"约定"做逐字节验证，不要相信"看起来差不多**"：任何跨模块的"约定"（比如 marker 字符串、特殊 token、字段命名），只要是靠字符串硬编码维护的，一律用 repr() 打印真实值做逐字符核对，不要凭记忆或凭"应该是这样"去写代码。
4. **用抽样统计量化问题规模**：定位到"疑似"问题点后，不要只验证一条样本就下结论，要跑抽样统计（比如 valid_ratio == 0 的样本比例、empty_label 占比）来确认问题是普遍的还是个别的——这个比例本身也决定了修复的紧迫程度和后续要不要加自动化阈值检查。

#### SFT数据预处理的专属checkpoint
- marker/special token是否来自tokenizer.xxx_token 动态获取,手写硬编码容易出现不一致的问题
- 检测apply_chat_template的渲染结果是否符合预期,避免手写字符串匹配
- 涉及 reasoning_content / tool_calls 等可选字段时，是否确认了 tokenizer 模板对这些字段的原生支持方式（该透传就透传，不要手动拼接进 content）？
- 数据集里可选字段的分布是否抽样统计过（比如"多少比例样本有 reasoning"），而不是假设全有或全无？
- labels 里的哨兵值（-100）是否与 dtype 的正负号兼容？
- input_ids 的最大值是否小于所选 dtype 的表示上限，且这个上限是从 tokenizer.vocab_size 实测得出，不是从配置文件抄的？
- 是否对"全部样本都没有匹配到有效 label"这种极端情况加了预处理阶段的自动中断检查（阈值报警），而不是等训练跑起来才用肉眼发现 loss 异常？
- 写入 .bin 用的 dtype，是否和读取端（Dataset 类）通过 meta 文件同步，而不是两边各自硬编码？

