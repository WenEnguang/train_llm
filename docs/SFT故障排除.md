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
`dtype = np.uint16 if vocab_size <= 65535 else np.uint32`因为在SFT中，labels中有-100的值，而uint16无法表示负数，所以会导致溢出错误。解决方法是将dtype改为np.int16或者np.int32，以便能够表示负数。

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

### loss:nan
在修复完成上述的任务之后，就可以正常运行了:`bash core/train_sft.sh`,但是发现：

<details>
<summary>展开</summary>
```text
2026-08-15 16:42:30.600 | INFO     | __main__:train:201 - 已加载模型
2026-08-15 16:42:30.600 | INFO     | __main__:train:204 - 模型的参数量:63912192
2026-08-15 16:42:30.628 | INFO     | __main__:train:220 - 
开始 SFT 训练: sft_h768_l8_lr2e-05_bs32_20260815_164230
2026-08-15 16:42:30.628 | INFO     | __main__:train:221 -   Epochs: 3
2026-08-15 16:42:30.628 | INFO     | __main__:train:222 -   有效批次: 32 (micro=4 × accum=8)
2026-08-15 16:42:30.628 | INFO     | __main__:train:223 -   学习率: 2e-05 (warmup 2547/84912 步)
2026-08-15 16:42:30.628 | INFO     | __main__:train:224 - ==========================================
Epoch 1/3 | Step 100/226430 | Loss: nan | 9.2s                                                                                                        
Epoch 1/3 | Step 200/226430 | Loss: nan | 18.2s          
```
</details>
截取部分的训练日志，发现loss为nan。

首先这个nan是第一次打印(step:100)时就已经出现了，而不是在训练过程中逐渐发散的。这时候的optimizer.step()大概只走了12次左右（100/accum=8 = 12.5),这时候的warmup才是刚刚起步，lr并不是十分地大，甚至还是接近于零。如果是"学习率太大训练发散"，不可能这么快就砸出 nan。所以我们先把"超参不当导致数值发散"这个假设排除掉，重点转向：数据或 loss 计算本身从结构上就是错的。




