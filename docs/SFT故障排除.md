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