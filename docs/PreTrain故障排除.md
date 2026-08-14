####  ValueError: mmap length is greater than file size
`bash core/train_pretrain.sh`出现值错误问题，详情如下：
<details>
<summary>展开</summary>

```text
设备: cuda
TensorBoard: tensorboard --logdir /home/user/data/2025/wen/train_llm/runs
加载数据
Traceback (most recent call last):
  File "/home/user/data/2025/wen/train_llm/core/train_pretrain.py", line 315, in <module>
    train(args)
  File "/home/user/data/2025/wen/train_llm/core/train_pretrain.py", line 182, in train
    dataset = PretrainDataset(bin_path, data_config)
  File "/home/user/data/2025/wen/train_llm/core/train_pretrain.py", line 109, in __init__
    self.data = np.memmap(
  File "/home/user/anaconda3/envs/nlpV2_wen/lib/python3.10/site-packages/numpy/_core/memmap.py", line 289, in __new__
    mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
ValueError: mmap length is greater than file size
```
</details>
具体原因很清晰，mmap的长度和文件大小不匹配，导致无法映射内存。
之前的生成.bin文件时的输出结果如下：
<details>
<summary>展开</summary>

```text
🔤 加载 tokenizer...
 第一遍扫描：统计信息...
  行数: 1,270,238, 原始 tokens: 329,954,848
  有效 tokens: 329,953,280 (丢弃 1568)
 第二遍：tokenize 并写入...
✅ 完成！序列数: 161,110, 文件大小: 629.3 MB
```
</details>

#### 排查步骤
1. 检查 bin 文件大小和config文件中的总token数是否匹配
```bash
ls -la data/processed/pretrain_tokens.bin
>>> output:
-rw-r--r-- 1 root root 659906560  8月 13 12:44 data/processed/pretrain_tokens.bin

cat data/processed/pretrain_config.json
>>> output:
{
  "total_tokens": 329953280,
  "num_sequences": 161110,
  "seq_len": 2048,
  "vocab_size": 6400,
  "dtype": "<class 'numpy.uint16'>"
}
```
预处理脚本在`configjson`中存放的是`"dtype": "<class 'numpy.uint16'>"`，但是在训练脚本中判断的是`dtype = np.uint16 if config["dtype"] == "uint16" else np.uint32`,两者的`"<class 'numpy.uint16'>" 和 "uint16"`字符串是完全不相等的。
- 预处理脚本中：
```python
dtype = np.uint16

config = {
    "dtype": str(dtype),
}
```
    - print(dtype)会输出: <class 'numpy.uint16'>，是一个完整的类表示
    - print(str(dtype))会输出: <class 'numpy.uint16'>，已经是一个长字符串
- 训练脚本中：
```python
# 从 JSON 读回来
config["dtype"]   # 值 = "<class 'numpy.uint16'>"

# 代码比较
config["dtype"] == "uint16"
# "<class 'numpy.uint16'>" == "uint16"
# → False！
```


