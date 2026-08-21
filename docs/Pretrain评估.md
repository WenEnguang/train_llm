# 预训练模型的评估指标
## 1、通用分类和理解任务指标

- 准确率（Accuracy）：分类任务中预测正确的样本比例，适用于情感分析、命名实体识别等 搜狐。
- 精确率（Precision）与召回率（Recall）：精确率衡量预测为正类的样本中实际为正的比例，召回率衡量实际正类中被正确预测的比例 搜狐。
- F1值（F1-Score）：精确率和召回率的调和平均，综合考虑两者
## 2、语言模型任务指标

- 困惑度（Perplexity）：衡量语言模型对测试数据的概率分布拟合程度，值越低表示模型预测越准确。
- BLEU（Bilingual Evaluation Understudy）：评估机器生成文本与参考文本的n-gram重叠度，常用于机器翻译和文本生成。

## 3、其他文本质量和多样性指标

- ROUGE：用于文本摘要评估，衡量生成摘要与参考摘要的重叠度。
- METEOR：结合精确率和召回率的翻译质量指标。
- MTLD（Textual Entropy）：衡量文本的词法多样性 。
- Rewardscore：基于奖励模型的生成质量得分。
- Unieval系列：评估自然性、连贯性和可理解性。

## 本次模型采用困惑度（Perplexity）作为评估指标。
```text
加载 checkpoint → 创建模型 → 加载数据（不 shuffle）→ 逐 batch 算 loss → 汇总求 PPL
```
评估和训练的区别在于：
- 不需要优化器、梯度累计、混合精度
- 不需要shuffle，数据的顺序与否不影响结果·
- 不需要反向传播
- 只计算前向传播，计算loss

## 评估结果
{
  "experiment_name": "pretrain_h768_l8_lr0.0005_bs48_20260814_204950",
  "checkpoint": "train_llm/checkpoints/pretrain_h768_l8_lr0.0005_bs48_20260814_204950_final.pth",
  "avg_loss": 3.177740888595581,
  "perplexity": 23.992490573011157,
  "eval_tokens": 329953280
}

| 指标 | 值 | 评价 |
| :---: | :---: | :---: |
| avg_loss | 3.1777 | 较低，模型在训练集上表现良好，从一开始的8.17下降到现在的3.18 |
| perplexity | 23.9925 | 较低，模型对训练数据的预测能力较强 |
| eval_tokens | 329953280 | 评估使用的总 token 数量，数据量较为充足 |
![[pretrain_loss_avg.png]]
![[sft_loss.png]]