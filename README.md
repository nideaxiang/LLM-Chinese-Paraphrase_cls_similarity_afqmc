# 自己做的小项目，算是第一个纯手撕的项目，好累(；′⌒`)

一个最小可复现的中文**复述句识别**（Paraphrase Identification）微调项目。  
基于 `bert-base-chinese`，在 AFQMC 数据集上训练 3 个 epoch，即可快速判断两句中文是否语义等价。

---

## 1. 任务背景
AFQMC（Ant Financial Question Matching Corpus）要求模型输出 0/1 标签：  
1 → 两句语义等价（复述）；0 → 不等价。

---

## 2. 微调流程
1. 数据  
   - 训练集 8 000 条，验证集 3 000 条。  
   - 仅取前 N 条加速实验，不改变原始分布。

2. 模型  
   - 骨架：`bert-base-chinese`  
   - 结构：CLS 向量 → Dropout → 768×2 线性层 → Softmax。  
   - 参数量 ≈ 102 M，其中可训练 102 M（全量微调）。

3. 训练  
   - batch_size=4，lr=1e-5，线性衰减，warmup=0。  
   - 3 epoch 共约 6 k 步，在单张 RTX-3060 上 30 min 完成。

4. 效果  
   | 验证集准确率 | 最好 checkpoint |
   |-------------|----------------|
   | 67.8%      | `` |

---

## 3. 已知不足
1. 数据面  
   - AFQMC 偏向金融/客服领域，覆盖面窄，模型在口语、文学、医疗等场景容易误判。  
   - 样本长度普遍 ≤ 64，长句复述能力未验证。

2. 训练面  
   - batch=4 太小，梯度噪声大；增大 batch 并同步调 lr 有望提升 1~2 个点。  
   - 未做 K-fold、数据增强（同义改写、对抗样本），鲁棒性不足。  
   - 指标仅用 Accuracy，未考虑 F1、Precision/Recall，阈值默认 0.5。

3. 模型面  
   - 直接取 CLS 向量，未试 ESIM、Siamese、Sentence-BERT 等交互/双塔结构。  
   - 未引入领域继续预训练（DAPT）或任务自适应预训练（TAPT）。  
   - 未做知识蒸馏，模型 340 MB 仍显笨重，边缘设备部署困难。

---

## 4. 快速上手
```bash
pip install -r requirements.txt
python main.py          # 微调
