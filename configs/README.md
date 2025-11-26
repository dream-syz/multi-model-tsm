# 训练配置说明

## 快速开始

### 1. 基础训练（简单平均融合）
```bash
bash configs/train_rtd_resnet50.sh
```

### 2. 自定义融合策略

#### 可学习权重融合（推荐）
```bash
python main.py mmvpr RTD \
    --arch resnet50 --num_segments 8 \
    --fusion_type learned \
    --gd 20 --lr 0.001 --lr_steps 40 60 --epochs 80 \
    --batch-size 32 -j 16 --dropout 0.5 \
    --consensus_type avg --eval-freq 1 \
    --shift --shift_div 8 --shift_place blockres --npb
```

#### 注意力机制融合（最佳）
```bash
python main.py mmvpr RTD \
    --arch resnet50 --num_segments 8 \
    --fusion_type attention \
    --gd 20 --lr 0.001 --lr_steps 40 60 --epochs 80 \
    --batch-size 32 -j 16 --dropout 0.5 \
    --consensus_type avg --eval-freq 1 \
    --shift --shift_div 8 --shift_place blockres --npb \
    --label_smoothing 0.1 --warmup_epochs 5
```

## 融合策略对比

| 策略 | 优点 | 缺点 | 推荐场景 |
|-----|------|------|---------|
| `avg` | 简单，无额外参数 | 性能一般 | Baseline实验 |
| `learned` | 自动学习模态重要性 | 轻微增加参数 | 日常训练（推荐） |
| `attention` | 动态调整模态权重 | 计算量稍大 | 追求最佳性能 |

## 超参数调优建议

### 批大小（Batch Size）
- **V100 32GB**: 64
- **RTX 3090 24GB**: 32-48
- **RTX 2080Ti 11GB**: 16-24
- **显存不足**: 减小batch size或使用梯度累积

### 学习率（Learning Rate）
- **ResNet50**: 0.001（推荐）
- **ResNet101**: 0.0005
- **MobileNetV2**: 0.001

### 学习率衰减策略
- **step**: 在指定epoch衰减（推荐）
- **cos**: 余弦退火

### 正则化
- **Dropout**: 0.5-0.7
- **Weight Decay**: 5e-4
- **Label Smoothing**: 0.1（推荐）
- **Warmup**: 5 epochs（大batch size推荐）

## 训练技巧

### 1. 多模态数据平衡
确保三种模态数据质量一致：
- 检查RGB、IR、Depth数据是否对齐
- 验证数据加载是否正确

### 2. 渐进式训练
```bash
# 第1阶段：冻结backbone，只训练分类头（5 epochs）
python main.py mmvpr RTD --arch resnet50 --epochs 5 --lr 0.01 ...

# 第2阶段：解冻所有层，全局微调
python main.py mmvpr RTD --arch resnet50 --epochs 50 --lr 0.001 --resume checkpoint/xxx.pth ...
```

### 3. 学习率预热
对于大批量训练（batch_size >= 64），建议使用学习率预热：
```bash
--warmup_epochs 5
```

### 4. 标签平滑
减少过拟合，提高泛化能力：
```bash
--label_smoothing 0.1
```

## 性能优化

### 混合精度训练（推荐）
使用FP16训练可以显著加速：
```bash
--amp  # 自动混合精度
```

### 数据加载优化
```bash
-j 16  # 增加数据加载线程数
```

## 实验追踪

### TensorBoard可视化
```bash
tensorboard --logdir=log/
```

### 查看训练日志
```bash
tail -f log/TSM_mmvpr_RTD_resnet50_*/log.csv
```

## 常见问题

### Q: 训练很慢怎么办？
A: 
1. 增加`-j`参数（数据加载线程数）
2. 使用`--amp`启用混合精度训练
3. 减小batch size，使用梯度累积

### Q: 显存不足怎么办？
A:
1. 减小batch size
2. 减少num_segments（如改为4）
3. 使用更小的模型（如MobileNetV2）

### Q: 如何选择最佳融合策略？
A:
1. 先用`avg`跑baseline
2. 再用`learned`，通常能提升1-2%
3. 最后尝试`attention`，可能再提升0.5-1%

### Q: 验证集准确率不提升了怎么办？
A:
1. 检查是否过拟合（训练集准确率远高于验证集）
2. 增加正则化（dropout, label smoothing）
3. 使用数据增强
4. 提前停止训练

## 预期性能

| 模型 | 融合策略 | Segments | Top-1 Acc | Top-5 Acc |
|-----|---------|----------|-----------|-----------|
| ResNet50 | avg | 8 | ~75% | ~92% |
| ResNet50 | learned | 8 | ~77% | ~93% |
| ResNet50 | attention | 8 | ~78% | ~94% |

*注：实际性能取决于数据集质量和训练配置*

