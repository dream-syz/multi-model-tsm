# 🚀 快速开始指南

## 📦 数据集放置位置

### 默认路径（推荐）
将下载的数据集解压到：
```
/data/Track3/
```

### 完整目录结构
```
/data/Track3/
├── training_set/
│   ├── train_videofolder.txt
│   ├── val_videofolder.txt
│   ├── rgb_data/
│   │   ├── 1/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   ├── 2/
│   │   └── ... (共2000个视频)
│   ├── ir_data/
│   │   ├── 1/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── ... (共2000个视频)
│   └── depth_data/
│       ├── 1/
│       │   ├── 000001.png  # 注意是PNG格式
│       │   └── ...
│       └── ... (共2000个视频)
└── test_set/
    ├── test_videofolder.txt
    ├── rgb_data/
    ├── ir_data/
    └── depth_data/
```

### 自定义路径
如果你要使用其他路径，修改以下文件：
```bash
# 编辑这两个文件的第3行
ops/dataset_config.py
ops/dataset_config_for_pred.py

# 修改为：
ROOT_DATASET = '/your/custom/path/'
```

---

## ⚙️ 环境准备

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 验证数据集
下载数据后，运行验证脚本：
```bash
python check_dataset.py
```

输出示例：
```
==================================================
✓ training_set/rgb_data          找到 2000 个项目
✓ training_set/ir_data           找到 2000 个项目  
✓ training_set/depth_data        找到 2000 个项目
✓ test_set/rgb_data              找到 500 个项目
...
✅ 所有检查通过！数据集配置正确。
==================================================
```

---

## 🎯 开始训练

### 方案1：一键训练（推荐）
```bash
# 使用专业配置的训练脚本
bash configs/train_rtd_resnet50.sh
```

### 方案2：自定义训练

#### Baseline（简单平均融合）
```bash
python main.py mmvpr RTD \
    --arch resnet50 --num_segments 8 \
    --fusion_type avg \
    --gd 20 --lr 0.001 --lr_steps 40 60 --epochs 50 \
    --batch-size 32 -j 16 --dropout 0.5 \
    --consensus_type avg --eval-freq 1 \
    --shift --shift_div 8 --shift_place blockres --npb
```

#### 进阶（可学习权重融合）⭐ 推荐
```bash
python main.py mmvpr RTD \
    --arch resnet50 --num_segments 8 \
    --fusion_type learned \
    --gd 20 --lr 0.001 --lr_steps 40 60 --epochs 80 \
    --batch-size 32 -j 16 --dropout 0.5 \
    --label_smoothing 0.1 --warmup_epochs 5 \
    --consensus_type avg --eval-freq 1 \
    --shift --shift_div 8 --shift_place blockres --npb
```

#### 最佳（注意力融合）⭐⭐
```bash
python main.py mmvpr RTD \
    --arch resnet50 --num_segments 8 \
    --fusion_type attention \
    --gd 20 --lr 0.001 --lr_steps 50 75 --epochs 100 \
    --batch-size 32 -j 16 --dropout 0.5 \
    --label_smoothing 0.1 --warmup_epochs 5 --amp \
    --consensus_type avg --eval-freq 1 \
    --shift --shift_div 8 --shift_place blockres --npb
```

---

## 📊 监控训练

### TensorBoard可视化
```bash
tensorboard --logdir=log/ --port=6006
```
然后访问：http://localhost:6006

### 查看训练日志
```bash
# 实时查看
tail -f log/TSM_mmvpr_RTD_*/log.csv

# 查看最佳准确率
grep "Best Prec" log/TSM_mmvpr_RTD_*/log.csv
```

---

## 🎯 生成提交文件

训练完成后，使用最佳模型生成预测：

```bash
python generate_submission.py mmvpr \
    --weights checkpoint/TSM_mmvpr_RTD_resnet50_*/ckpt.best.pth.tar \
    --test_segments 8 \
    --batch_size 1 \
    --test_crops 1 \
    --csv_file submission.csv
```

生成的`submission.csv`即可提交到CodaLab。

---

## 🔧 根据GPU调整配置

| GPU型号 | 显存 | 推荐batch_size | 预计训练时间 |
|---------|------|---------------|-------------|
| V100 32GB | 32GB | 64 | 6小时 |
| A100 40GB | 40GB | 80 | 5小时 |
| RTX 3090 | 24GB | 32-48 | 8小时 |
| RTX 2080Ti | 11GB | 16-24 | 12小时 |
| RTX 3060 | 12GB | 16-20 | 14小时 |

**显存不足？** 尝试：
- 减小`--batch-size`
- 减少`--num_segments`（8→4）
- 添加`--amp`启用混合精度

---

## 📈 预期性能

| 配置 | Top-1 准确率 | 训练时间 | 备注 |
|-----|-------------|---------|------|
| avg fusion | ~75% | 8h | Baseline |
| learned fusion | ~77% | 8h | **推荐** |
| attention fusion | ~78-80% | 9h | 最佳 |

---

## 🆘 遇到问题？

### 数据相关
- **问题**: 提示找不到数据集
- **解决**: 
  1. 运行`python check_dataset.py`检查
  2. 确认`ROOT_DATASET`路径设置正确

### 训练相关
- **问题**: 显存溢出 (CUDA out of memory)
- **解决**: 减小batch_size或使用--amp

- **问题**: 准确率不提升
- **解决**: 
  1. 检查学习率是否过大/过小
  2. 确认数据是否正确加载
  3. 尝试增加label_smoothing

### 性能相关
- **问题**: 训练太慢
- **解决**:
  1. 增加`-j`参数（数据加载线程）
  2. 使用`--amp`混合精度训练
  3. 检查GPU利用率（nvidia-smi）

---

## 📚 更多文档

- **数据集详细说明**: `DATASET_SETUP.md`
- **Bug修复记录**: `BUGFIXES.md`
- **优化方案详解**: `OPTIMIZATIONS.md`
- **配置说明**: `configs/README.md`

---

## ✅ 完整工作流程

```bash
# 1. 准备数据集
# 下载并解压到 /data/Track3/

# 2. 验证数据
python check_dataset.py

# 3. 安装依赖
pip install -r requirements.txt

# 4. 开始训练（选择一种）
bash configs/train_rtd_resnet50.sh  # 推荐
# 或手动指定参数
python main.py mmvpr RTD --arch resnet50 --fusion_type learned ...

# 5. 监控训练
tensorboard --logdir=log/

# 6. 生成提交文件
python generate_submission.py mmvpr --weights checkpoint/xxx.pth ...

# 7. 提交到CodaLab
# 上传 submission.csv
```

---

**祝训练顺利！如有问题请参考详细文档或联系技术支持。** 🎉

