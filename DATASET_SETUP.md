# 数据集准备指南

## 📦 数据集下载

从官方渠道下载 **Multi-Modal Visual Pattern Recognition Challenge - Track 3** 数据集。

填写申请表格获取数据集：
[Multi-Modal Visual Pattern Recognition Challenge Datasets Request Form](https://docs.google.com/forms/d/e/1FAIpQLSeJGZTYW-JS0-IJKnWgYGnE0EgdXnoL7Yi0xc-F9Z6XU1X4Zg/viewform)

---

## 📂 数据集存放位置

### 方式一：使用默认路径（推荐）

**默认数据集根目录**：`/data/Track3/`

请将下载的数据集放置在服务器的 `/data/Track3/` 目录下。

### 方式二：自定义路径

如果你希望使用其他路径，需要修改以下文件：

1. **训练数据配置**：`ops/dataset_config.py` 第3行
   ```python
   ROOT_DATASET = '/data/Track3/'  # 修改为你的路径
   ```

2. **测试数据配置**：`ops/dataset_config_for_pred.py` 第3行
   ```python
   ROOT_DATASET = '/data/Track3/'  # 修改为你的路径
   ```

---

## 📁 完整目录结构

将数据集按照以下结构组织：

```
/data/Track3/
├── training_set/
│   ├── train_videofolder.txt          # 训练集列表文件
│   ├── val_videofolder.txt            # 验证集列表文件
│   ├── rgb_data/                      # RGB数据目录
│   │   ├── 1/                         # 视频ID为1的帧序列
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   ├── 2/                         # 视频ID为2的帧序列
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   └── ...                        # 更多视频
│   ├── ir_data/                       # 热红外数据目录
│   │   ├── 1/
│   │   │   ├── 000001.jpg
│   │   │   ├── 000002.jpg
│   │   │   └── ...
│   │   ├── 2/
│   │   │   └── ...
│   │   └── ...
│   └── depth_data/                    # 深度数据目录
│       ├── 1/
│       │   ├── 000001.png             # 注意：深度图是PNG格式
│       │   ├── 000002.png
│       │   └── ...
│       ├── 2/
│       │   └── ...
│       └── ...
└── test_set/
    ├── test_videofolder.txt           # 测试集列表文件
    ├── rgb_data/                      # 测试集RGB数据
    │   ├── 1/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    ├── ir_data/                       # 测试集热红外数据
    │   ├── 1/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    └── depth_data/                    # 测试集深度数据
        ├── 1/
        │   ├── 000001.png
        │   └── ...
        └── ...
```

---

## 📝 列表文件格式

### train_videofolder.txt / val_videofolder.txt 格式

每行格式：`视频目录名 帧数 类别标签`

示例：
```
1 150 0
2 120 5
3 180 12
...
```

- **第1列**：视频目录名（对应rgb_data/ir_data/depth_data下的子目录）
- **第2列**：该视频的总帧数
- **第3列**：动作类别标签（0-19，共20个类别）

### test_videofolder.txt 格式

每行格式：`视频目录名 帧数`

示例：
```
1 150
2 120
3 180
...
```

**注意**：测试集文件不包含类别标签。

---

## 🎯 数据集详细信息

### 训练集
- **视频数量**：2000个（每种模态）
- **总帧数**：32,000+ 帧（每种模态）
- **分辨率**：
  - RGB: 455×256
  - Thermal IR: 320×256
  - Depth: 640×360
- **时长**：2-13秒
- **格式**：
  - RGB: JPG
  - Thermal IR: JPG
  - Depth: PNG

### 测试集
- **视频数量**：500个（每种模态）
- **总帧数**：8,300+ 帧（每种模态）
- **其他参数同训练集**

### 动作类别（20个）

| 编号 | 类别 | 编号 | 类别 |
|-----|------|-----|------|
| 0 | switch light | 10 | open the umbrella |
| 1 | up the stairs | 11 | orchestra conducting |
| 2 | pack backpack | 12 | rope skipping |
| 3 | ride a bike | 13 | shake hands |
| 4 | turn around | 14 | squat |
| 5 | fold clothes | 15 | swivel |
| 6 | hug somebody | 16 | tie shoes |
| 7 | long jump | 17 | tie hair |
| 8 | move the chair | 18 | twist waist |
| 9 | down the stairs | 19 | wear hat |

---

## ✅ 数据验证检查清单

下载并放置好数据后，请检查：

- [ ] 目录结构是否正确
- [ ] `train_videofolder.txt` 和 `val_videofolder.txt` 文件是否存在
- [ ] `test_videofolder.txt` 文件是否存在
- [ ] RGB数据目录下的图片格式是否为 `.jpg`
- [ ] IR数据目录下的图片格式是否为 `.jpg`
- [ ] Depth数据目录下的图片格式是否为 `.png`
- [ ] 每个视频目录下的帧序列是否完整（从000001开始）
- [ ] 训练集是否有2000个视频
- [ ] 测试集是否有500个视频

---

## 🔧 快速验证脚本

创建一个简单的验证脚本来检查数据集是否正确：

```python
# check_dataset.py
import os

ROOT_DATASET = '/data/Track3/'  # 修改为你的路径

def check_dataset():
    print("=" * 50)
    print("数据集验证检查")
    print("=" * 50)
    
    # 检查目录
    dirs_to_check = [
        'training_set/rgb_data',
        'training_set/ir_data',
        'training_set/depth_data',
        'test_set/rgb_data',
        'test_set/ir_data',
        'test_set/depth_data',
    ]
    
    for dir_path in dirs_to_check:
        full_path = os.path.join(ROOT_DATASET, dir_path)
        if os.path.exists(full_path):
            num_videos = len(os.listdir(full_path))
            print(f"✓ {dir_path}: {num_videos} 个视频目录")
        else:
            print(f"✗ {dir_path}: 不存在！")
    
    # 检查列表文件
    files_to_check = [
        'training_set/train_videofolder.txt',
        'training_set/val_videofolder.txt',
        'test_set/test_videofolder.txt',
    ]
    
    print("\n列表文件检查：")
    for file_path in files_to_check:
        full_path = os.path.join(ROOT_DATASET, file_path)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                lines = f.readlines()
            print(f"✓ {file_path}: {len(lines)} 行")
        else:
            print(f"✗ {file_path}: 不存在！")
    
    print("=" * 50)

if __name__ == '__main__':
    check_dataset()
```

运行验证：
```bash
python check_dataset.py
```

---

## 🚀 开始训练

数据集准备好后，即可开始训练：

```bash
# 使用默认配置训练
python main.py mmvpr RTD \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
     --batch-size 64 -j 16 --dropout 0.5 \
     --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb
```

---

## ❓ 常见问题

### Q1: 我的数据集路径不是 `/data/Track3/`，怎么办？
**A**: 修改 `ops/dataset_config.py` 和 `ops/dataset_config_for_pred.py` 中的 `ROOT_DATASET` 变量。

### Q2: 为什么有些图片是 .jpg，有些是 .png？
**A**: Depth（深度图）使用PNG格式保存以保持精度，RGB和IR使用JPG格式。

### Q3: train_videofolder.txt 和 val_videofolder.txt 有什么区别？
**A**: train_videofolder.txt 用于训练，val_videofolder.txt 用于验证。你需要自己划分，或者使用官方提供的划分。

### Q4: 数据集太大，硬盘空间不够怎么办？
**A**: 可以考虑：
- 只使用单个或两个模态进行实验
- 使用软链接将数据集链接到其他磁盘
- 使用数据集的子集进行快速实验

### Q5: 如何验证数据加载是否正确？
**A**: 可以在训练开始时打印一个batch的数据shape，确认维度正确：
```python
# 在训练循环中添加
print(f"Input shape: {input.shape}")  # 应该是 [batch_size, channels, height, width]
```

---

**更新日期**：2025-11-26  
**文档版本**：v1.0

