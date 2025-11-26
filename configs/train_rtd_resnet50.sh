#!/bin/bash
# 多模态动作识别训练脚本 - 专业配置
# 基于多模态融合最佳实践

# ==================== 基础配置 ====================
DATASET="mmvpr"
MODALITY="RTD"
ARCH="resnet50"
NUM_SEGMENTS=8

# ==================== 多模态融合配置 ====================
# 融合策略: 
#   - avg: 简单平均融合 (baseline)
#   - learned: 可学习权重融合 (推荐)
#   - attention: 注意力机制融合 (最佳，但计算量稍大)
FUSION_TYPE="learned"

# ==================== 训练超参数 ====================
BATCH_SIZE=32          # 根据GPU显存调整 (V100: 64, RTX3090: 32, RTX2080Ti: 16)
NUM_WORKERS=16         # 数据加载线程数
EPOCHS=80              # 总训练轮数
LR=0.001               # 初始学习率
LR_STEPS="40 60"       # 学习率衰减节点
WEIGHT_DECAY=5e-4      # 权重衰减
GRADIENT_CLIP=20       # 梯度裁剪
WARMUP_EPOCHS=5        # 学习率预热轮数

# ==================== 正则化配置 ====================
DROPOUT=0.5            # Dropout比率
LABEL_SMOOTHING=0.1    # 标签平滑（减少过拟合）

# ==================== TSM配置 ====================
SHIFT_DIV=8            # 时序位移的通道分割数
SHIFT_PLACE="blockres" # 位移位置

# ==================== 数据增强配置 ====================
# 注意：多模态数据增强需要保持空间一致性

# ==================== 输出配置 ====================
LOG_DIR="log"
MODEL_DIR="checkpoint"
EXPERIMENT_NAME="TSM_${DATASET}_${MODALITY}_${ARCH}_fusion_${FUSION_TYPE}_seg${NUM_SEGMENTS}_e${EPOCHS}"

# ==================== GPU配置 ====================
# 单GPU训练
# CUDA_VISIBLE_DEVICES=0

# 多GPU训练（推荐）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# ==================== 开始训练 ====================
echo "=========================================="
echo "实验名称: ${EXPERIMENT_NAME}"
echo "融合策略: ${FUSION_TYPE}"
echo "批大小: ${BATCH_SIZE}"
echo "学习率: ${LR}"
echo "=========================================="

python main.py ${DATASET} ${MODALITY} \
    --arch ${ARCH} \
    --num_segments ${NUM_SEGMENTS} \
    --consensus_type avg \
    --fusion_type ${FUSION_TYPE} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --lr_steps ${LR_STEPS} \
    --lr_type step \
    --dropout ${DROPOUT} \
    --weight-decay ${WEIGHT_DECAY} \
    --gd ${GRADIENT_CLIP} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --label_smoothing ${LABEL_SMOOTHING} \
    -j ${NUM_WORKERS} \
    --eval-freq 1 \
    --print-freq 20 \
    --shift \
    --shift_div ${SHIFT_DIV} \
    --shift_place ${SHIFT_PLACE} \
    --npb \
    --root_log ${LOG_DIR} \
    --root_model ${MODEL_DIR} \
    2>&1 | tee logs/train_${EXPERIMENT_NAME}.log

echo "=========================================="
echo "训练完成！"
echo "日志保存在: logs/train_${EXPERIMENT_NAME}.log"
echo "模型保存在: ${MODEL_DIR}/${EXPERIMENT_NAME}/"
echo "=========================================="

