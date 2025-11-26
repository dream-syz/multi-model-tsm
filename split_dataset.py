#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMAR数据集划分脚本
将500个训练视频划分为训练集和验证集
推荐：400训练 + 100验证（80:20）
"""

import os
import random
import argparse
from collections import Counter

def split_dataset(input_file, train_ratio=0.8, seed=42, stratified=True):
    """
    将数据集列表划分为训练集和验证集
    
    Args:
        input_file: 原始训练列表文件路径
        train_ratio: 训练集比例（默认0.8，即80%训练，20%验证）
        seed: 随机种子（保证可重复）
        stratified: 是否进行分层划分（保证每个类别按比例划分）
    """
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误：找不到文件 {input_file}")
        print(f"   请确认路径是否正确")
        return
    
    # 读取所有数据
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print("=" * 70)
    print("MMAR数据集划分工具")
    print("=" * 70)
    print(f"总共读取 {len(lines)} 个视频")
    
    # 解析数据（格式：video_id num_frames label）
    data_with_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            video_id = parts[0]
            num_frames = parts[1]
            label = int(parts[2])
            data_with_labels.append((line, label))
        else:
            print(f"⚠️  警告：跳过格式不正确的行：{line.strip()}")
    
    print(f"有效视频数: {len(data_with_labels)}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 统计类别分布
    labels = [item[1] for item in data_with_labels]
    label_counter = Counter(labels)
    num_classes = len(label_counter)
    
    print(f"\n类别统计：")
    print(f"  类别数: {num_classes}")
    print(f"  最少样本的类别: {min(label_counter.values())} 个")
    print(f"  最多样本的类别: {max(label_counter.values())} 个")
    print(f"  平均每类: {len(data_with_labels)/num_classes:.1f} 个")
    
    # 划分策略
    if stratified:
        print(f"\n使用分层划分（每个类别按 {train_ratio*100:.0f}:{(1-train_ratio)*100:.0f} 划分）")
        train_lines = []
        val_lines = []
        
        # 按类别分组
        class_groups = {}
        for line, label in data_with_labels:
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(line)
        
        # 每个类别单独划分
        for label, class_lines in class_groups.items():
            random.shuffle(class_lines)
            split_point = int(len(class_lines) * train_ratio)
            train_lines.extend(class_lines[:split_point])
            val_lines.extend(class_lines[split_point:])
            
            print(f"  类别 {label:2d}: {len(class_lines):3d} 个 → "
                  f"训练 {split_point:3d}, 验证 {len(class_lines)-split_point:3d}")
    
    else:
        print(f"\n使用随机划分")
        all_lines = [item[0] for item in data_with_labels]
        random.shuffle(all_lines)
        split_point = int(len(all_lines) * train_ratio)
        train_lines = all_lines[:split_point]
        val_lines = all_lines[split_point:]
    
    # 生成输出文件路径
    dir_name = os.path.dirname(input_file)
    train_file = os.path.join(dir_name, 'train_videofolder.txt')
    val_file = os.path.join(dir_name, 'val_videofolder.txt')
    
    # 备份原始文件
    backup_file = input_file + '.backup'
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(input_file, backup_file)
        print(f"\n✓ 原始文件已备份到: {backup_file}")
    
    # 保存训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # 保存验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print("\n" + "=" * 70)
    print("✅ 数据集划分完成！")
    print("=" * 70)
    print(f"训练集: {len(train_lines):3d} 个视频 ({len(train_lines)/len(data_with_labels)*100:.1f}%)")
    print(f"  保存到: {train_file}")
    print(f"验证集: {len(val_lines):3d} 个视频 ({len(val_lines)/len(data_with_labels)*100:.1f}%)")
    print(f"  保存到: {val_file}")
    
    # 验证类别分布
    train_labels = [int(line.strip().split()[2]) for line in train_lines]
    val_labels = [int(line.strip().split()[2]) for line in val_lines]
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    print(f"\n类别覆盖检查：")
    print(f"  训练集类别数: {len(train_counter)} / {num_classes}")
    print(f"  验证集类别数: {len(val_counter)} / {num_classes}")
    
    # 检查是否所有类别都在训练集和验证集中
    missing_in_train = set(labels) - set(train_labels)
    missing_in_val = set(labels) - set(val_labels)
    
    if not missing_in_train and not missing_in_val:
        print("  ✓ 所有类别在训练集和验证集中都有样本")
    else:
        if missing_in_train:
            print(f"  ⚠️  警告：类别 {missing_in_train} 只在验证集中")
        if missing_in_val:
            print(f"  ⚠️  警告：类别 {missing_in_val} 只在训练集中")
    
    print("=" * 70)
    print("\n下一步：运行 python check_dataset.py 验证数据集配置")
    print("然后可以开始训练了！")


def main():
    parser = argparse.ArgumentParser(
        description='MMAR数据集划分工具 - 将训练集划分为训练集和验证集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 使用默认配置（80%训练，20%验证，分层划分）
  python split_dataset.py
  
  # 自定义划分比例（90%训练，10%验证）
  python split_dataset.py --ratio 0.9
  
  # 使用随机划分（不推荐）
  python split_dataset.py --no-stratified
  
  # 指定输入文件
  python split_dataset.py --input /path/to/train_videofolder.txt

推荐配置：
  对于500个训练样本：
  - 80:20 划分（400训练 + 100验证）✅ 推荐
  - 使用分层划分保证每个类别都有代表
        """
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='training_set/train_videofolder.txt',
        help='原始训练列表文件路径（默认: training_set/train_videofolder.txt）'
    )
    parser.add_argument(
        '--ratio', 
        type=float, 
        default=0.8,
        help='训练集比例（默认0.8，即80%%训练，20%%验证）'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='随机种子，用于复现结果（默认42）'
    )
    parser.add_argument(
        '--no-stratified',
        action='store_true',
        help='不使用分层划分（默认使用分层划分）'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        args.input, 
        args.ratio, 
        args.seed, 
        stratified=not args.no_stratified
    )


if __name__ == '__main__':
    main()

