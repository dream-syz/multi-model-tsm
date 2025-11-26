#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集验证脚本
用于检查Multi-Modal Action Recognition数据集是否正确配置
"""

import os
import sys

# 从配置文件导入数据集根路径
try:
    from ops.dataset_config import ROOT_DATASET
except ImportError:
    print("错误：无法导入 ops.dataset_config")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


def check_directory(path, expected_count=None):
    """检查目录是否存在及其内容"""
    if os.path.exists(path):
        if os.path.isdir(path):
            contents = os.listdir(path)
            count = len(contents)
            if expected_count:
                if count == expected_count:
                    return True, f"✓ 找到 {count} 个项目 (符合预期)"
                else:
                    return False, f"⚠ 找到 {count} 个项目 (预期 {expected_count})"
            else:
                return True, f"✓ 找到 {count} 个项目"
        else:
            return False, "✗ 路径存在但不是目录"
    else:
        return False, "✗ 目录不存在"


def check_file(path):
    """检查文件是否存在"""
    if os.path.exists(path):
        if os.path.isfile(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            return True, f"✓ 文件存在，包含 {len(lines)} 行"
        else:
            return False, "✗ 路径存在但不是文件"
    else:
        return False, "✗ 文件不存在"


def check_video_frames(video_path, expected_ext):
    """检查视频目录下的帧文件"""
    if not os.path.exists(video_path):
        return False, "目录不存在"
    
    files = sorted(os.listdir(video_path))
    if not files:
        return False, "目录为空"
    
    # 检查文件扩展名
    correct_ext = sum(1 for f in files if f.endswith(expected_ext))
    
    if correct_ext == len(files):
        return True, f"所有 {len(files)} 个文件格式正确 ({expected_ext})"
    else:
        return False, f"{correct_ext}/{len(files)} 个文件格式正确"


def main():
    print("=" * 70)
    print(" " * 20 + "数据集验证检查")
    print("=" * 70)
    print(f"\n数据集根目录: {ROOT_DATASET}\n")
    
    if not os.path.exists(ROOT_DATASET):
        print(f"❌ 错误：数据集根目录不存在！")
        print(f"   请检查路径：{ROOT_DATASET}")
        print(f"\n如需修改路径，请编辑：")
        print(f"   - ops/dataset_config.py (训练)")
        print(f"   - ops/dataset_config_for_pred.py (测试)")
        return
    
    # 检查目录结构
    print("【1】检查目录结构")
    print("-" * 70)
    
    dirs_to_check = {
        'training_set': None,
        'training_set/rgb_data': 2000,
        'training_set/ir_data': 2000,
        'training_set/depth_data': 2000,
        'test_set': None,
        'test_set/rgb_data': 500,
        'test_set/ir_data': 500,
        'test_set/depth_data': 500,
    }
    
    all_dirs_ok = True
    for dir_path, expected_count in dirs_to_check.items():
        full_path = os.path.join(ROOT_DATASET, dir_path)
        ok, msg = check_directory(full_path, expected_count)
        status = "✓" if ok else "✗"
        print(f"{status} {dir_path:<35} {msg}")
        if not ok:
            all_dirs_ok = False
    
    # 检查列表文件
    print("\n【2】检查列表文件")
    print("-" * 70)
    
    files_to_check = [
        'training_set/train_videofolder.txt',
        'training_set/val_videofolder.txt',
        'test_set/test_videofolder.txt',
    ]
    
    all_files_ok = True
    for file_path in files_to_check:
        full_path = os.path.join(ROOT_DATASET, file_path)
        ok, msg = check_file(full_path)
        status = "✓" if ok else "✗"
        print(f"{status} {file_path:<35} {msg}")
        if not ok:
            all_files_ok = False
    
    # 检查示例视频的帧格式
    print("\n【3】检查示例视频帧格式")
    print("-" * 70)
    
    sample_checks = [
        ('training_set/rgb_data/1', '.jpg', 'RGB训练样本'),
        ('training_set/ir_data/1', '.jpg', 'IR训练样本'),
        ('training_set/depth_data/1', '.png', 'Depth训练样本'),
    ]
    
    frames_ok = True
    for video_path, ext, desc in sample_checks:
        full_path = os.path.join(ROOT_DATASET, video_path)
        ok, msg = check_video_frames(full_path, ext)
        status = "✓" if ok else "✗"
        print(f"{status} {desc:<35} {msg}")
        if not ok:
            frames_ok = False
    
    # 总结
    print("\n" + "=" * 70)
    print(" " * 25 + "验证总结")
    print("=" * 70)
    
    if all_dirs_ok and all_files_ok and frames_ok:
        print("✅ 所有检查通过！数据集配置正确。")
        print("\n你可以开始训练了：")
        print("   python main.py mmvpr RTD --arch resnet50 --num_segments 8 ...")
    else:
        print("❌ 发现问题！请检查上述标记为 ✗ 的项目。")
        print("\n常见问题：")
        print("   1. 确认数据集已完整下载")
        print("   2. 确认目录结构符合 DATASET_SETUP.md 中的说明")
        print("   3. 确认 ROOT_DATASET 路径设置正确")
        
    print("=" * 70)


if __name__ == '__main__':
    main()

