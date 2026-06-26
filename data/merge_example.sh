#!/bin/bash

# JSON 文件合并示例脚本
# 此脚本展示了如何使用 merge_json.py 合并多个 JSON 文件

# 设置脚本所在目录为工作目录
cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "JSON 合并示例"
echo "========================================"

# 示例 1: 合并 Canon1DsMkIII 的 v2 版本训练集和测试集
echo ""
echo "示例 1: 合并 Canon1DsMkIII v2 训练集和测试集"
echo "----------------------------------------"
python3 merge_json.py \
  --input Canon1DsMkIII_direction_16bit_v2_train.json \
          Canon1DsMkIII_direction_16bit_v2_test.json \
  --prefix /mnt/dataset/camera_images \
  --output Canon1DsMkIII_v2_merged.json

# 示例 2: 合并多个不同相机的数据
echo ""
echo "示例 2: 合并多个相机的数据"
echo "----------------------------------------"
python3 merge_json.py \
  --input Canon1DsMkIII_direction_16bit_v2.json \
          Canon600D_direction_16bit_v2.json \
          OlympusEPL6_direction_16bit_v2.json \
  --prefix /mnt/dataset/all_cameras \
  --output all_cameras_v2_merged.json

echo ""
echo "========================================"
echo "所有示例执行完成"
echo "========================================"

