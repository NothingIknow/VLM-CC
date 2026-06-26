#!/bin/bash
# 白平衡处理脚本

# 图像路径
IMAGE_PATH="/mnt/disk3/shuwei/Gehler/cs/all_normalized/IMG_0564.png"

# RGB光照估计 
RGB_R=0.450412
RGB_G=0.780699
RGB_B=0.433172


# 输出路径（可选，如果不指定会自动生成）
OUTPUT_PATH="/home/shuwei/LLaMA-Factory/visualization/color_space/IMG_0564.png"

# 其他参数
DATASET_ROOT="/mnt/disk3/shuwei/NUS-8"
TARGET_SIZE=512
USE_MASK=true

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/apply_white_balance.py"

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到Python脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 检查图像文件是否存在
if [ ! -f "$IMAGE_PATH" ]; then
    echo "错误: 找不到图像文件: $IMAGE_PATH"
    exit 1
fi

# 构建命令
CMD="python $PYTHON_SCRIPT"
CMD="$CMD --image_path \"$IMAGE_PATH\""
CMD="$CMD --rgb $RGB_R $RGB_G $RGB_B"
CMD="$CMD --dataset_root \"$DATASET_ROOT\""
CMD="$CMD --target_size $TARGET_SIZE"

# 如果指定了输出路径，添加输出参数
if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output_path \"$OUTPUT_PATH\""
fi

# 如果不使用mask，添加no_mask参数
if [ "$USE_MASK" = false ]; then
    CMD="$CMD --no_mask"
fi

# 打印命令
echo "执行命令:"
echo "$CMD"
echo ""

# 执行命令
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 白平衡处理完成！"
else
    echo ""
    echo "❌ 白平衡处理失败！"
    exit 1
fi

