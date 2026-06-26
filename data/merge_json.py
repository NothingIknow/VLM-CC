#!/usr/bin/env python3
"""
JSON 文件合并工具
功能：合并多个 JSON 文件并修改其中的图片路径
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_json_file(filepath):
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载: {filepath} ({len(data)} 条记录)")
        return data
    except FileNotFoundError:
        print(f"✗ 错误: 文件不存在 - {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ 错误: JSON 解析失败 - {filepath}\n  {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"✗ 错误: 读取文件失败 - {filepath}\n  {e}", file=sys.stderr)
        sys.exit(1)


def update_image_paths(data, prefix):
    """
    更新数据中的图片路径
    
    Args:
        data: JSON 数据列表
        prefix: 要添加的前置路径
    
    Returns:
        更新后的数据
    """
    updated_count = 0
    for item in data:
        if 'images' in item and isinstance(item['images'], list):
            new_images = []
            for img_path in item['images']:
                # 使用 os.path.join 合并路径
                new_path = os.path.join(prefix, img_path)
                new_images.append(new_path)
                updated_count += 1
            item['images'] = new_images
    
    print(f"✓ 更新了 {updated_count} 个图片路径")
    return data


def merge_json_files(input_files, prefix, output_file):
    """
    合并多个 JSON 文件并更新图片路径
    
    Args:
        input_files: 输入的 JSON 文件列表
        prefix: 图片路径前缀
        output_file: 输出文件路径
    """
    print("=" * 60)
    print("JSON 文件合并工具")
    print("=" * 60)
    
    # 1. 加载所有 JSON 文件
    print(f"\n[1/4] 加载 {len(input_files)} 个 JSON 文件...")
    all_data = []
    for filepath in input_files:
        data = load_json_file(filepath)
        all_data.extend(data)
    
    print(f"\n✓ 总共加载 {len(all_data)} 条记录")
    
    # 2. 更新图片路径
    print(f"\n[2/4] 更新图片路径 (前缀: '{prefix}')...")
    updated_data = update_image_paths(all_data, prefix)
    
    # 3. 保存合并后的文件
    print(f"\n[3/4] 保存到文件: {output_file}...")
    try:
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
        
        # 获取文件大小
        file_size = output_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        print(f"✓ 成功保存 ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"✗ 错误: 保存文件失败\n  {e}", file=sys.stderr)
        sys.exit(1)
    
    # 4. 完成
    print(f"\n[4/4] 完成!")
    print("=" * 60)
    print(f"合并结果:")
    print(f"  - 输入文件: {len(input_files)} 个")
    print(f"  - 总记录数: {len(updated_data)} 条")
    print(f"  - 输出文件: {output_file}")
    print(f"  - 路径前缀: {prefix}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='合并多个 JSON 文件并更新图片路径',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合并两个 JSON 文件
  python merge_json.py \\
    --input file1.json file2.json \\
    --prefix /data/images \\
    --output merged.json
  
  # 合并多个相机数据
  python merge_json.py \\
    --input Canon1DsMkIII_v2.json Canon600D_v2.json OlympusEPL6_v2.json \\
    --prefix /mnt/dataset/camera_images \\
    --output all_cameras_merged.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        nargs='+',
        required=True,
        help='输入的 JSON 文件列表（可以指定多个）'
    )
    
    parser.add_argument(
        '--prefix', '-p',
        required=True,
        help='添加到图片路径前的前缀路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='输出的 JSON 文件名'
    )
    
    args = parser.parse_args()
    
    # 执行合并
    merge_json_files(args.input, args.prefix, args.output)


if __name__ == '__main__':
    main()

