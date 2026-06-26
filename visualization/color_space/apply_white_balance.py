#!/usr/bin/env python3
"""
白平衡处理脚本
功能：给定RGB光照估计和图片路径，对图片进行完整的白平衡处理
包含：black level处理、saturation处理、CCM色彩校正、resize等
"""

import os
import sys
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
import cv2
import scipy.io

# 添加LLaMA-Factory到路径
sys.path.append('/home/shuwei/LLaMA-Factory')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhiteBalanceProcessor:
    """白平衡处理器，整合CCM、black level、resize等功能"""
    
    def __init__(self, dataset_root: str = "/mnt/disk3/shuwei/NUS-8", 
                 matrix_file: Optional[str] = None):
        """
        初始化白平衡处理器
        
        Args:
            dataset_root: 数据集根目录
            matrix_file: cam2srgb_matrices.json文件路径（可选，会自动查找）
        """
        self.dataset_root = dataset_root
        self.rgb_cam_cache = {}  # 色彩校正矩阵缓存
        self._mat_cache = {}  # GT数据缓存
        
        # 定位矩阵文件
        if matrix_file:
            self.matrix_file = matrix_file
        else:
            self.matrix_file = self._locate_matrix_file()
    
    @staticmethod
    def extract_camera_from_filename(image_path: str) -> str:
        """
        从图像文件名/路径自动提取相机名称
        
        支持四种数据集格式:
        1. NUS-8: {CameraName}_{ImageID}.PNG -> CameraName
        2. Gehler: {相机目录}/{imagecode}.png -> 从目录名提取
        3. Inter-tau: {CameraName}_{ImageCode}_{tau}.png -> CameraName
        4. Cube+: {数字}.PNG -> Canon_EOS_550D
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: 相机名称
        """
        basename = os.path.basename(image_path)
        dirname = os.path.basename(os.path.dirname(image_path))
        filename_no_ext = os.path.splitext(basename)[0]
        
        # 规则1: Gehler格式 - 如果父目录名是 canon1d 或 canon5d
        if dirname in ['canon1d', 'canon5d']:
            return dirname
        
        # 规则1.5: Gehler格式 - 基于文件名前缀识别
        if basename.startswith('IMG_'):
            return 'canon5d'
        if re.match(r'^[0-9][A-Z0-9]+', filename_no_ext) and not filename_no_ext.isdigit():
            return 'canon1d'
        
        # 规则2: Cube+格式 - 纯数字文件名
        if filename_no_ext.isdigit():
            return 'Canon_EOS_550D'
        
        # 规则3: Inter-tau格式
        if '_' in basename:
            parts = basename.split('_')
            if len(parts) >= 2:
                potential_camera = f"{parts[0]}_{parts[1]}"
                if potential_camera in ['Canon_5DSR', 'Nikon_D810']:
                    return potential_camera
        
        # 规则4: NUS-8格式
        if '_' in filename_no_ext:
            camera_name = filename_no_ext.split('_')[0]
            known_nus8_cameras = ['Canon1DsMkIII', 'Canon600D', 'FujifilmXM1', 
                                 'NikonD5200', 'OlympusEPL6', 'PanasonicGX1', 
                                 'SamsungNX2000', 'SonyA57']
            if camera_name in known_nus8_cameras:
                return camera_name
        
        raise ValueError(f"无法从文件名提取相机名: {basename}")
    
    def _locate_matrix_file(self) -> str:
        """定位cam2srgb_matrices.json文件"""
        possible_paths = [
            os.path.join(os.path.dirname(self.dataset_root), "NUS-8", "scripts", "cam2srgb_matrices.json"),
            "/mnt/disk3/shuwei/NUS-8/scripts/cam2srgb_matrices.json",
            "/home2/shuwei/NUS-8/scripts/cam2srgb_matrices.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"找到色彩矩阵文件: {path}")
                return path
        
        raise FileNotFoundError(f"未找到cam2srgb_matrices.json文件，尝试过的路径: {possible_paths}")
    
    def _load_camera_matrix(self, camera_name: str) -> np.ndarray:
        """
        加载指定相机的色彩校正矩阵（带缓存）
        
        Args:
            camera_name: 相机名称
            
        Returns:
            np.ndarray: rgb_cam矩阵，用于从相机RGB转换到标准RGB
        """
        # 检查缓存
        if camera_name in self.rgb_cam_cache:
            return self.rgb_cam_cache[camera_name]
        
        # 加载矩阵文件
        with open(self.matrix_file, 'r') as f:
            matrices = json.load(f)
        
        # 检查相机名是否存在
        if camera_name not in matrices:
            available_cameras = [k for k in matrices.keys() if k not in ['rgb_xyz', 'xyz_rgb']]
            raise ValueError(f"相机 '{camera_name}' 不在cam2srgb_matrices.json中。"
                           f"可用相机: {available_cameras}")
        
        # 计算cam2srgb矩阵
        cam_xyz = np.array(matrices[camera_name])  # 相机到XYZ
        xyz_rgb = np.array(matrices["xyz_rgb"])     # XYZ到RGB
        cam_rgb = cam_xyz @ xyz_rgb                 # 相机到RGB
        
        # 归一化
        row_sums = cam_rgb.sum(axis=1, keepdims=True)
        cam_rgb_norm = cam_rgb / row_sums
        
        # 计算逆矩阵（RGB to CAM的逆向）
        rgb_cam = np.linalg.inv(cam_rgb_norm)
        rgb_cam = rgb_cam.astype(np.float32)
        
        # 缓存矩阵
        self.rgb_cam_cache[camera_name] = rgb_cam
        
        logger.info(f"✅ 成功加载相机色彩校正矩阵: {camera_name}")
        return rgb_cam
    
    def get_camera_params(self, image_path: str, camera_name: Optional[str] = None) -> Tuple:
        """
        获取相机参数（black level、saturation level等）
        
        Args:
            image_path: 图像文件路径
            camera_name: 相机名称（可选，会自动提取）
            
        Returns:
            tuple: (darkness_level, saturation_level, CC_coords, camera_name)
        """
        if camera_name is None:
            camera_name = self.extract_camera_from_filename(image_path)
        
        # 根据相机名判断数据集类型并获取参数
        if camera_name in ['canon1d', 'canon5d']:
            # Gehler数据集
            image_name = os.path.basename(image_path)
            if camera_name == "canon1d":
                darkness_level = np.array([0, 0, 0], dtype=np.float32)
                saturation_level = np.array([3588, 3588, 3588], dtype=np.float32)
            elif camera_name == "canon5d":
                darkness_level = np.array([128, 128, 128], dtype=np.float32)
                saturation_level = np.array([3650, 3650, 3650], dtype=np.float32)
            
            # 从GT文件获取CC_coords
            mat_file = '/mnt/disk3/shuwei/Gehler/gehler_gt_aligned.mat'
            if mat_file not in self._mat_cache:
                self._mat_cache[mat_file] = scipy.io.loadmat(mat_file)
            mat = self._mat_cache[mat_file]
            name_list = mat['filenames'].flatten()
            idx_arr = np.where(name_list == image_name)[0]
            if len(idx_arr) > 0:
                idx = idx_arr[0]
                CC_coords = mat["cc_coords"][idx]
            else:
                CC_coords = None
                logger.warning(f"在Gehler GT文件中找不到图像: {image_name}，使用None作为CC_coords")
            
            return darkness_level, saturation_level, CC_coords, camera_name
        
        elif camera_name == 'Canon_EOS_550D':
            # Cube+数据集
            darkness_level = np.array([2048, 2048, 2048], dtype=np.float32)
            saturation_level = np.array([1, 1, 1], dtype=np.float32)
            # 色卡坐标：右下角 [y0, y1, x0, x1] (比例坐标)
            CC_coords = np.array([0.6, 1.0, 28/37.5, 1.0], dtype=np.float32)
            return darkness_level, saturation_level, CC_coords, camera_name
        
        elif camera_name in ['Canon_5DSR', 'Nikon_D810']:
            # Inter-tau数据集
            darkness_level = np.array([0, 0, 0], dtype=np.float32)
            saturation_level = np.array([1, 1, 1], dtype=np.float32)
            CC_coords = None
            return darkness_level, saturation_level, CC_coords, camera_name
        
        else:
            # NUS-8数据集
            image_name = os.path.basename(image_path)
            try:
                image_idx = int(image_name[-8:-4]) - 1
            except (ValueError, IndexError):
                logger.warning(f"无法从图像名 {image_name} 解析索引，使用0")
                image_idx = 0
            
            mat_file = os.path.join(self.dataset_root, f"{camera_name}_gt.mat")
            if mat_file not in self._mat_cache:
                self._mat_cache[mat_file] = scipy.io.loadmat(mat_file)
            mat = self._mat_cache[mat_file]
            
            darkness_level = mat["darkness_level"].squeeze()
            saturation_level = mat["saturation_level"].squeeze()
            CC_coords = mat["CC_coords"][image_idx]
            
            return darkness_level, saturation_level, CC_coords, camera_name
    
    def read_linear_png(self, path: str, black_level: np.ndarray, 
                       saturation_level: np.ndarray, CC_coords: Optional[np.ndarray],
                       target_size: Optional[int] = None, use_mask: bool = True) -> np.ndarray:
        """
        读取线性PNG图像并进行完整的预处理
        
        Args:
            path: 图像路径
            black_level: 黑电平
            saturation_level: 饱和度电平
            CC_coords: 色卡坐标 [y0, y1, x0, x1]
            target_size: 目标尺寸（短边），None表示不缩放
            use_mask: 是否遮掉色卡区域
            
        Returns:
            np.ndarray: 预处理后的0-1范围线性RGB图像
        """
        # 读取图像
        img_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img_raw.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 黑电平校正
        img -= black_level
        
        # 8/16位判别和归一化
        if img_raw.dtype == np.uint16:
            # 16bit图像
            img /= 65535.0
        else:
            # 8bit图像或其他
            if img.max() > 1.5:
                img /= 255.0
        
        # 饱和度裁剪
        img = np.clip(img, 0, saturation_level)
        
        # 全图归一化
        img /= np.max(img) + 1e-9
        
        # 遮掉色卡区域（仅在use_mask=True时）
        if use_mask and CC_coords is not None:
            # 判断CC_coords是比例坐标还是像素坐标
            is_ratio_coords = all(coord <= 1.0 for coord in CC_coords)
            
            if is_ratio_coords:
                h, w = img.shape[:2]
                y0_ratio, y1_ratio, x0_ratio, x1_ratio = map(float, CC_coords)
                y0 = int(y0_ratio * h)
                y1 = int(y1_ratio * h)
                x0 = int(x0_ratio * w)
                x1 = int(x1_ratio * w)
                
                y0 = max(0, min(y0, h-1))
                y1 = max(0, min(y1, h))
                x0 = max(0, min(x0, w-1))
                x1 = max(0, min(x1, w))
                
                img[y0:y1, x0:x1, :] = 0.0
            else:
                # NUS-8格式：像素坐标
                y0, y1, x0, x1 = map(int, CC_coords)
                img[y0:y1, x0:x1, :] = 0.0
        
        # 可选的尺寸调整
        if target_size is not None:
            h, w = img.shape[:2]
            if h < w:
                scale = target_size / h
                new_h = target_size
                new_w = int(w * scale)
            else:
                scale = target_size / w
                new_w = target_size
                new_h = int(h * scale)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def linear_to_srgb(self, img: np.ndarray, camera_name: str) -> np.ndarray:
        """
        将线性RGB转换为sRGB色彩空间，包含相机色彩校正
        
        Args:
            img: 线性RGB图像 [0, 1]
            camera_name: 相机名称
            
        Returns:
            np.ndarray: sRGB图像 [0, 1]
        """
        img = img / np.max(img + 1e-9)
        img = np.clip(img, 0.0, 1.0)
        
        # 加载对应相机的色彩校正矩阵
        rgb_cam = self._load_camera_matrix(camera_name)
        
        # 应用相机色彩校正矩阵 (cam2srgb)
        original_shape = img.shape
        img_flat = img.reshape(-1, 3)
        img_corrected = np.dot(img_flat, rgb_cam.T)
        img_corrected = img_corrected.reshape(original_shape)
        img_corrected = np.clip(img_corrected, 0.0, 1.0)
        
        # 应用sRGB gamma校正
        mask = img_corrected <= 0.0031308
        srgb = np.empty_like(img_corrected)
        srgb[mask] = 12.92 * img_corrected[mask]
        srgb[~mask] = 1.055 * np.power(img_corrected[~mask], 1/2.4) - 0.055
        return np.clip(srgb, 0, 1)
    
    def apply_white_balance(self, image_path: str, rgb_illum: Union[Tuple, list, np.ndarray],
                           output_path: Optional[str] = None, target_size: Optional[int] = None,
                           use_mask: bool = True, camera_name: Optional[str] = None) -> np.ndarray:
        """
        对图像应用白平衡处理（包含CCM、black level、resize等）
        
        Args:
            image_path: 原始图像路径
            rgb_illum: RGB光照估计 [R, G, B]（可以是tuple、list或numpy数组）
            output_path: 输出路径（可选，如果为None则不保存）
            target_size: 目标尺寸（短边），None表示不缩放
            use_mask: 是否遮掉色卡区域
            camera_name: 相机名称（可选，会自动提取）
            
        Returns:
            np.ndarray: 白平衡校正后的sRGB图像 [0, 1]
        """
        # 转换RGB光照为numpy数组
        if isinstance(rgb_illum, (tuple, list)):
            rgb_illum = np.array(rgb_illum, dtype=np.float32)
        elif isinstance(rgb_illum, np.ndarray):
            rgb_illum = rgb_illum.astype(np.float32)
        else:
            raise ValueError(f"不支持的rgb_illum类型: {type(rgb_illum)}")
        
        # 归一化光照
        rgb_illum = rgb_illum / np.linalg.norm(rgb_illum)
        
        # 获取相机参数
        if camera_name is None:
            camera_name = self.extract_camera_from_filename(image_path)
        
        darkness_level, saturation_level, CC_coords, camera_name = self.get_camera_params(
            image_path, camera_name
        )
        
        logger.info(f"处理图像: {os.path.basename(image_path)}")
        logger.info(f"相机: {camera_name}")
        logger.info(f"光照估计: {rgb_illum}")
        
        # 读取并预处理线性RGB图像
        img = self.read_linear_png(
            image_path, darkness_level, saturation_level, CC_coords,
            target_size=target_size, use_mask=use_mask
        )
        
        # 应用白平衡
        # 归一化到G通道为1
        illum_gain = rgb_illum / rgb_illum[1]
        
        # 应用白平衡增益
        img_corrected = img / illum_gain
        
        # 线性RGB转sRGB（使用对应相机的色彩矩阵）
        img_srgb = self.linear_to_srgb(img_corrected, camera_name)
        
        # 保存结果（如果需要）
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            img_16bit = (img_srgb * 65535).astype(np.uint16)
            img_16bit = cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_16bit)
            logger.info(f"✅ 已保存白平衡图像: {output_path}")
        
        return img_srgb


def main():
    parser = argparse.ArgumentParser(description="白平衡处理脚本（包含CCM、black level、resize等）")
    parser.add_argument("--image_path", required=True, help="输入图像路径")
    parser.add_argument("--rgb", required=True, nargs=3, type=float, 
                       help="RGB光照估计，例如: --rgb 0.366742 0.776986 0.511658")
    parser.add_argument("--output_path", help="输出图像路径（可选）")
    parser.add_argument("--dataset_root", default="/mnt/disk3/shuwei/NUS-8", 
                       help="数据集根目录")
    parser.add_argument("--matrix_file", default=None, 
                       help="cam2srgb_matrices.json文件路径（可选，会自动查找）")
    parser.add_argument("--camera_name", default=None, 
                       help="相机名称（可选，会自动从文件名提取）")
    parser.add_argument("--target_size", type=int, default=512, 
                       help="目标尺寸（短边），None表示不缩放")
    parser.add_argument("--no_mask", action="store_true", 
                       help="不遮掉色卡区域（默认会遮掉）")
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = WhiteBalanceProcessor(
        dataset_root=args.dataset_root,
        matrix_file=args.matrix_file
    )
    
    # 应用白平衡
    rgb_illum = tuple(args.rgb)
    output_path = args.output_path
    
    # 如果没有指定输出路径，自动生成
    if output_path is None:
        input_path = Path(args.image_path)
        output_path = str(input_path.parent / f"{input_path.stem}_wb{input_path.suffix}")
    
    try:
        img_srgb = processor.apply_white_balance(
            image_path=args.image_path,
            rgb_illum=rgb_illum,
            output_path=output_path,
            target_size=args.target_size,
            use_mask=not args.no_mask,
            camera_name=args.camera_name
        )
        
        logger.info(f"✅ 白平衡处理完成！")
        logger.info(f"输出图像: {output_path}")
        logger.info(f"图像尺寸: {img_srgb.shape}")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

