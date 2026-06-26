#!/usr/bin/env python3
"""Visualize colors in r/g, b/g chromaticity space.

This script visualizes given colors in the r/g vs b/g color space.
The background is colored according to the actual colors at each point,
and the given colors are marked with crosses (×).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def chromaticity_to_rgb(r_over_g: np.ndarray, b_over_g: np.ndarray) -> np.ndarray:
    """Convert r/g and b/g grids back to RGB colors for visualization.
    
    Given chromaticity coordinates, we assume G=1, reconstruct R=r/g*G and
    B=b/g*G, then normalize so the largest channel is 1 (if non-zero).
    This gives a vivid yet bounded RGB representation suitable for imshow.
    """
    g = np.ones_like(r_over_g, dtype=float)
    r = r_over_g.astype(float)
    b = b_over_g.astype(float)
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0.0, None)
    
    max_channel = np.max(rgb, axis=-1, keepdims=True)
    np.maximum(max_channel, 1e-9, out=max_channel)  # avoid division by zero
    normalized = rgb / max_channel
    return normalized


def create_color_background(
    rg_bounds: Tuple[float, float],
    bg_bounds: Tuple[float, float],
    resolution: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build meshgrid for chromaticity space and corresponding RGB colors."""
    rg_lin = np.linspace(rg_bounds[0], rg_bounds[1], resolution)
    bg_lin = np.linspace(bg_bounds[0], bg_bounds[1], resolution)
    rg_grid, bg_grid = np.meshgrid(rg_lin, bg_lin)
    rgb = chromaticity_to_rgb(rg_grid, bg_grid)
    return rg_grid, bg_grid, rgb


def visualize_colors(
    colors: List[Tuple[float, float, float]],
    labels: List[str] = None,
    output_path: Path = None,
    rg_bounds: Tuple[float, float] = None,
    bg_bounds: Tuple[float, float] = None,
    resolution: int = 500,
) -> None:
    """Visualize colors in r/g, b/g chromaticity space.
    
    Args:
        colors: List of RGB tuples to visualize, e.g., [(1, 3, 4), ...]
        labels: List of labels for each color, e.g., ["prediction", "ground truth"]
        output_path: Path to save the output PNG file
        rg_bounds: Tuple of (min, max) for r/g axis. If None, auto-calculated.
        bg_bounds: Tuple of (min, max) for b/g axis. If None, auto-calculated.
        resolution: Resolution of the background grid
    """
    if labels is None:
        labels = [f"Color {i+1}" for i in range(len(colors))]
    
    if len(labels) != len(colors):
        raise ValueError("Number of labels must match number of colors")
    
    # Calculate r/g and b/g ratios for all colors
    rg_ratios = []
    bg_ratios = []
    for rgb in colors:
        r, g, b = rgb
        if g == 0:
            print(f"Warning: Color {rgb} has G=0, skipping...")
            continue
        rg_ratios.append(r / g)
        bg_ratios.append(b / g)
    
    # Auto-calculate bounds if not provided
    if rg_bounds is None:
        margin = 0.2
        rg_min = min(rg_ratios) - margin if rg_ratios else 0.0
        rg_max = max(rg_ratios) + margin if rg_ratios else 1.0
        rg_bounds = (max(0.0, rg_min), rg_max)
    
    if bg_bounds is None:
        margin = 0.2
        bg_min = min(bg_ratios) - margin if bg_ratios else 0.0
        bg_max = max(bg_ratios) + margin if bg_ratios else 1.0
        bg_bounds = (max(0.0, bg_min), bg_max)
    
    # Create color background
    rg_grid, bg_grid, background_rgb = create_color_background(
        rg_bounds, bg_bounds, resolution
    )
    
    # Create figure
    # figsize: (width, height) in inches - 控制图片尺寸
    fig, ax = plt.subplots(figsize=(10, 8))
    # extent: [xmin, xmax, ymin, ymax] - 定义图像显示的范围
    extent = [rg_grid.min(), rg_grid.max(), bg_grid.min(), bg_grid.max()]
    # imshow: 显示背景颜色图
    #   origin="lower": y轴从下往上增长（默认从上往下）
    #   extent: 图像在数据坐标系中的范围
    ax.imshow(background_rgb, origin="lower", extent=extent)
    
    # Plot each color with a cross marker
    for i, (rgb, label) in enumerate(zip(colors, labels)):
        r, g, b = rgb
        if g == 0:
            continue
        
        rg_ratio = r / g
        bg_ratio = b / g
        
        # 根据label确定标记颜色：ground truth用红色，prediction用白色
        marker_color = "#FF3333" if label.lower() == "ground truth" else "red"
        
        # scatter: 绘制散点图（颜色点）
        #   rg_ratio, bg_ratio: 点的x和y坐标（r/g和b/g比值）
        #   marker="x": 标记形状为叉号（×）
        #   s=100: 标记的大小（像素面积的平方）
        #   color: 标记的颜色（ground truth为红色#FF3333，prediction为白色）
        #   linewidths=3: 标记线条的宽度
        #   label: 图例标签文本
        #   zorder=10: 图层顺序，数值越大越在上层
        ax.scatter(
            rg_ratio,
            bg_ratio,
            marker="x",
            s=300,
            color=marker_color,
            linewidths=3,
            # label=f"{label}: RGB=({r:.3f}, {g:.3f}, {b:.3f})",
            zorder=10,
        )

    # Connect multiple predictions with arrows in input order
    # Keep behavior unchanged if there is only one prediction
    prediction_points = []
    for (rgb, label) in zip(colors, labels):
        if label.lower() == "prediction":
            r, g, b = rgb
            if g == 0:
                continue
            prediction_points.append((r / g, b / g))

    # if len(prediction_points) > 1:
    #     # 依次连接prediction点，用箭头表示顺序
    #     for (x1, y1), (x2, y2) in zip(prediction_points[:-1], prediction_points[1:]):
    #         # annotate: 绘制箭头连接两个点
    #         #   "": 空字符串表示不显示文本，只显示箭头
    #         #   xy=(x2, y2): 箭头指向的终点坐标
    #         #   xytext=(x1, y1): 箭头起始点的坐标
    #         #   arrowprops: 箭头属性字典
    #         #     arrowstyle="->": 箭头样式为简单箭头
    #         #     color="red": 箭头颜色
    #         #     lw=2.0: 箭头线条宽度（linewidth）
    #         #     alpha=0.9: 透明度（0-1，1为完全不透明）
    #         #   zorder=11: 图层顺序，确保箭头在散点之上
    #         ax.annotate(
    #             "",
    #             xy=(x2, y2),
    #             xytext=(x1, y1),
    #             arrowprops=dict(arrowstyle="->", color="black", lw=2.0, alpha=0.9),
    #             zorder=11,
    #         )
    
    # 设置图表标题和坐标轴标签
    #   fontsize: 字体大小
    #   fontweight="bold": 字体粗细（粗体）
    # ax.set_title("Color Visualization in r/g vs b/g Space", fontsize=14, fontweight="bold")
    ax.set_xlabel("r / g", fontsize=12)  # x轴标签：红色与绿色的比值
    ax.set_ylabel("b / g", fontsize=12)  # y轴标签：蓝色与绿色的比值
    
    # 隐藏坐标轴刻度数字
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    
    # if len(colors) > 0:
    #     # legend: 显示图例
    #     #   facecolor="white": 图例背景颜色
    #     #   framealpha=0.9: 图例背景透明度
    #     #   loc="upper left": 图例位置（左上角）
    #     #   fontsize=10: 图例字体大小
    #     ax.legend(facecolor="white", framealpha=0.9, loc="upper left", fontsize=10)
    
    # grid: 显示网格线
    #   color="white": 网格线颜色
    #   linestyle="--": 线条样式（虚线）
    #   linewidth=0.8: 线条宽度
    #   alpha=0.6: 透明度
    ax.grid(color="white", linestyle="--", linewidth=0.8, alpha=0.6)
    
    # tight_layout: 自动调整子图参数，使布局紧凑
    fig.tight_layout()
    if output_path:
        # savefig: 保存图片
        #   dpi=200: 分辨率（每英寸点数），数值越大图片越清晰
        #   bbox_inches="tight": 自动裁剪图片边缘空白
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    plt.close(fig)


def main() -> None:
    """Main function to create color visualization."""
    # Get the directory where this script is located
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "color_visualization.png"
    
    # Prediction sequence and ground truth
    predictions = [
        # (   0.384812593460083,
        # 0.8286288380622864,
        # 0.40656301379203796),
        #  (   0.384812593460083,
        # 0.8286288380622864,
        # 0.55656301379203796),
        # (   0.384812593460083,
        # 0.8286288380622864,
        # 0.65656301379203796),
        # (   0.384812593460083,
        # 0.8286288380622864,
        # 0.60656301379203796),
        # (0.3573501408100128,
        # 0.769493043422699,
        # 0.5293216109275818),
        # (0.3573501408100128,
        # 0.769493043422699,
        # 0.5293216109275818),
    ]

    # ground_truth = (0.3372473120689392,
    # 0.7464116811752319,
    # 0.5737018585205078)

    # colors = [*predictions, ground_truth]
    colors = [*predictions]
    # labels = ["prediction"] * len(predictions) + ["ground truth"]
    labels = ["prediction"] * len(predictions)
    
    visualize_colors(
        colors=colors,
        labels=labels,
        output_path=output_path,
        rg_bounds=(max(0.0, 0.10), 0.9),
        bg_bounds=(max(0.0, 0.10), 0.9)
    )


if __name__ == "__main__":
    main()

