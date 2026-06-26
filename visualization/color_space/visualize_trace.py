#!/usr/bin/env python3
"""Visualize color trace in r/g, b/g chromaticity space.

This script visualizes the color movement trajectory during iterations
in the r/g vs b/g color space. The background is colored according to
the actual colors at each point, and the trajectory is shown with arrows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
    bg_grid, rg_grid = np.meshgrid(bg_lin, rg_lin)  # 交换顺序：x轴是b/g，y轴是r/g
    rgb = chromaticity_to_rgb(rg_grid, bg_grid)
    return bg_grid, rg_grid, rgb  # 返回顺序：bg_grid, rg_grid, rgb


def load_trace_data(json_path: Path) -> Dict[str, Any]:
    """Load trace data from JSON file.
    
    Args:
        json_path: Path to the JSON file containing iteration data
        
    Returns:
        Dictionary containing:
            - predictions: List of RGB tuples for each iteration
            - ground_truth: RGB tuple for ground truth
            - iterations: List of iteration numbers
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = []
    iterations = []
    
    for iter_data in data.get('iterations', []):
        pred_illum = iter_data.get('pred_illum')
        if pred_illum and len(pred_illum) == 3:
            predictions.append(tuple(pred_illum))
            iterations.append(iter_data.get('iteration', len(predictions)))
    
    gt_illum = data.get('gt_illum', [])
    ground_truth = tuple(gt_illum) if gt_illum and len(gt_illum) == 3 else None
    
    final_smoothed_illum = data.get('final_smoothed_illum', [])
    final_smoothed = tuple(final_smoothed_illum) if final_smoothed_illum and len(final_smoothed_illum) == 3 else None
    
    return {
        'predictions': predictions,
        'ground_truth': ground_truth,
        'iterations': iterations,
        'final_smoothed': final_smoothed,
    }


def visualize_trace(
    predictions: List[Tuple[float, float, float]],
    ground_truth: Tuple[float, float, float] = None,
    final_smoothed: Tuple[float, float, float] = None,
    iterations: List[int] = None,
    output_path: Path = None,
    rg_bounds: Tuple[float, float] = None,
    bg_bounds: Tuple[float, float] = None,
    resolution: int = 500,
) -> None:
    """Visualize color trace in r/g, b/g chromaticity space.
    
    Args:
        predictions: List of RGB tuples for each iteration
        ground_truth: RGB tuple for ground truth (optional)
        final_smoothed: RGB tuple for final smoothed illumination (optional)
        iterations: List of iteration numbers (optional, for labeling)
        output_path: Path to save the output PNG file
        rg_bounds: Tuple of (min, max) for r/g axis. If None, auto-calculated.
        bg_bounds: Tuple of (min, max) for b/g axis. If None, auto-calculated.
        resolution: Resolution of the background grid
    """
    if not predictions:
        print("Warning: No predictions to visualize")
        return
    
    # Calculate r/g and b/g ratios for all points
    all_rg_ratios = []
    all_bg_ratios = []
    
    for rgb in predictions:
        r, g, b = rgb
        if g == 0:
            print(f"Warning: Prediction {rgb} has G=0, skipping...")
            continue
        all_rg_ratios.append(r / g)
        all_bg_ratios.append(b / g)
    
    if ground_truth:
        r, g, b = ground_truth
        if g != 0:
            all_rg_ratios.append(r / g)
            all_bg_ratios.append(b / g)
    
    if final_smoothed:
        r, g, b = final_smoothed
        if g != 0:
            all_rg_ratios.append(r / g)
            all_bg_ratios.append(b / g)

    if not all_rg_ratios:
        print("Error: No valid color points to visualize")
        return
    
    # Auto-calculate bounds if not provided
    # Use larger margin to leave some space around the points
    if rg_bounds is None:
        margin = 0.08
        rg_min = min(all_rg_ratios) - margin
        rg_max = max(all_rg_ratios) + margin
        rg_bounds = (max(0.0, rg_min), rg_max)
    
    if bg_bounds is None:
        margin = 0.02
        bg_min = min(all_bg_ratios) - margin
        bg_max = max(all_bg_ratios) + margin
        bg_bounds = (max(0.0, bg_min), bg_max)
    
    # Create color background
    bg_grid, rg_grid, background_rgb = create_color_background(
        rg_bounds, bg_bounds, resolution
    )
    
    # Create figure with better styling (4:3 aspect ratio, height:width)
    fig, ax = plt.subplots(figsize=(12, 16))
    # Axes: x-axis is b/g, y-axis is r/g
    extent = [bg_grid.min(), bg_grid.max(), rg_grid.min(), rg_grid.max()]
    ax.imshow(background_rgb, origin="lower", extent=extent, alpha=0.85)
    
    # Prepare trajectory points
    traj_rg = []
    traj_bg = []
    for rgb in predictions:
        r, g, b = rgb
        if g == 0:
            continue
        traj_rg.append(r / g)
        traj_bg.append(b / g)
    
    # Draw trajectory with arrows and color gradient
    if len(traj_rg) > 1:
        # Use colormap for trajectory (blue to red)
        n_segments = len(traj_rg) - 1
        cmap = plt.cm.viridis  # or 'plasma', 'coolwarm', etc.
        colors = [cmap(i / n_segments) for i in range(n_segments)]
        
        # Draw trajectory line segments with arrows
        for i in range(n_segments):
            # Axes: x is b/g, y is r/g
            x1, y1 = traj_bg[i], traj_rg[i]
            x2, y2 = traj_bg[i + 1], traj_rg[i + 1]
            
            # Draw arrow (larger size)
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[i],
                    lw=3.5,
                    alpha=0.8,
                    connectionstyle="arc3,rad=0.0",
                    mutation_scale=25,
                ),
                zorder=5,
            )
        
        # Draw trajectory line (smooth curve)
        # Axes: x is b/g, y is r/g
        ax.plot(
            traj_bg,
            traj_rg,
            color='white',
            linestyle='--',
            linewidth=1.5,
            alpha=0.4,
            zorder=4,
        )
    
    # Draw trajectory points with gradient colors
    if len(traj_rg) > 0:
        # Start point: green circle (smaller size)
        # Axes: x is b/g, y is r/g
        ax.scatter(
            traj_bg[0],
            traj_rg[0],
            marker='o',
            s=200,
            color='#00FF00',
            edgecolors='white',
            linewidths=2.0,
            label='Gray-World',
            zorder=10,
        )

        # Middle points: colored dots (smaller size)
        if len(traj_rg) > 2:
            for i in range(1, len(traj_rg) - 1):
                color = cmap(i / (len(traj_rg) - 1))
                # Axes: x is b/g, y is r/g
                ax.scatter(
                    traj_bg[i],
                    traj_rg[i],
                    marker='o',
                    s=80,
                    color=color,
                    edgecolors='white',
                    linewidths=1.2,
                    alpha=0.9,
                    zorder=8,
                )
                # Optionally add iteration number
                if iterations and i < len(iterations):
                    ax.annotate(
                        str(iterations[i]),
                        xy=(traj_bg[i], traj_rg[i]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=20,
                        color='black',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='white', linewidth=1),
                        zorder=12,
                    )
        
        # End point: red circle (same marker as start but red)
        # Axes: x is b/g, y is r/g
        ax.scatter(
            traj_bg[-1],
            traj_rg[-1],
            marker='o',
            s=200,
            color='#FF3333',
            edgecolors='white',
            linewidths=2.0,
            label='End',
            zorder=11,
        )
    
    # Draw final_smoothed if provided (star marker, larger size, black edge)
    if final_smoothed:
        r, g, b = final_smoothed
        if g != 0:
            final_rg = r / g
            final_bg = b / g
            # Axes: x is b/g, y is r/g
            ax.scatter(
                final_bg,
                final_rg,
                marker='*',
                s=450,
                color='#FF3333',
                edgecolors='black',
                linewidths=2.5,
                label='Final',
                zorder=12,
            )
    
    # Draw ground truth if provided (larger size)
    if ground_truth:
        r, g, b = ground_truth
        if g != 0:
            gt_rg = r / g
            gt_bg = b / g
            # Axes: x is b/g, y is r/g
            ax.scatter(
                gt_bg,
                gt_rg,
                marker='*',
                s=450,
                color='#FFD700',  # Gold
                edgecolors='black',
                linewidths=2.5,
                label='Ground Truth',
                zorder=12,
            )

    # Set labels and styling
    # Axes labels: x is b/g, y is r/g
    ax.set_xlabel("b / g", fontsize=18, fontweight='bold')
    ax.set_ylabel("r / g", fontsize=18, fontweight='bold')
    ax.grid(color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    
    # Add legend
    ax.legend(
        facecolor="white",
        framealpha=0.95,
        loc="lower left",
        fontsize=24,
        edgecolor='gray',
        fancybox=True,
        shadow=True,
    )
    
    # Improve overall appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved trace visualization to {output_path}")
    plt.close(fig)


def main() -> None:
    """Main function to create color trace visualization."""
    # Get the directory where this script is located
    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "results_IMG_0564.json"
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    
    # Load trace data from JSON
    trace_data = load_trace_data(json_path)
    predictions = trace_data['predictions']
    ground_truth = trace_data['ground_truth']
    iterations = trace_data['iterations']
    final_smoothed = trace_data.get('final_smoothed')
    
    if not predictions:
        print("Error: No predictions found in JSON file")
        return
    
    # Generate output filename based on JSON filename
    output_filename = f"trace_{json_path.stem}.png"
    output_path = base_dir / output_filename
    
    print(f"Loaded {len(predictions)} iterations from {json_path.name}")
    if ground_truth:
        print(f"Ground truth: RGB={ground_truth}")
    if final_smoothed:
        print(f"Final smoothed: RGB={final_smoothed}")
    
    # Visualize trace
    visualize_trace(
        predictions=predictions,
        ground_truth=ground_truth,
        final_smoothed=final_smoothed,
        iterations=iterations,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

