#!/usr/bin/env python3
"""Generate chromaticity visualizations in r/g, b/g space.

This script creates several stylistic variants of a chromaticity scatter plot
for a sample color defined in RGB space. Each variant shades the background
according to the chromaticity coordinates by reversing r/g and b/g back to
RGB (with a simple normalization) so the canvas reflects the perceived hue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ChromaticitySample:
    """Simple container for RGB colors and their chromaticity coordinates."""

    name: str
    rgb: Tuple[float, float, float]

    @property
    def rg(self) -> float:
        return self.rgb[0] / self.rgb[1]

    @property
    def bg(self) -> float:
        return self.rgb[2] / self.rgb[1]


def chromaticity_grid_to_rgb(r_over_g: np.ndarray, b_over_g: np.ndarray) -> np.ndarray:
    """Convert r/g and b/g grids back to RGB colors for visualization.

    Given chromaticity coordinates, we assume G=1, reconstruct R=r/g*G and
    B=b/g*G, then normalize so the largest channel is 1 (if non-zero). This
    gives a vivid yet bounded RGB representation suitable for imshow.
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


def create_background(
    rg_bounds: Tuple[float, float],
    bg_bounds: Tuple[float, float],
    resolution: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build meshgrid for chromaticity space and corresponding RGB colors."""

    rg_lin = np.linspace(rg_bounds[0], rg_bounds[1], resolution)
    bg_lin = np.linspace(bg_bounds[0], bg_bounds[1], resolution)
    rg_grid, bg_grid = np.meshgrid(rg_lin, bg_lin)
    rgb = chromaticity_grid_to_rgb(rg_grid, bg_grid)
    return rg_grid, bg_grid, rgb


def plot_variant(
    name: str,
    background_rgb: np.ndarray,
    rg_grid: np.ndarray,
    bg_grid: np.ndarray,
    samples: Iterable[ChromaticitySample],
    output_path: Path,
) -> None:
    """Render a chromaticity plot with classic styling."""

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    extent = [rg_grid.min(), rg_grid.max(), bg_grid.min(), bg_grid.max()]
    ax.imshow(background_rgb, origin="lower", extent=extent)

    for sample in samples:
        ax.scatter(
            sample.rg,
            sample.bg,
            marker="x",
            s=160,
            color="white",
            linewidths=2.4,
            label=f"{sample.name} RGB={sample.rgb}",
            zorder=10,
        )

    ax.set_title("Chromaticity Map (r/g vs. b/g)")
    ax.set_xlabel("r / g")
    ax.set_ylabel("b / g")
    ax.legend(facecolor="white", framealpha=0.85, loc="upper left")
    ax.grid(color="white", linestyle="--", linewidth=0.6, alpha=0.7)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_contour_variant(
    name: str,
    background_rgb: np.ndarray,
    rg_grid: np.ndarray,
    bg_grid: np.ndarray,
    samples: Iterable[ChromaticitySample],
    output_path: Path,
) -> None:
    """Render a version with contours to emphasize chromaticity magnitude."""

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    extent = [rg_grid.min(), rg_grid.max(), bg_grid.min(), bg_grid.max()]
    ax.imshow(background_rgb, origin="lower", extent=extent)

    magnitude = np.sqrt(np.square(rg_grid) + np.square(bg_grid))
    contour = ax.contour(
        rg_grid,
        bg_grid,
        magnitude,
        levels=12,
        colors="black",
        linewidths=0.6,
        alpha=0.5,
    )
    ax.clabel(contour, inline=True, fontsize=7, fmt="{:.2f}")

    for sample in samples:
        ax.scatter(
            sample.rg,
            sample.bg,
            marker="x",
            s=180,
            color="black",
            linewidths=2.2,
            zorder=10,
        )
        ax.annotate(
            sample.name,
            (sample.rg, sample.bg),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
        )

    ax.set_xlabel("r / g")
    ax.set_ylabel("b / g")
    ax.set_title("Chromaticity Magnitude Contours")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_dark_variant(
    name: str,
    background_rgb: np.ndarray,
    rg_grid: np.ndarray,
    bg_grid: np.ndarray,
    samples: Iterable[ChromaticitySample],
    output_path: Path,
) -> None:
    """Render a dark-themed chromaticity plot for higher contrast."""

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    extent = [rg_grid.min(), rg_grid.max(), bg_grid.min(), bg_grid.max()]
    ax.imshow(background_rgb, origin="lower", extent=extent, alpha=0.85)

    overlay = ax.scatter(
        [sample.rg for sample in samples],
        [sample.bg for sample in samples],
        marker="x",
        s=200,
        color="#FFD166",
        linewidths=2.8,
        zorder=10,
    )

    ax.set_xlabel("r / g")
    ax.set_ylabel("b / g")
    ax.set_title("Chromaticity – Dark Variant")
    ax.set_facecolor("#111111")
    ax.grid(color="white", linestyle=":", linewidth=0.4, alpha=0.35)

    handles = [overlay]
    labels = ["Sample chromaticity"]
    ax.legend(handles, labels, loc="upper left")
    fig.suptitle(name, color="white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    plt.style.use("default")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir

    sample_color = ChromaticitySample(name="Sample", rgb=(1.0, 3.0, 4.0))

    rg_bounds = (0.0, 2.2)
    bg_bounds = (0.0, 2.6)
    rg_grid, bg_grid, background_rgb = create_background(rg_bounds, bg_bounds)

    variants = [
        (
            "Chromaticity Visualization – Classic",
            plot_variant,
            output_dir / "chromaticity_classic.png",
        ),
        (
            "Chromaticity Visualization – Contours",
            plot_contour_variant,
            output_dir / "chromaticity_contours.png",
        ),
        (
            "Chromaticity Visualization – Dark",
            plot_dark_variant,
            output_dir / "chromaticity_dark.png",
        ),
    ]

    for name, renderer, path in variants:
        renderer(
            name,
            background_rgb,
            rg_grid,
            bg_grid,
            samples=[sample_color],
            output_path=path,
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

