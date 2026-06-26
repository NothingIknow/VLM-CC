#!/usr/bin/env python3
"""Paper benchmark: run LoRA on test set and report angular-error statistics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from tqdm import tqdm

VLM_CC_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(VLM_CC_ROOT / "src"))
sys.path.insert(0, str(EVAL_DIR))

from direction_iteration import (  # noqa: E402
    CUBEPLUS_CC,
    CUBEPLUS_DARK,
    CUBEPLUS_GT_MAT,
    CUBEPLUS_SAT,
    DirectionTester,
    GEHLER_CALIB,
    GEHLER_GT_MAT,
    VALID_CAMERAS,
    angular_error,
)

PAPER = {
    "Gehler": {"mean": 1.52, "median": 1.18, "trimean": 1.20, "best_25": 0.41, "worst_25": 3.29},
    "NUS8": {"mean": 1.83, "median": 1.44, "trimean": 1.50, "best_25": 0.51, "worst_25": 3.88},
    "Cube+": {"mean": 1.51, "median": 1.09, "trimean": 1.21, "best_25": 0.41, "worst_25": 3.28},
}

BENCHMARKS = {
    "Gehler": {
        "lora": VLM_CC_ROOT / "models/nus-cube-inter_qwen2.5vl-7b",
        "input": Path("/mnt/disk3/shuwei/Gehler/cs/all"),
        "nus8_root": None,
        "glob_ext": [".png", ".PNG"],
    },
    "NUS8": {
        "lora": VLM_CC_ROOT / "models/gehler-cube-inter_qwen2.5vl-7b",
        "input": Path("/mnt/disk3/shuwei/NUS-8/testset-200_PNG"),
        "nus8_root": Path("/mnt/disk3/shuwei/NUS-8"),
        "glob_ext": [".png", ".PNG"],
    },
    "Cube+": {
        "lora": VLM_CC_ROOT / "models/nus-gehler-inter_qwen2.5vl-7b",
        "input": Path("/mnt/disk3/shuwei/Cube+/testset-200_PNG"),
        "nus8_root": None,
        "glob_ext": [".png", ".PNG"],
    },
}

_MAT_CACHE: Dict[str, dict] = {}


def _load_mat(path: str) -> dict:
    if path not in _MAT_CACHE:
        _MAT_CACHE[path] = scipy.io.loadmat(path)
    return _MAT_CACHE[path]


def gt_calib(
    image_path: str,
    nus8_root: Optional[Path],
    camera_override: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str, np.ndarray]:
    if camera_override:
        if camera_override not in VALID_CAMERAS:
            raise ValueError(f"Unknown camera: {camera_override}")
        camera = camera_override
    else:
        camera = DirectionTester.camera_from_path(image_path)
    name = os.path.basename(image_path)

    if camera in GEHLER_CALIB:
        mat = _load_mat(GEHLER_GT_MAT)
        idx = np.where(mat["filenames"].flatten() == name)[0]
        if len(idx) == 0:
            raise ValueError(f"Gehler GT not found: {name}")
        dark, sat = GEHLER_CALIB[camera]
        gt = mat["gt_illum"][idx[0]].astype(np.float32)
        gt = gt / gt[1]
        cc = mat["cc_coords"][idx[0]]
        return _vec3(dark), _vec3(sat), cc, camera, gt / np.linalg.norm(gt)

    if camera == "Canon_EOS_550D":
        mat = _load_mat(CUBEPLUS_GT_MAT)
        filenames = mat["filenames"].flatten()
        idx = next(
            (i for i, fn in enumerate(filenames)
             if (fn[0] if isinstance(fn, np.ndarray) else str(fn)) == name),
            None,
        )
        if idx is None:
            raise ValueError(f"Cube+ GT not found: {name}")
        gt = mat["gt"][idx].astype(np.float32)
        gt_norm = gt / np.linalg.norm(gt)
        return CUBEPLUS_DARK.copy(), CUBEPLUS_SAT.copy(), CUBEPLUS_CC.copy(), camera, gt_norm

    root = str(nus8_root or "/mnt/disk3/shuwei/NUS-8")
    mat_path = os.path.join(root, f"{camera}_gt.mat")
    mat = _load_mat(mat_path)
    try:
        idx = int(name[-8:-4]) - 1
    except (ValueError, IndexError):
        idx = 0
    gt = mat["groundtruth_illuminants"][idx].astype(np.float32)
    gt_norm = gt / np.linalg.norm(gt)
    return (
        mat["darkness_level"].squeeze(),
        mat["saturation_level"].squeeze(),
        mat["CC_coords"][idx],
        camera,
        gt_norm,
    )


def _vec3(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float32)
    return np.full(3, value, dtype=np.float32)


def error_metrics(errors: List[float]) -> Dict[str, float]:
    if not errors:
        return {k: 0.0 for k in ("mean", "median", "trimean", "best_25", "worst_25")}
    arr = np.array(errors, dtype=np.float64)
    p25, p50, p75 = np.percentile(arr, [25, 50, 75])
    tri = float(np.sum([p25, 2 * p50, p75]) / 4)
    best_25 = float(arr[arr <= p25].mean())
    worst_25 = float(arr[arr >= p75].mean())
    return {
        "mean": float(arr.mean()),
        "median": float(p50),
        "trimean": tri,
        "best_25": best_25,
        "worst_25": worst_25,
    }


def list_images(cfg: dict) -> List[str]:
    files: List[str] = []
    for ext in cfg["glob_ext"]:
        files.extend(str(p) for p in cfg["input"].rglob(f"*{ext}"))
        files.extend(str(p) for p in cfg["input"].rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def run_benchmark(
    name: str,
    base_model: str,
    output_dir: Path,
    infer_dtype: str,
    max_images: Optional[int] = None,
    start_angle: float = 5.0,
    end_angle: float = 0.1,
    angle_reduction_factor: float = 0.5,
    max_iterations: int = 20,
    lora: Optional[Path] = None,
    use_gw_first: bool = False,
    img_size: int = 256,
) -> Dict:
    cfg = BENCHMARKS[name]
    lora_path = Path(lora) if lora else cfg["lora"]
    images = list_images(cfg)
    if max_images:
        images = images[:max_images]

    output_dir.mkdir(parents=True, exist_ok=True)
    nus8_root = cfg["nus8_root"]

    tester = DirectionTester(
        str(lora_path),
        base_model=base_model,
        infer_dtype=infer_dtype,
    )

    def patched_get_camera_params(image_path: str, camera_override: Optional[str] = None):
        dark, sat, cc, camera, _gt = gt_calib(image_path, nus8_root, camera_override)
        return dark, sat, cc, camera

    tester.get_camera_params = patched_get_camera_params  # type: ignore[method-assign]

    if not tester.load_model():
        raise RuntimeError("Model load failed")

    errors: List[float] = []
    failed: List[str] = []
    try:
        for image_path in tqdm(images, desc=name):
            stem = Path(image_path).stem
            try:
                _, _, _, _, gt_norm = gt_calib(image_path, nus8_root)
                result = tester.run_iteration_test(
                    image_path,
                    max_iterations=max_iterations,
                    output_dir=str(output_dir / stem),
                    use_gw_first=use_gw_first,
                    start_angle=start_angle,
                    end_angle=end_angle,
                    angle_reduction_factor=angle_reduction_factor,
                    img_size=img_size,
                    save_json=False,
                    save_trace=False,
                )
                if "final_smoothed_illum" not in result:
                    raise RuntimeError("no final_smoothed_illum")
                pred = np.array(result["final_smoothed_illum"], dtype=np.float32)
                errors.append(angular_error(pred, gt_norm))
            except Exception as exc:
                failed.append(f"{stem}: {exc}")
    finally:
        tester.cleanup()

    metrics = error_metrics(errors)
    paper = PAPER[name]
    report = {
        "benchmark": name,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "angle_reduction_factor": angle_reduction_factor,
        "max_iterations": max_iterations,
        "use_gw_first": use_gw_first,
        "img_size": img_size,
        "lora": str(lora_path),
        "input": str(cfg["input"]),
        "num_images": len(images),
        "num_success": len(errors),
        "num_failed": len(failed),
        "metrics": metrics,
        "paper": paper,
        "failed_samples": failed[:20],
    }
    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def print_report(report: Dict) -> None:
    name = report["benchmark"]
    m = report["metrics"]
    p = report["paper"]
    print(f"\n{'=' * 72}")
    print(f"{name}  ({report['num_success']}/{report['num_images']} images)")
    print(f"{'Metric':<12} {'Ours':>10} {'Paper':>10} {'Delta':>10}")
    print("-" * 72)
    for key in ("mean", "median", "trimean", "best_25", "worst_25"):
        ours = m[key]
        paper = p[key]
        print(f"{key:<12} {ours:10.4f} {paper:10.4f} {ours - paper:+10.4f}")
    if report["num_failed"]:
        print(f"Failed: {report['num_failed']} (see benchmark_summary.json)")


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-CC paper benchmark reproduction")
    parser.add_argument(
        "--benchmark", choices=list(BENCHMARKS.keys()) + ["all"], default="all",
    )
    parser.add_argument(
        "-b", "--base_model",
        default="/mnt/disk1/qinqian/hf_home/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4",
    )
    parser.add_argument("--output_root", default="outputs/benchmark")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--start_angle", type=float, default=5.0)
    parser.add_argument("--end_angle", type=float, default=0.1)
    parser.add_argument("--angle_reduction_factor", type=float, default=0.5)
    parser.add_argument(
        "-t", "--iters", "--iterations", "--max_iterations",
        dest="max_iterations", type=int, default=20,
        help="Max iterations per image (0 = until convergence)",
    )
    parser.add_argument("--max_images", type=int, default=None, help="Debug: limit image count")
    parser.add_argument(
        "--lora", default=None,
        help="Override LoRA checkpoint path (default: models/*_lora for each benchmark)",
    )
    parser.add_argument("--use_gw_first", "--gw", dest="use_gw_first", action="store_true", default=True,
                        help="Gray-world first step (on by default)")
    parser.add_argument("--nogw", "--no_use_gw_first", dest="use_gw_first", action="store_false",
                        help="Disable the gray-world first step")
    parser.add_argument("--img_size", type=int, default=256, help="Inference image size (default: 256)")
    args = parser.parse_args()

    names = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    all_reports = []
    for name in names:
        out = Path(args.output_root) / name.lower().replace("+", "plus")
        report = run_benchmark(
            name,
            args.base_model,
            out,
            args.dtype,
            args.max_images,
            args.start_angle,
            args.end_angle,
            args.angle_reduction_factor,
            args.max_iterations,
            Path(args.lora) if args.lora else None,
            args.use_gw_first,
            args.img_size,
        )
        print_report(report)
        all_reports.append(report)

    summary_path = Path(args.output_root) / "all_benchmarks.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {summary_path}")


if __name__ == "__main__":
    main()
