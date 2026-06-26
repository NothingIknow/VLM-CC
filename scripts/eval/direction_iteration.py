#!/usr/bin/env python3
"""Iterative direction-based white balance inference."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

VLM_CC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VLM_CC_ROOT / "src"))
sys.path.insert(0, str(VLM_CC_ROOT / "visualization" / "color_space"))

from llamafactory.chat import ChatModel
from visualize_trace import visualize_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PROMPT = """**Task** Look at the attached image and decide which primary colour channel-Red, Green, or Blue-dominates the illuminant.
**Private reasoning steps**
1. **Locate neutral references** Mentally pick up to three regions that should be colour-neutral (e.g., grey concrete, white clouds, metal railings). Avoid vividly coloured or saturated areas, deep shadows, and highlights.
2. **Compare channel strengths** For each selected region, judge the average R, G, B intensities and note which channel appears strongest.
3. **Aggregate** Combine evidence across the chosen regions to identify the single channel with the consistently highest relative intensity.
4. **Think silently** Keep all deliberation internal; do **not** reveal your reasoning.

**Answer format** Reply with exactly one of the tokens below (including the period) and nothing else:
- Red.
- Green.
- Blue."""

NUS8_CAMERAS = {
    "Canon1DsMkIII", "Canon600D", "FujifilmXM1", "NikonD5200",
    "OlympusEPL6", "PanasonicGX1", "SamsungNX2000", "SonyA57",
}
GEHLER_CALIB = {
    "canon1d": (0, 3588),
    "canon5d": (128, 3650),
}
CUBEPLUS_CC = np.array([0.6, 1.0, 28 / 37.5, 1.0], dtype=np.float32)
CUBEPLUS_DARK = np.array([2048, 2048, 2048], dtype=np.float32)
CUBEPLUS_SAT = np.ones(3, dtype=np.float32)
NUS8_DARK = np.zeros(3, dtype=np.float32)
NUS8_SAT = np.full(3, 65535.0, dtype=np.float32)
MATRIX_CANDIDATES = (
    VLM_CC_ROOT / "data" / "matrices.json",
    VLM_CC_ROOT / "data" / "cam2srgb_matrices.json",
    VLM_CC_ROOT / "assets" / "cam2srgb_matrices.json",
)
VALID_CAMERAS = set(NUS8_CAMERAS) | set(GEHLER_CALIB.keys()) | {"Canon_EOS_550D"}
GEHLER_GT_MAT = "/mnt/disk3/shuwei/Gehler/gehler_gt_aligned.mat"
CUBEPLUS_GT_MAT = "/mnt/disk3/shuwei/Cube+/gt_file/ground_truth.mat"


def angular_error(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred / np.linalg.norm(pred)
    gt = gt / np.linalg.norm(gt)
    return float(np.degrees(np.arccos(np.clip(np.dot(pred, gt), -1.0, 1.0))))


def _as_vector3(value, dtype=np.float32) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value.astype(dtype)
    elif isinstance(value, (list, tuple)):
        arr = np.array(value, dtype=dtype)
    else:
        arr = np.full(3, value, dtype=dtype)
    return arr.squeeze()


def geometric_mean_illumination(illums: List[np.ndarray]) -> np.ndarray:
    logs = sum(np.log(np.maximum(i, 1e-9)) for i in illums) / len(illums)
    out = np.exp(logs)
    return out / np.linalg.norm(out)


def rotation_angle(
    iteration: int,
    max_iterations: Optional[int],
    start_angle: float,
    end_angle: float,
) -> float:
    if not max_iterations or max_iterations <= 1:
        progress = min(iteration / 19.0, 1.0)
    else:
        progress = min(max(iteration / (max_iterations - 1), 0.0), 1.0)
    return start_angle - progress * (start_angle - end_angle)


def rotate_illumination(current: np.ndarray, color_bias: str, angle_deg: float) -> np.ndarray:
    axis = {"Red.": 0, "Green.": 1, "Blue.": 2}.get(color_bias)
    if axis is None:
        return current.copy()

    u = current / np.linalg.norm(current)
    v = np.zeros(3, dtype=np.float32)
    v[axis] = 1.0
    r = v - np.dot(u, v) * u
    s = np.linalg.norm(r)
    if s < 1e-8:
        return u.copy()

    a = np.radians(angle_deg)
    out = np.cos(a) * u + np.sin(a) * (r / s)
    return out / np.linalg.norm(out)


class StepTimer:
    """Accumulate wall times for named steps (seconds)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.steps: Dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.steps[name] = self.steps.get(name, 0.0) + seconds

    @contextmanager
    def block(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def summary(self, num_iterations: int, num_predict_calls: int) -> Dict[str, float]:
        total = self.steps.get("total", 0.0)
        out = {k: round(v, 4) for k, v in self.steps.items()}
        out["num_iterations"] = num_iterations
        out["num_predict_calls"] = num_predict_calls
        if num_predict_calls and "predict" in self.steps:
            out["predict_per_call"] = round(self.steps["predict"] / num_predict_calls, 4)
        if total > 0:
            for key in ("setup", "apply_wb", "predict", "rotate", "save", "other"):
                if key in self.steps:
                    out[f"{key}_pct"] = round(100.0 * self.steps[key] / total, 1)
        return out

    def log_summary(self, image_name: str, num_iterations: int, num_predict_calls: int) -> None:
        if not self.enabled:
            return
        s = self.summary(num_iterations, num_predict_calls)
        logger.info(
            "Timing %s: total=%.3fs iters=%d predict=%d (%.3fs/call) | "
            "setup=%.3fs apply_wb=%.3fs predict=%.3fs rotate=%.3fs save=%.3fs",
            image_name,
            s.get("total", 0.0),
            num_iterations,
            num_predict_calls,
            s.get("predict_per_call", 0.0),
            s.get("setup", 0.0),
            s.get("apply_wb", 0.0),
            s.get("predict", 0.0),
            s.get("rotate", 0.0),
            s.get("save", 0.0),
        )


def collect_image_files(input_path: Path, formats: List[str]) -> List[str]:
    if input_path.is_file():
        return [str(input_path)] if input_path.suffix.lower() in {f.lower() for f in formats} else []
    files: List[str] = []
    for fmt in formats:
        files.extend(str(p) for p in input_path.glob(f"*{fmt.lower()}"))
        files.extend(str(p) for p in input_path.glob(f"*{fmt.upper()}"))
    return sorted(set(files))


class DirectionTester:
    def __init__(
        self,
        model_path: Optional[str],
        base_model: Optional[str] = None,
        matrix_file: Optional[str] = None,
        infer_dtype: str = "float32",
        template: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.base_model = base_model or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.infer_dtype = infer_dtype
        # Pick the chat template from the base model unless overridden.
        self.template = template or (
            "intern_vl" if "intern" in self.base_model.lower() else "qwen2_vl"
        )
        self.matrix_file = self._resolve_matrix_file(matrix_file)
        self.chat_model: Optional[ChatModel] = None
        self._rgb_cam_cache: Dict[str, np.ndarray] = {}
        self._timer: Optional[StepTimer] = None

    def _resolve_matrix_file(self, override: Optional[str]) -> str:
        if override and os.path.exists(override):
            return override
        for path in MATRIX_CANDIDATES:
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"Matrix file not found under {VLM_CC_ROOT}")

    def _camera_matrix(self, camera_name: str) -> np.ndarray:
        if camera_name in self._rgb_cam_cache:
            return self._rgb_cam_cache[camera_name]
        with open(self.matrix_file, encoding="utf-8") as f:
            matrices = json.load(f)
        if camera_name not in matrices:
            raise ValueError(f"Unknown camera in matrix file: {camera_name}")
        cam_rgb = np.array(matrices[camera_name]) @ np.array(matrices["xyz_rgb"])
        cam_rgb = cam_rgb / cam_rgb.sum(axis=1, keepdims=True)
        rgb_cam = np.linalg.inv(cam_rgb).astype(np.float32)
        self._rgb_cam_cache[camera_name] = rgb_cam
        return rgb_cam

    @staticmethod
    def camera_from_path(image_path: str) -> str:
        basename = os.path.basename(image_path)
        stem = os.path.splitext(basename)[0]
        parent = os.path.basename(os.path.dirname(image_path))

        if parent in GEHLER_CALIB:
            return parent
        if basename.startswith("IMG_"):
            return "canon5d"
        if re.match(r"^[0-9][A-Z0-9]+", stem) and not stem.isdigit():
            return "canon1d"
        if stem.isdigit():
            return "Canon_EOS_550D"
        if "_" in stem:
            camera = stem.split("_", 1)[0]
            if camera in NUS8_CAMERAS:
                return camera
        raise ValueError(f"Cannot infer camera from filename: {basename}")

    def resolve_camera(self, image_path: str, camera_override: Optional[str] = None) -> str:
        if camera_override:
            if camera_override not in VALID_CAMERAS:
                raise ValueError(
                    f"Unknown camera: {camera_override}. "
                    f"Valid values: {', '.join(sorted(VALID_CAMERAS))}"
                )
            return camera_override
        return self.camera_from_path(image_path)

    def get_camera_params(
        self,
        image_path: str,
        camera_override: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
        """Return black level, saturation, optional color-chart mask, and camera name."""
        camera = self.resolve_camera(image_path, camera_override)

        if camera in GEHLER_CALIB:
            dark_val, sat_val = GEHLER_CALIB[camera]
            return _as_vector3(dark_val), _as_vector3(sat_val), None, camera

        if camera == "Canon_EOS_550D":
            return CUBEPLUS_DARK.copy(), CUBEPLUS_SAT.copy(), CUBEPLUS_CC.copy(), camera

        return NUS8_DARK.copy(), NUS8_SAT.copy(), None, camera

    def read_linear(
        self,
        path: str,
        black_level,
        saturation_level,
        cc_coords,
        target_size: Optional[int] = 512,
        use_mask: bool = True,
    ) -> np.ndarray:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(raw.astype(np.float32), cv2.COLOR_BGR2RGB) - black_level
        img /= 65535.0 if raw.dtype == np.uint16 else (255.0 if img.max() > 1.5 else 1.0)
        img = np.clip(img, 0, saturation_level) / (np.max(img) + 1e-9)

        if use_mask and cc_coords is not None:
            if all(float(c) <= 1.0 for c in cc_coords):
                h, w = img.shape[:2]
                y0, y1, x0, x1 = [int(float(c) * dim) for c, dim in zip(cc_coords, (h, h, w, w))]
            else:
                y0, y1, x0, x1 = map(int, cc_coords)
            img[y0:y1, x0:x1] = 0.0

        if target_size is not None:
            h, w = img.shape[:2]
            if h < w:
                new_h, new_w = target_size, int(w * target_size / h)
            else:
                new_w, new_h = target_size, int(h * target_size / w)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    def linear_to_srgb(self, img: np.ndarray, camera_name: str) -> np.ndarray:
        img = np.clip(img / (np.max(img) + 1e-9), 0.0, 1.0)
        flat = img.reshape(-1, 3)
        corrected = np.clip(flat @ self._camera_matrix(camera_name).T, 0.0, 1.0).reshape(img.shape)
        mask = corrected <= 0.0031308
        srgb = np.empty_like(corrected)
        srgb[mask] = 12.92 * corrected[mask]
        srgb[~mask] = 1.055 * np.power(corrected[~mask], 1 / 2.4) - 0.055
        return np.clip(srgb, 0, 1)

    def _pil_from_path(self, path: str) -> Image.Image:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw.dtype == np.uint16:
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
        else:
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil._float32_data = img
        pil._is_16bit = raw.dtype == np.uint16
        return pil

    def _pil_from_srgb(self, srgb: np.ndarray) -> Image.Image:
        pil = Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))
        pil._float32_data = srgb.astype(np.float32)
        pil._is_16bit = True
        return pil

    def load_model(self) -> bool:
        try:
            args = {
                "model_name_or_path": self.base_model,
                "template": self.template,
                "infer_backend": "huggingface",
                "infer_dtype": self.infer_dtype,
                "trust_remote_code": True,
                "max_new_tokens": 512,
                "temperature": 1e-6,
                "top_p": 0.9,
            }
            if self.model_path:
                args["adapter_name_or_path"] = self.model_path
                args["finetuning_type"] = "lora"
            self.chat_model = ChatModel(args)
            return True
        except Exception as exc:
            logger.error("Model load failed: %s", exc)
            return False

    def predict(self, image: Union[str, Image.Image]) -> str:
        if self.chat_model is None:
            raise RuntimeError("Model not loaded")
        if isinstance(image, str):
            with self._timer_block("predict_prep"):
                image = self._pil_from_path(image)
        with self._timer_block("predict"):
            response = self.chat_model.chat([{"role": "user", "content": PROMPT}], images=[image])
        return response[0].response_text.strip()

    @contextmanager
    def _timer_block(self, name: str) -> Iterator[None]:
        timer = self._timer
        if timer is None:
            yield
            return
        with timer.block(name):
            yield

    def apply_wb(
        self,
        linear_img: np.ndarray,
        illum: np.ndarray,
        camera_name: str,
        return_pil: bool = False,
    ):
        gain = illum / illum[1]
        srgb = self.linear_to_srgb(linear_img / gain, camera_name)
        return self._pil_from_srgb(srgb) if return_pil else srgb

    def save_wb_png(
        self,
        linear_img: np.ndarray,
        illum: np.ndarray,
        camera_name: str,
        save_path: str,
    ) -> str:
        srgb = self.apply_wb(linear_img, illum, camera_name, return_pil=False)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img_16bit = (np.clip(srgb, 0, 1) * 65535).astype(np.uint16)
        cv2.imwrite(save_path, cv2.cvtColor(img_16bit, cv2.COLOR_RGB2BGR))
        return save_path

    def gray_world_illum(
        self,
        image_path: str,
        img_size: Optional[int],
        camera_override: Optional[str] = None,
    ) -> np.ndarray:
        dark, sat, cc, _camera = self.get_camera_params(image_path, camera_override)
        linear = self.read_linear(image_path, dark, sat, cc, target_size=img_size, use_mask=True)
        gw = np.mean(linear, axis=(0, 1))
        return gw / np.linalg.norm(gw)

    def run_iteration_test(
        self,
        image_path: str,
        max_iterations: Optional[int] = 0,
        output_dir: str = "outputs/eval",
        use_gw_first: bool = False,
        start_angle: float = 5.0,
        end_angle: float = 1.0,
        angle_reduction_factor: float = 0.5,
        img_size: Optional[int] = 256,
        save_intermediate_images: bool = False,
        save_final_image: bool = False,
        camera_override: Optional[str] = None,
        save_json: bool = True,
        save_trace: bool = True,
        profile_timing: bool = False,
    ) -> Dict:
        timer = StepTimer(enabled=profile_timing)
        self._timer = timer
        t_total = time.perf_counter()
        try:
            with timer.block("setup"):
                os.makedirs(output_dir, exist_ok=True)
                image_name = os.path.basename(image_path)
                dark, sat, cc, camera = self.get_camera_params(image_path, camera_override)
                linear = self.read_linear(image_path, dark, sat, cc, target_size=img_size, use_mask=True)

            current_illum = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3)
            current_image: Optional[Image.Image] = None
            recent_biases: List[str] = []
            recent_illums: List[np.ndarray] = []
            cur_start, cur_end = start_angle, end_angle
            first_rgb_set = None
            angle_halved_at = None

            results = {
                "image_name": image_name,
                "camera": camera,
                "iterations": [],
            }

            iteration = 0
            num_predict_calls = 0
            unlimited = max_iterations is None or max_iterations <= 0
            while unlimited or iteration < max_iterations:
                try:
                    if iteration == 0 and use_gw_first:
                        with timer.block("apply_wb"):
                            new_illum = self.gray_world_illum(image_path, img_size, camera_override)
                        color_bias = "Gray World"
                        angle = 0.0
                    else:
                        if current_image is None:
                            with timer.block("apply_wb"):
                                current_image = self.apply_wb(linear, current_illum, camera, return_pil=True)
                        color_bias = self.predict(current_image)
                        num_predict_calls += 1
                        recent_biases = (recent_biases + [color_bias])[-3:]
                        with timer.block("rotate"):
                            angle = rotation_angle(iteration, max_iterations, cur_start, cur_end)
                            new_illum = rotate_illumination(current_illum, color_bias, angle)

                    recent_illums = (recent_illums + [new_illum.copy()])[-3:]
                    with timer.block("apply_wb"):
                        current_image = self.apply_wb(linear, new_illum, camera, return_pil=True)
                    current_illum = new_illum

                    results["iterations"].append({
                        "iteration": iteration + 1,
                        "color_bias": color_bias,
                        "rotation_angle": float(angle),
                        "pred_illum": [float(x) for x in new_illum.tolist()],
                    })
                    if save_intermediate_images:
                        with timer.block("save"):
                            iter_path = self.save_wb_png(
                                linear,
                                new_illum,
                                camera,
                                os.path.join(output_dir, f"iteration_{iteration + 1:02d}_{image_name}"),
                            )
                        results["iterations"][-1]["image_path"] = iter_path

                    biases = {b for b in recent_biases if b != "Gray World"}
                    if len(recent_biases) == 3 and len(biases) == 3:
                        if first_rgb_set is None:
                            first_rgb_set = biases
                            cur_start *= angle_reduction_factor
                            cur_end *= angle_reduction_factor
                            angle_halved_at = iteration + 1
                            recent_biases = []
                        else:
                            results["converged"] = True
                            results["convergence_reason"] = "semantic_convergence_twice"
                            results["convergence_iteration"] = iteration + 1
                            results["angle_halved_at_iteration"] = angle_halved_at
                            break
                except Exception as exc:
                    logger.error("Iteration %d failed: %s", iteration + 1, exc)
                    break
                finally:
                    iteration += 1

            if results["iterations"]:
                results["final_illum"] = results["iterations"][-1]["pred_illum"]
            if len(recent_illums) >= 2:
                smoothed = geometric_mean_illumination(recent_illums)
                results["final_smoothed_illum"] = [float(x) for x in smoothed.tolist()]
            elif results["iterations"]:
                results["final_smoothed_illum"] = results["final_illum"]

            if "converged" not in results:
                results["converged"] = False
                results["convergence_reason"] = "no_convergence_or_terminated" if unlimited else "max_iterations"
                results["convergence_iteration"] = len(results["iterations"]) if unlimited else max_iterations
                if angle_halved_at is not None:
                    results["angle_halved_at_iteration"] = angle_halved_at

            with timer.block("save"):
                if save_final_image and "final_smoothed_illum" in results:
                    smoothed = np.array(results["final_smoothed_illum"], dtype=np.float32)
                    final_path = self.save_wb_png(
                        linear,
                        smoothed,
                        camera,
                        os.path.join(output_dir, f"final_smoothed_{image_name}"),
                    )
                    results["final_smoothed_image"] = final_path

                stem = os.path.splitext(image_name)[0]
                if save_json:
                    with open(os.path.join(output_dir, f"results_{stem}.json"), "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                if save_trace and results["iterations"]:
                    visualize_trace(
                        predictions=[tuple(r["pred_illum"]) for r in results["iterations"]],
                        ground_truth=None,
                        final_smoothed=tuple(results["final_smoothed_illum"]) if "final_smoothed_illum" in results else None,
                        iterations=[r["iteration"] for r in results["iterations"]],
                        output_path=Path(output_dir) / f"trace_results_{stem}.png",
                    )

            timer.add("total", time.perf_counter() - t_total)
            if profile_timing:
                timing = timer.summary(len(results["iterations"]), num_predict_calls)
                results["timing"] = timing
                timer.log_summary(image_name, len(results["iterations"]), num_predict_calls)
            return results
        finally:
            self._timer = None

    def run_batch_test(
        self,
        image_files: List[str],
        max_iterations: Optional[int] = 0,
        base_output_dir: str = "outputs/eval",
        use_gw_first: bool = False,
        start_angle: float = 5.0,
        end_angle: float = 1.0,
        angle_reduction_factor: float = 0.5,
        img_size: Optional[int] = 256,
        save_intermediate_images: bool = False,
        save_final_image: bool = False,
        camera_override: Optional[str] = None,
        profile_timing: bool = False,
    ) -> Dict:
        os.makedirs(base_output_dir, exist_ok=True)
        ok, fail = 0, 0
        batch_timer = StepTimer(enabled=profile_timing)
        total_predict_calls = 0
        for image_path in tqdm(image_files, desc="eval"):
            out_dir = os.path.join(base_output_dir, os.path.splitext(os.path.basename(image_path))[0])
            try:
                result = self.run_iteration_test(
                    image_path, max_iterations, out_dir, use_gw_first,
                    start_angle, end_angle, angle_reduction_factor, img_size,
                    save_intermediate_images, save_final_image,
                    camera_override,
                    save_json=not profile_timing,
                    save_trace=not profile_timing,
                    profile_timing=profile_timing,
                )
                if profile_timing and "timing" in result:
                    for key in ("setup", "apply_wb", "predict", "rotate", "save", "total"):
                        if key in result["timing"]:
                            batch_timer.add(key, float(result["timing"][key]))
                    total_predict_calls += int(result["timing"].get("num_predict_calls", 0))
                if result.get("iterations"):
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:
                fail += 1
                logger.error("%s: %s", os.path.basename(image_path), exc)
        logger.info("Done: %d/%d succeeded", ok, len(image_files))
        summary = {"total_images": len(image_files), "successful_tests": ok, "failed_tests": fail}
        if profile_timing and batch_timer.steps:
            n = max(ok, 1)
            logger.info(
                "Batch timing avg over %d images: total=%.3fs setup=%.3fs apply_wb=%.3fs "
                "predict=%.3fs (%.3fs/call, %d calls) rotate=%.3fs save=%.3fs",
                ok,
                batch_timer.steps.get("total", 0.0) / n,
                batch_timer.steps.get("setup", 0.0) / n,
                batch_timer.steps.get("apply_wb", 0.0) / n,
                batch_timer.steps.get("predict", 0.0) / n,
                batch_timer.steps.get("predict", 0.0) / max(total_predict_calls, 1),
                total_predict_calls,
                batch_timer.steps.get("rotate", 0.0) / n,
                batch_timer.steps.get("save", 0.0) / n,
            )
            summary["batch_timing"] = {k: round(v / n, 4) for k, v in batch_timer.steps.items()}
        return summary

    def cleanup(self) -> None:
        self.chat_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Backward-compatible aliases for smoke_test_reproduce.py
DirectionTester.load_direction_model = DirectionTester.load_model
DirectionTester.predict_color_bias = DirectionTester.predict
DirectionTester.get_gt_data = DirectionTester.get_camera_params
DirectionTester.extract_camera_from_filename = staticmethod(DirectionTester.camera_from_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM-CC direction iteration white balance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/eval/direction_iteration.py -l models/nus-cube-inter_qwen2.5vl-7b -i assets/smoke_test -o outputs/eval\n"
            "  python scripts/eval/direction_iteration.py -l models/nus-cube-inter_qwen2.5vl-7b -i img.png -o out \\\n"
            "    -b Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --cam canon5d --save-final\n"
            "\n"
            "BASE_MODEL (-b): HuggingFace ID (auto-download) or local model directory."
        ),
    )
    req = parser.add_argument_group("required")
    req.add_argument(
        "-l", "--lora", "--model_path",
        dest="model_path", required=True,
        help="LoRA checkpoint directory",
    )
    req.add_argument(
        "-i", "--input", "--input_path",
        dest="input_path", required=True,
        help="Input image or folder",
    )

    io = parser.add_argument_group("io")
    io.add_argument(
        "-o", "--output", "--output_dir",
        dest="output_dir", default="outputs/eval",
        help="Output directory",
    )
    io.add_argument(
        "-b", "--base", "--base_model",
        dest="base_model", default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Base VLM: HuggingFace ID or local path",
    )
    io.add_argument(
        "-m", "--matrix", "--matrix_file",
        dest="matrix_file", default=None,
        help="CCM matrix JSON (default: data/matrices.json)",
    )

    algo = parser.add_argument_group("algorithm")
    algo.add_argument("-t", "--iters", "--iterations", dest="iterations", type=int, default=0,
                      help="Max iterations; 0 = until convergence")
    algo.add_argument("--gw", "--use_gw_first", dest="use_gw_first", action="store_true", default=True,
                      help="Start with gray-world illumination (on by default)")
    algo.add_argument("--nogw", "--no_gw_first", dest="use_gw_first", action="store_false",
                      help="Disable the gray-world first step")
    algo.add_argument("--start_angle", type=float, default=5.0)
    algo.add_argument("--end_angle", type=float, default=0.1)
    algo.add_argument("--angle_reduction_factor", type=float, default=0.5)
    algo.add_argument("--img_size", type=int, default=256)
    algo.add_argument("--cam", "--camera", dest="camera", default=None,
                      help="Override auto-detected camera type")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--dtype", "--infer_dtype",
        dest="infer_dtype", choices=["float32", "bfloat16"], default="float32",
        help="Inference dtype (float32 ~34GB, bfloat16 ~17GB)",
    )
    runtime.add_argument(
        "--save-inter", "--save_intermediate_images",
        dest="save_intermediate_images", action="store_true",
        help="Save iteration_{NN}_{name}.png per step",
    )
    runtime.add_argument(
        "--save-final", "--save_final_image",
        dest="save_final_image", action="store_true",
        help="Save final_smoothed_{name}.png",
    )
    runtime.add_argument(
        "--profile", "--profile_timing",
        dest="profile_timing", action="store_true",
        help="Log per-step timing (setup / apply_wb / predict / save)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.camera and args.camera not in VALID_CAMERAS:
        raise SystemExit(
            f"Unknown --camera: {args.camera}. "
            f"Valid values: {', '.join(sorted(VALID_CAMERAS))}"
        )

    img_size = None if args.img_size == 0 else args.img_size
    max_iterations = None if args.iterations <= 0 else args.iterations
    image_files = collect_image_files(Path(args.input_path), [".png", ".jpg", ".jpeg"])
    if not image_files:
        raise SystemExit(f"No images found: {args.input_path}")

    tester = DirectionTester(
        args.model_path,
        base_model=args.base_model,
        matrix_file=args.matrix_file or str(VLM_CC_ROOT / "data" / "matrices.json"),
        infer_dtype=args.infer_dtype,
    )
    try:
        if not tester.load_model():
            raise SystemExit(1)
        tester.run_batch_test(
            image_files, max_iterations, args.output_dir, args.use_gw_first,
            args.start_angle, args.end_angle, args.angle_reduction_factor, img_size,
            args.save_intermediate_images, args.save_final_image,
            args.camera, args.profile_timing,
        )
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
