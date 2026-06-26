# VLM-CC: White-Balance First, Adjust Later

**Cross-Camera Color Constancy via Vision-Language Evaluation** — CVPR 2026

[Shuwei Li](https://nothingiknow.github.io), [Lei Tan](https://stone96123.github.io), [Robby T. Tan](https://tanrobby.github.io/) · National University of Singapore

[![Paper](https://img.shields.io/badge/CVPR-2026-blue)](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_White-Balance_First_Adjust_Later_Cross-Camera_Color_Constancy_via_Vision-Language_Evaluation_CVPR_2026_paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2605.19613-b31b1b)](https://arxiv.org/abs/2605.19613)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## 1. 📖 Overview

Learning-based color constancy methods tend to overfit to the spectral response of the
**training camera**, so they generalize poorly to unseen sensors. VLM-CC reframes the problem
as **vision-language evaluation**: rather than regressing the illuminant RGB directly, a
vision-language model judges the dominant **color-cast direction** (Red / Green / Blue) of a
tentatively white-balanced image, and an iterative loop adjusts the illuminant estimate until
the residual cast vanishes — *white-balance first, adjust later*.

```
predict direction → update illuminant → apply white balance → re-evaluate → … (until converged)
```

This VLM-based, evaluation-driven formulation transfers across cameras, reaching or
surpassing prior cross-camera results on the standard benchmarks. The method is implemented as a
LoRA fine-tune on top of **Qwen2.5-VL** (or the lightweight **InternVL3.5-1B**), built on
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with a 16-bit image-precision patch
(`src/llamafactory/data/mm_plugin.py`) that preserves RAW/sRGB precision through the pipeline.

> **Note.** We first release a **simplified evaluation pipeline** (without the color prior) to
> make reproduction easy — it already performs close to the numbers reported in the paper, and
> the full pipeline will follow soon. The original VLM-CC (Qwen2.5-VL-7B) is heavy, but the
> **lightweight InternVL3.5-1B version needs only ~3.0 GB of GPU memory** — give it a try (see
> [§5.2 Limited GPU memory](#52-limited-gpu-memory)).

## 2. 📊 Dataset

VLM-CC uses four standard color-constancy datasets: **NUS-8**, **Cube+**, **Inter-tau**, and
**Gehler** (ColorChecker). Models are trained in a **leave-one-out** (cross-dataset) protocol:
train on three datasets, evaluate zero-shot on the held-out one.


## 3. ⚙️ Requirements

- Python ≥ 3.9 (tested on 3.10)
- PyTorch with a matching CUDA toolkit (tested on PyTorch 2.8 / CUDA 12.x)
- GPU memory (peak, measured at inference):
  - **Qwen2.5-VL-7B** — ~17 GB (bfloat16) / ~34 GB (float32)
  - **InternVL3.5-1B** — **~3 GB** (bfloat16); fits a single ≥ 4 GB GPU

  Low on memory? See [5.2 Limited GPU memory](#52-limited-gpu-memory).

```bash
cd VLM-CC
pip install -e .
pip install opencv-python   # required for the 16-bit image pipeline
```

`requirements.txt` pins the core dependencies (transformers, peft, scipy, opencv-python, …).

## 4. 🎯 Pretrained Models

The released LoRA adapters are named by their **training datasets and
backbone** (N = NUS-8, C = Cube+, I = Inter-tau, G = Gehler). Each is evaluated zero-shot on the
held-out benchmark.

| Eval benchmark | Adapter directory (= training data) | Backbone | Base model |
|----------------|-------------------------------------|----------|-----------|
| Gehler | `nus-cube-inter_qwen2.5vl-7b/` | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Cube+  | `nus-gehler-inter_qwen2.5vl-7b/` | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| NUS-8  | `gehler-cube-inter_qwen2.5vl-7b/` | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Gehler (efficient) | `nus-cube-inter_internvl3.5-1b/` | InternVL3.5-1B | `OpenGVLab/InternVL3_5-1B-HF` |

The weight files are hosted on HuggingFace ([`PeanutBrain/vlm-cc`](https://huggingface.co/PeanutBrain/vlm-cc)),
not in git. Fetch them after cloning:

```bash
pip install huggingface_hub
python scripts/download_models.py            # all adapters
# python scripts/download_models.py --adapter nus-cube-inter_internvl3.5-1b
```

> The **InternVL3.5-1B** adapter is a lightweight alternative for Gehler — a model **7× smaller**
> (**~3 GB** vs ~17 GB GPU memory in bfloat16) that still lands around the paper numbers. LoRA is loaded separately via `--lora`; the
> `base_model_name_or_path` in `adapter_config.json` is **not** used automatically, so always
> match each adapter with the base model in the table.

## 5. 🚀 Usage

### 5.1 Quick demo

Run the iterative white balance on the two bundled sample images (no ground truth required):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/direction_iteration.py \
    --lora models/nus-cube-inter_qwen2.5vl-7b \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --input assets/smoke_test \
    --output outputs/eval \
    --iters 0
```

Results go to `outputs/eval/results_<stem>.json` (resolved camera under `"camera"`).

<details>
<summary><b>Full <code>direction_iteration.py</code> CLI flags</b></summary>

| Flag | Long form | Default | Description |
|------|-----------|---------|-------------|
| `-l` | `--lora` | *(required)* | LoRA directory |
| `-i` | `--input` | *(required)* | Image file or folder |
| `-o` | `--output` | `outputs/eval` | Output directory |
| `-b` | `--base` | `Qwen/Qwen2.5-VL-7B-Instruct` | Base VLM (HF id or local path) |
| `-m` | `--matrix` | `data/matrices.json` | CCM matrix file |
| `-t` | `--iters` | `0` | Max steps; `0` = converge |
| `--nogw` | `--no_gw_first` | gw **on** by default | Disable the gray-world first step |
| `--cam` | `--camera` | auto | Override camera type |
| `--dtype` | | `float32` | `float32` (~34 GB) or `bfloat16` (~17 GB) |
| `--save-inter` | | off | Save per-iteration PNGs |
| `--save-final` | | off | Save final smoothed PNG |
| | `--start_angle` / `--end_angle` / `--angle_reduction_factor` | `5` / `0.1` / `0.5` | Rotation schedule |
| | `--img_size` | `256` | Resize for the WB pipeline |
</details>

<details>
<summary><b>Camera-type detection</b></summary>

Each image needs a **camera type** for black/saturation calibration and the matching CCM in
`matrices.json`. By default the script auto-detects it from the file path and name, following the
standard naming convention of each dataset — so the bundled samples and the benchmark datasets
work out of the box.

To override it (e.g. for your own images), pass `--cam <type>`. Valid values:
- **Gehler:** `canon1d`, `canon5d`
- **Cube+:** `Canon_EOS_550D`
- **NUS-8:** `Canon1DsMkIII`, `Canon600D`, `FujifilmXM1`, `NikonD5200`, `OlympusEPL6`, `PanasonicGX1`, `SamsungNX2000`, `SonyA57`
</details>


### 5.2 Limited GPU memory

If you don't have a large GPU, switch to the **InternVL3.5-1B** model with **bfloat16** and a
smaller `--img_size`. This runs in **~3 GB** of GPU memory (fits a ≥ 4 GB card) — vs ~17 GB for
the Qwen2.5-VL-7B model — with only a small accuracy cost:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/direction_iteration.py \
    --lora models/nus-cube-inter_internvl3.5-1b \
    --base OpenGVLab/InternVL3_5-1B-HF \
    --input assets/smoke_test --output outputs/eval \
    --dtype bfloat16 --img_size 256 --iters 30
```

Knobs that lower memory / speed up, in order of impact: `--dtype bfloat16` (halves the 7B from
~34 GB → ~17 GB), the InternVL3.5-1B model (~3 GB), and a smaller `--img_size` (e.g. `256`).
The same `--lora` / `--base_model` / `--dtype` / `--img_size` flags apply to
`benchmark_reproduce.py` for full-dataset runs.

### 5.3 Evaluation / benchmark reproduction

`scripts/eval/benchmark_reproduce.py` runs a full test set and reports angular-error statistics
against the paper. `--benchmark` accepts `Gehler`, `NUS8`, `Cube+`, or `all`; a
`benchmark_summary.json` is written under `--output_root`.

```bash
# Qwen2.5-VL-7B
CUDA_VISIBLE_DEVICES=0 python scripts/eval/benchmark_reproduce.py \
    --benchmark Gehler \
    --lora models/nus-cube-inter_qwen2.5vl-7b \
    --start_angle 3 --iterations 30 --img_size 512 --dtype bfloat16 \
    --output_root outputs/benchmark/gehler

# InternVL3.5-1B (efficient) — also pass the InternVL base model
CUDA_VISIBLE_DEVICES=0 python scripts/eval/benchmark_reproduce.py \
    --benchmark Gehler \
    --lora models/nus-cube-inter_internvl3.5-1b \
    --base_model OpenGVLab/InternVL3_5-1B-HF \
    --start_angle 3 --iterations 30 --img_size 512 --dtype bfloat16 \
    --output_root outputs/benchmark/gehler_internvl
```

The eval auto-selects the chat template from the base model (`qwen2_vl` / `intern_vl`).

## 6. 🙏 Acknowledgements

Built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (the training/inference
framework; the only modification is the 16-bit `mm_plugin` patch). We also thank the authors of
[CCMNet](https://github.com/DY112/CCMNet) for their work on cross-camera color constancy, which
inspired parts of this project. We thank the authors of the NUS-8, Cube+, Inter-tau, and Gehler
ColorChecker datasets.

## 7. 📚 Citation

```bibtex
@InProceedings{Li_2026_CVPR,
    author    = {Li, Shuwei and Tan, Lei and Tan, Robby T.},
    title     = {White-Balance First, Adjust Later: Cross-Camera Color Constancy via Vision-Language Evaluation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {1331-1341}
}
```

## 8. 📄 License

Apache-2.0, inherited from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). The VLM-CC
additions (16-bit `mm_plugin` patch, direction/iteration eval, configs) are released under the
same license. See [LICENSE](LICENSE).
