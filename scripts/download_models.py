#!/usr/bin/env python3
"""Download VLM-CC LoRA weights from HuggingFace into models/.

The large adapter weights (`adapter_model.safetensors`, ~103 MB each) are not stored in
git — they live in a HuggingFace model repo. The small config/tokenizer files are kept in
git, so this script only fetches the missing weight files.

Usage:
    python scripts/download_models.py                  # all adapters
    python scripts/download_models.py --adapter nus-cube-inter_qwen2.5vl-7b
    HF_REPO_ID=other-user/vlm-cc python scripts/download_models.py

Set the repo id via --repo or the HF_REPO_ID env var (default below).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Update this to your HuggingFace model repo, or pass --repo / set HF_REPO_ID.
DEFAULT_REPO_ID = "PeanutBrain/vlm-cc"

ADAPTERS = [
    "nus-cube-inter_qwen2.5vl-7b",      # Gehler eval (Qwen2.5-VL-7B)
    "gehler-cube-inter_qwen2.5vl-7b",   # NUS-8 eval  (Qwen2.5-VL-7B)
    "nus-gehler-inter_qwen2.5vl-7b",    # Cube+ eval  (Qwen2.5-VL-7B)
    "nus-cube-inter_internvl3.5-1b",    # Gehler eval (InternVL3.5-1B, efficient option)
]
WEIGHT_FILE = "adapter_model.safetensors"

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VLM-CC LoRA weights from HuggingFace")
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID),
        help="HuggingFace model repo id holding the adapter subfolders",
    )
    parser.add_argument(
        "--adapter",
        choices=ADAPTERS + ["all"],
        default="all",
        help="Which adapter to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the weight file already exists",
    )
    args = parser.parse_args()

    print(f"Downloading from HuggingFace repo: {args.repo}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    targets = ADAPTERS if args.adapter == "all" else [args.adapter]
    for name in targets:
        dest = MODELS_DIR / name / WEIGHT_FILE
        if dest.exists() and not args.force:
            print(f"[skip] {name}/{WEIGHT_FILE} already present")
            continue
        print(f"[get ] {name}/{WEIGHT_FILE}  <-  {args.repo}")
        hf_hub_download(
            repo_id=args.repo,
            filename=f"{name}/{WEIGHT_FILE}",
            local_dir=str(MODELS_DIR),
        )
        print(f"[ok  ] {dest}")

    print("\nDone.")


if __name__ == "__main__":
    main()
