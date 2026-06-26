#!/usr/bin/env python3
"""Download VLM-CC LoRA adapters from HuggingFace into models/.

The adapter directories (weights + tokenizer/config) are hosted on a HuggingFace model repo,
not in git. This script fetches the full adapter folders into `models/<adapter>/`.

Usage:
    python scripts/download_models.py                  # all adapters
    python scripts/download_models.py --adapter nus-cube-inter_internvl3.5-1b
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
    parser = argparse.ArgumentParser(description="Download VLM-CC LoRA adapters from HuggingFace")
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
        help="Re-download even if the adapter is already present",
    )
    args = parser.parse_args()

    print(f"Downloading from HuggingFace repo: {args.repo}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed. Run: pip install huggingface_hub")

    targets = ADAPTERS if args.adapter == "all" else [args.adapter]
    for name in targets:
        if (MODELS_DIR / name / WEIGHT_FILE).exists() and not args.force:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}  <-  {args.repo}")
        snapshot_download(
            repo_id=args.repo,
            allow_patterns=f"{name}/*",
            local_dir=str(MODELS_DIR),
        )
        print(f"[ok  ] {MODELS_DIR / name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
