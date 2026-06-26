# Pretrained adapters

The VLM-CC LoRA adapters are hosted on HuggingFace, not in git. Download them here with:

```bash
python scripts/download_models.py            # all adapters
# python scripts/download_models.py --adapter nus-cube-inter_internvl3.5-1b
```

This populates `models/<adapter>/` for each released adapter. See the main
[README](../README.md) (§4 Pretrained Models) for the list and usage.
