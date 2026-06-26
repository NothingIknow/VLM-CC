---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-VL-7B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
- color-constancy
- white-balance
model-index:
- name: nus-cube-inter_qwen2.5vl-7b
  results: []
---

# VLM-CC · Qwen2.5-VL-7B direction LoRA (NUS-8, Cube+ and Inter-tau)

A LoRA adapter for [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
trained for **illuminant-direction estimation** (color constancy / auto white balance) as part of
the [VLM-CC](https://github.com/NothingIknow/VLM-CC) project. Given an image, the model predicts the
dominant direction (Red / Green / Blue) of the scene illuminant, which is then refined through an
iterative white-balance loop.

This is a **cross-dataset** model: trained on **NUS-8, Cube+ and Inter-tau** and intended for zero-shot evaluation on
the held-out **Gehler** benchmark.

## Intended uses & limitations

- **Use:** illuminant estimation / auto white balance on RAW-derived 16-bit sRGB images, through the
  VLM-CC iterative inference pipeline.
- **Requires** the base model `Qwen/Qwen2.5-VL-7B-Instruct` plus the VLM-CC eval code, which applies
  the `qwen2_vl` chat template and a 16-bit image pipeline. Loading the bare adapter outside that
  pipeline will not reproduce the reported behaviour.

## Training data

NUS-8, Cube+ and Inter-tau color-constancy datasets (16-bit sRGB, illuminant-direction labels).

## Training procedure

LoRA (rank 8) applied to all linear layers, including the vision tower. Trained in **fp32** to
preserve the 16-bit image precision through the pipeline.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0004
- train_batch_size: 52
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 832
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- precision: fp32

### Framework

Trained with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (PEFT / LoRA).
