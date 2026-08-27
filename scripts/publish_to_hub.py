"""Publish a trained checkpoint to Hugging Face Hub as a public model
repo, with a model card explaining what it is and how to load it.

TinyLoRA's checkpoint (a 13-number vector + a tiny config file) isn't a
format the standard push_to_hub() convenience methods understand, so this
script uses huggingface_hub's create_repo/upload_folder directly for all
5 methods uniformly, rather than mixing that with PeftModel.push_to_hub()
for some and manual uploads for others.

Usage:
    python scripts/publish_to_hub.py --method qlora --hf-username swathikchgithub
"""

import argparse
import json

from huggingface_hub import HfApi, create_repo

METHODS = ["full_ft", "lora", "lora_fa", "qlora", "tinylora"]

MODEL_CARD_TEMPLATE = """---
license: mit
base_model: {base_model}
tags:
- lora-methods-comparison
- {method}
---

# {title}

Part of [lora-methods-comparison](https://github.com/swathikchgithub/lora-methods-comparison)
— a rigorous, apples-to-apples comparison of Full fine-tuning, LoRA,
LoRA-FA, QLoRA, and TinyLoRA, trained on the same task (AG News topic
classification), same base model ({base_model}), same GPU, same
evaluation harness.

## This checkpoint

{method_description}

**Trainable parameters:** {trainable_params:,}
**Test accuracy:** {accuracy}
**Peak GPU memory during training:** {peak_mem} GB
**Training wall-clock time:** {train_time}s

## How to load

{load_instructions}

See the [repo README](https://github.com/swathikchgithub/lora-methods-comparison)
for the full comparison across all 5 methods, including methodology and
the complete results table.
"""

METHOD_DESCRIPTIONS = {
    "full_ft": "Standard full fine-tuning — every parameter in the base model was updated.",
    "lora": "Standard LoRA (`peft.LoraConfig`, rank 16, alpha 32) — low-rank adapters on attention and MLP projections, base weights frozen.",
    "lora_fa": "LoRA-FA — same as LoRA, but the `A` matrices are frozen at their random initialization; only the `B` matrices train, halving LoRA's trainable parameter count.",
    "qlora": "QLoRA — LoRA adapters on top of a 4-bit NF4-quantized frozen base model.",
    "tinylora": "TinyLoRA — a from-paper implementation of [arXiv:2602.04118](https://arxiv.org/abs/2602.04118): a single trainable vector, shared across every adapted layer in the whole model, projected through a fixed random tensor and combined with each layer's own frozen truncated-SVD basis. Not from the `peft` library — implemented directly from the paper's mechanism.",
}

LOAD_INSTRUCTIONS = {
    "full_ft": """```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```""",
    "lora": """```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```""",
    "qlora": """```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained("{base_model}", quantization_config=bnb_config)
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```""",
    "tinylora": """Requires `tinylora.py` from the
[project repo](https://github.com/swathikchgithub/lora-methods-comparison/blob/main/scripts/tinylora.py)
(not a standard peft format):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from tinylora import load_tinylora
from huggingface_hub import snapshot_download

base = AutoModelForCausalLM.from_pretrained("{base_model}")
checkpoint_dir = snapshot_download("{repo_id}")
model, v = load_tinylora(base, checkpoint_dir)
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
```""",
}
LOAD_INSTRUCTIONS["lora_fa"] = LOAD_INSTRUCTIONS["lora"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--hf-username", required=True)
    parser.add_argument("--results-file", default="results/comparison_report.json")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir or f"checkpoints/{args.method}"
    repo_id = f"{args.hf_username}/ag-news-{args.method.replace('_', '-')}"

    stats = {"accuracy": "N/A", "trainable_params": 0,
             "peak_gpu_memory_gb": "N/A", "wall_clock_s": "N/A"}
    try:
        with open(args.results_file) as f:
            all_results = json.load(f)
        stats = next(r for r in all_results if r["method"] == args.method)
    except (FileNotFoundError, StopIteration):
        print("Warning: no eval results found yet, publishing with placeholder stats")

    model_card = MODEL_CARD_TEMPLATE.format(
        base_model=args.base_model,
        method=args.method,
        title=f"AG News — {args.method}",
        method_description=METHOD_DESCRIPTIONS[args.method],
        trainable_params=stats["trainable_params"],
        accuracy=f"{stats['accuracy']:.1%}" if isinstance(stats["accuracy"], float) else stats["accuracy"],
        peak_mem=stats["peak_gpu_memory_gb"],
        train_time=stats["wall_clock_s"],
        load_instructions=LOAD_INSTRUCTIONS[args.method].format(
            repo_id=repo_id, base_model=args.base_model),
    )
    with open(f"{checkpoint_dir}/README.md", "w") as f:
        f.write(model_card)

    api = HfApi()
    create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "runs", "*.log"],
    )
    print(f"Published to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
