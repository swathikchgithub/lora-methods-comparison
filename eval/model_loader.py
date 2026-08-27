"""Method-specific model loading for evaluation. Each of the 5 methods
saves its trained weights differently, so loading them back for
inference isn't one-size-fits-all."""

import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tinylora import load_tinylora


def load_trained_model(method, base_model_name, checkpoint_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=torch.bfloat16, device_map="auto")

    elif method in ("lora", "lora_fa"):
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, checkpoint_dir)

    elif method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config, device_map="auto")
        model = PeftModel.from_pretrained(base, checkpoint_dir)

    elif method == "tinylora":
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
        model, _ = load_tinylora(base, checkpoint_dir)

    else:
        raise ValueError(f"unknown method {method}")

    model.eval()
    return model, tokenizer


def load_train_stats(checkpoint_dir):
    with open(f"{checkpoint_dir}/train_stats.json") as f:
        return json.load(f)
