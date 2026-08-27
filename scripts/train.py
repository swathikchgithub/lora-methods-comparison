"""Train one of 5 fine-tuning methods on AG News, same base model, same
GPU, same data — the whole point is a controlled comparison.

Usage:
    python scripts/train.py --method qlora --base-model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from tinylora import apply_tinylora, save_tinylora

METHODS = ["full_ft", "lora", "lora_fa", "qlora", "tinylora"]

# LoRA hyperparameters shared by lora / lora_fa / qlora, for direct
# comparability - same rank/alpha as the validated ticket-triage recipe.
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# TinyLoRA is intentionally left at rank 2 / u=13 to match the paper's
# own main-experiment setting, not tuned per-task.
TINYLORA_RANK = 2
TINYLORA_U = 13

# Full fine-tuning updates every parameter directly, so it needs a much
# smaller LR than the LoRA family (whose adapters start near-zero and
# need a bigger nudge to move at all) or TinyLoRA (whose entire update is
# funneled through 13 numbers, so each one needs an outsized LR to move
# the needle across the whole model).
METHOD_LR = {
    "full_ft": 2e-5,
    "lora": 2e-4,
    "lora_fa": 2e-4,
    "qlora": 2e-4,
    "tinylora": 5e-2,
}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train-file", default="data/train.jsonl")
    p.add_argument("--val-file", default="data/val.jsonl")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.method}"
    if args.learning_rate is None:
        args.learning_rate = METHOD_LR[args.method]
    return args


def load_model_for_method(method, base_model_name, seed):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=bnb_config, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto")

    if method in ("lora", "lora_fa", "qlora"):
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)
        if method == "lora_fa":
            frozen = 0
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    param.requires_grad = False
                    frozen += 1
            print(f"LoRA-FA: froze {frozen} lora_A tensors, only lora_B trains")
        model.print_trainable_parameters()
        tinylora_v = None

    elif method == "tinylora":
        model, tinylora_v = apply_tinylora(model, rank=TINYLORA_RANK, u=TINYLORA_U, seed=seed)

    elif method == "full_ft":
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Full FT: trainable params: {n_trainable:,} || all params: {n_total:,} "
              f"|| trainable%: {100 * n_trainable / n_total:.4f}")
        tinylora_v = None
    else:
        raise ValueError(f"unknown method {method}")

    return model, tokenizer, tinylora_v


def main():
    args = build_args()
    print(f"=== Training method: {args.method} | lr={args.learning_rate} ===")

    model, tokenizer, tinylora_v = load_model_for_method(args.method, args.base_model, args.seed)

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    val_ds = load_dataset("json", data_files=args.val_file, split="train")

    def format_example(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        bf16=True,
        optim="paged_adamw_8bit" if args.method == "qlora" else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.time()
    trainer.train()
    wall_clock_s = time.time() - start
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    if args.method == "tinylora":
        save_tinylora(tinylora_v, rank=TINYLORA_RANK, u=TINYLORA_U, seed=args.seed,
                      target_modules=LORA_TARGET_MODULES, out_dir=args.output_dir)
    else:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) \
        if args.method != "tinylora" else TINYLORA_U

    print(f"\n=== {args.method} training complete ===")
    print(f"Best eval_loss: {trainer.state.best_metric}")
    print(f"Wall-clock: {wall_clock_s:.1f}s")
    print(f"Peak GPU memory: {peak_mem_gb:.2f} GB" if peak_mem_gb else "Peak GPU memory: N/A (no CUDA)")
    print(f"Trainable params: {n_trainable:,}")
    print(f"Adapter/model saved to {args.output_dir}")

    import json
    with open(f"{args.output_dir}/train_stats.json", "w") as f:
        json.dump({
            "method": args.method,
            "best_eval_loss": trainer.state.best_metric,
            "wall_clock_s": wall_clock_s,
            "peak_gpu_memory_gb": peak_mem_gb,
            "trainable_params": n_trainable,
            "learning_rate": args.learning_rate,
        }, f, indent=2)


if __name__ == "__main__":
    main()
