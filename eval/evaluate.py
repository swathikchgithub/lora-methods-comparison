"""Comparison harness: evaluate all 5 trained methods on the same
held-out AG News test set and produce one side-by-side report.

Usage:
    python eval/evaluate.py --base-model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import re
import time

import torch

from model_loader import load_trained_model, load_train_stats

METHODS = ["full_ft", "lora", "lora_fa", "qlora", "tinylora"]
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def parse_prediction(raw_text):
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, False
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, False
    if "category" not in parsed:
        return None, False
    return parsed, True


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens=40):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    completion_ids = output_ids[0][prompt_tokens:]
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return raw_text, latency_ms


def evaluate_method(method, base_model_name, test_examples, checkpoint_dir):
    print(f"\nEvaluating {method}...")
    model, tokenizer = load_trained_model(method, base_model_name, checkpoint_dir)

    correct = 0
    parse_failures = 0
    latencies = []

    for ex in test_examples:
        messages = ex["messages"][:2]  # system + user only, not the ground-truth assistant turn
        raw, latency_ms = generate(model, tokenizer, messages)
        latencies.append(latency_ms)
        parsed, ok = parse_prediction(raw)
        if not ok:
            parse_failures += 1
            continue
        if parsed.get("category") == ex["category"]:
            correct += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accuracy = correct / len(test_examples)
    parse_success_rate = 1 - (parse_failures / len(test_examples))
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    train_stats = load_train_stats(checkpoint_dir)

    return {
        "method": method,
        "accuracy": accuracy,
        "json_parse_success_rate": parse_success_rate,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        **train_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--test-file", default="data/test.jsonl")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()

    with open(args.test_file) as f:
        test_examples = [json.loads(line) for line in f]

    results = []
    for method in METHODS:
        checkpoint_dir = f"{args.checkpoints_dir}/{method}"
        result = evaluate_method(method, args.base_model, test_examples, checkpoint_dir)
        results.append(result)
        print(f"  {method}: accuracy={result['accuracy']:.1%}, "
              f"trainable_params={result['trainable_params']:,}, "
              f"peak_mem={result['peak_gpu_memory_gb']:.2f}GB, "
              f"train_time={result['wall_clock_s']:.1f}s")

    import os
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/comparison_report.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# Full FT vs LoRA vs LoRA-FA vs QLoRA vs TinyLoRA\n",
        "| Method | Accuracy | Trainable params | Peak GPU mem | Train time | JSON valid |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['method']} | {r['accuracy']:.1%} | {r['trainable_params']:,} | "
            f"{r['peak_gpu_memory_gb']:.2f} GB | {r['wall_clock_s']:.0f}s | "
            f"{r['json_parse_success_rate']:.1%} |"
        )
    lines.append("\n## Latency\n")
    lines.append("| Method | p50 (ms) | p95 (ms) |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r['method']} | {r['p50_latency_ms']:.0f} | {r['p95_latency_ms']:.0f} |")

    with open(f"{args.out}/comparison_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\nDone. Reports written to {args.out}/comparison_report.{{json,md}}")


if __name__ == "__main__":
    main()
