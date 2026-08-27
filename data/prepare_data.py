"""Prepare AG News for the 5-methods comparison: subsample, format for the
Qwen2.5 chat template, and split into train/val/test.

AG News ships with 120k train / 7.6k test examples — far more than this
comparison project needs (the point is comparing training *methods* on
identical data, not maximizing dataset size). Subsampled to a scale
comparable to the ticket-triage project for consistency and fast, cheap
iteration across 5 separate training runs.

Usage:
    python data/prepare_data.py --n-train 800 --n-val 100 --n-test 200 --seed 42
"""

import argparse
import json
import random

from datasets import load_dataset

LABELS = ["World", "Sports", "Business", "Sci/Tech"]

SYSTEM_PROMPT = (
    "You are a news categorization assistant. Given a news headline and "
    "description, classify it into exactly one category. Respond with "
    "ONLY a JSON object with one key: \"category\" (one of: World, Sports, "
    "Business, Sci/Tech). No extra text, no markdown, just the JSON object."
)


def to_chat_example(text, label_idx):
    assistant_content = json.dumps({"category": LABELS[label_idx]}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": LABELS[label_idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="data/")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    ds_train_full = load_dataset("ag_news", split="train")
    ds_test_full = load_dataset("ag_news", split="test")

    # Stratified subsample from the official train split for train+val,
    # keeping test entirely separate from AG News's own official test
    # split — no risk of leakage since these are disjoint HF splits to
    # begin with, not something we're constructing ourselves this time.
    by_label = {i: [] for i in range(len(LABELS))}
    for ex in ds_train_full:
        by_label[ex["label"]].append(ex["text"])
    for label_examples in by_label.values():
        rng.shuffle(label_examples)

    per_label_train = args.n_train // len(LABELS)
    per_label_val = args.n_val // len(LABELS)

    train, val = [], []
    for label_idx, texts in by_label.items():
        train += [to_chat_example(t, label_idx) for t in texts[:per_label_train]]
        val += [to_chat_example(t, label_idx) for t in texts[per_label_train:per_label_train + per_label_val]]

    test_by_label = {i: [] for i in range(len(LABELS))}
    for ex in ds_test_full:
        test_by_label[ex["label"]].append(ex["text"])
    for label_examples in test_by_label.values():
        rng.shuffle(label_examples)

    per_label_test = args.n_test // len(LABELS)
    test = []
    for label_idx, texts in test_by_label.items():
        test += [to_chat_example(t, label_idx) for t in texts[:per_label_test]]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = f"{args.outdir.rstrip('/')}/{name}.jsonl"
        with open(path, "w") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} examples -> {path}")

    from collections import Counter
    print("\nCategory balance check:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}:", dict(Counter(r["category"] for r in split)))


if __name__ == "__main__":
    main()
