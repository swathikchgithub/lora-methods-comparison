# Five Ways to Fine-Tune: Full FT vs LoRA vs LoRA-FA vs QLoRA vs TinyLoRA

A rigorous, apples-to-apples comparison of five parameter-efficiency tiers —
**Full fine-tuning, LoRA, LoRA-FA, QLoRA, and TinyLoRA** — trained on the
same task, same base model, same GPU, with the same evaluation harness.
The question this project answers with real numbers, not assumptions: as
you shrink the trainable parameter count from 100% down to a literal
handful of parameters, how much task performance do you actually give up,
and where does the curve bend?

> **Status:** project scaffolded, methodology and plan below are final.
> Training runs and the results table are in progress — see
> [Results](#results) for what's real vs. pending.

## The task

[AG News](https://huggingface.co/datasets/ag_news) — 4-class news topic
classification (World / Sports / Business / Sci-Tech). A standard, public
benchmark rather than a synthetic dataset, deliberately: the point of this
project is comparing *methods*, not building a task from scratch, and a
benchmark other people already know lets anyone reviewing this calibrate
the results immediately.

## The model

Qwen2.5-0.5B-Instruct — small enough that even full fine-tuning (weights +
gradients + Adam's momentum/variance state, the most memory-hungry of the
five methods) fits comfortably on the same GPU tier as the other four, so
this is a genuinely controlled comparison, not five different hardware
setups.

## The five methods

| # | Method | Trainable surface | Notes |
|---|---|---|---|
| 1 | **Full fine-tuning** | 100% of parameters | Baseline — the most capacity, the most memory, the most compute. |
| 2 | **LoRA** | Low-rank adapters `A`, `B` at every target layer | Standard `peft.LoraConfig`. |
| 3 | **LoRA-FA** | Only `B` (A frozen at random init) | Halves LoRA's trainable parameter count for roughly the same mechanism. |
| 4 | **QLoRA** | Same as LoRA, base weights in 4-bit NF4 | The exact recipe validated in the [ticket-triage project](https://github.com/swathikchgithub/llm-lora-ticket-triage), reused here for comparability. |
| 5 | **TinyLoRA** | A single shared vector projected through a fixed random tensor | From the Feb 2026 paper ([TinyLoRA: Minimal 13-Parameter Reasoning](https://www.emergentmind.com/papers/2602.04118)) — not in the mainline `peft` library, implemented directly from the paper's mechanism. |

## Comparison methodology

For each method, the same four things get measured on the same held-out
test set:

- **Trainable parameter count** — the headline efficiency number
- **Peak GPU memory during training** — the practical cost of each method
- **Training wall-clock time** — same GPU, same effective batch size
- **Held-out accuracy** — does less trainable capacity actually cost
  accuracy, and how much

Training loss curves are logged for all five runs and compared side by
side, the same way overfitting was diagnosed and caught in the
ticket-triage project.

## Trained adapters

All five trained adapters are published to Hugging Face Hub — links go
here once training completes.

## Results

*(Pending — filled in with real numbers once all five training runs and
the comparison harness complete. No numbers are estimated or invented in
the meantime.)*

## Repo structure

```
lora-methods-comparison/
├── data/           # AG News prep/formatting
├── scripts/        # one training script per method
├── eval/           # shared comparison harness (accuracy, memory, speed)
├── results/        # comparison_report.md/json once runs complete
└── docs/           # method-specific notes (e.g. TinyLoRA implementation)
```

## License

[MIT](LICENSE)
