# Five Ways to Fine-Tune: Full FT vs LoRA vs LoRA-FA vs QLoRA vs TinyLoRA

A rigorous, apples-to-apples comparison of five parameter-efficiency tiers —
**Full fine-tuning, LoRA, LoRA-FA, QLoRA, and TinyLoRA** — trained on the
same task, same base model, same GPU, with the same evaluation harness.
The question this project answers with real numbers, not assumptions: as
you shrink the trainable parameter count from 100% down to a literal
handful of parameters, how much task performance do you actually give up,
and where does the curve bend?

> **Status:** complete. All five methods trained, evaluated on the same
> held-out test set, and published to Hugging Face Hub — see
> [Results](#results) for the real numbers.

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

All five trained checkpoints are public on Hugging Face Hub, each with a
model card documenting its method and how to load it:

- [ag-news-full-ft](https://huggingface.co/swathikchhuggingface/ag-news-full-ft)
- [ag-news-lora](https://huggingface.co/swathikchhuggingface/ag-news-lora)
- [ag-news-lora-fa](https://huggingface.co/swathikchhuggingface/ag-news-lora-fa)
- [ag-news-qlora](https://huggingface.co/swathikchhuggingface/ag-news-qlora)
- [ag-news-tinylora](https://huggingface.co/swathikchhuggingface/ag-news-tinylora) — the actual uploaded checkpoint is 12.2kB total.

## Results

Trained on an RTX PRO 4500 (Blackwell, RunPod), 3 epochs, same 800/100/200
AG News split, same evaluation harness, same GPU for all five runs.

| Method | Accuracy | Trainable params | Peak GPU mem | Train time | p50 / p95 latency |
|---|---|---|---|---|---|
| Full fine-tuning | 85.0% | 494,032,768 | 7.64 GB | 90s | 118 / 152 ms |
| LoRA | 85.5% | 8,798,208 | 5.17 GB | 105s | 258 / 332 ms |
| **LoRA-FA** | **86.5%** | 4,866,048 | 4.61 GB | 99s | 260 / 334 ms |
| QLoRA | 85.5% | 8,798,208 | 3.32 GB | 205s | 427 / 547 ms |
| TinyLoRA | 60.5% | **13** | 4.25 GB | 111s | 280 / 314 ms |

JSON output validity was 100% for all five methods — none of this
comparison is confounded by output-format failures.

### What this actually shows

- **LoRA-FA beat full fine-tuning — the most parameter-constrained of the
  four "real" methods won, not the least.** 86.5% vs. 85.0%, with
  roughly 100x fewer trainable parameters than full FT. This isn't an
  anomaly to explain away: on a small dataset (800 examples) and a task
  easy enough for a 0.5B model, full FT's extra capacity doesn't
  translate to better generalization and may add optimization noise,
  while LoRA-FA's tighter constraint (only `B` trains, `A` stays frozen
  at random init) acts as implicit regularization. More trainable
  parameters is not automatically better.
- **TinyLoRA — 13 parameters, 60.5% accuracy, dramatically above the 25%
  random-chance baseline for 4 classes.** Genuinely meaningful signal
  from a checkpoint smaller than this sentence. It's also honestly far
  short of matching the other four methods, and far short of the
  original paper's own headline results — worth being explicit about why
  rather than implying a clean replication: the paper's 13-parameter
  result was on an 8B model doing math reasoning (GSM8K) via SFT and RL,
  not a 0.5B model doing 4-way topic classification via plain SFT.
  Different scale, different task, different training signal. The
  mechanism is faithfully implemented; the specific numbers were never
  going to transfer directly.
- **QLoRA was the slowest to train (205s) and slowest to serve (427ms
  p50), despite having the lowest peak memory (3.32GB).** Same finding
  as the [ticket-triage project](https://github.com/swathikchgithub/llm-lora-ticket-triage):
  4-bit quantization overhead is a real, measurable cost, and it only
  pays off when VRAM is the actual binding constraint. On a 32GB card
  training a 0.5B model, it wasn't — the memory savings bought nothing
  here, and the overhead was pure cost. (Note: this run's checkpoints
  weren't merged before latency measurement, unlike the ticket-triage
  project's final numbers — merging would likely close most of this gap,
  the same way it did there.)
- **Full fine-tuning was the fastest to train (90s) and by far the
  fastest to serve (118ms p50)** — no adapter-matmul overhead on every
  forward pass, no quantization dequant cost. It paid for that speed
  with 57x the memory of QLoRA and, on this task, no accuracy advantage
  at all.

### The debugging story — arguably the more interesting part

Two real, non-trivial bugs surfaced building the from-paper TinyLoRA
implementation, both caught by tests before — or immediately after —
touching a real GPU, not left to be discovered as a silent wrong answer:

1. **A non-deterministic SVD would have silently broken checkpoint
   reconstruction.** `torch.svd_lowrank`'s randomized approximation draws
   from the global RNG with no way to seed it, so two calls on identical
   weights produced different `U`/`V` — which would have undermined the
   entire "reconstruct from just a 13-number vector and a seed" premise
   the paper's efficiency claim depends on. A save/load round-trip test
   caught it before any GPU time was spent. Fixed by switching to exact
   `torch.linalg.svd`.
2. **Two tensors (`P` and the shared vector `v`) silently defaulted to
   CPU** because `torch.randn(...)` and `torch.zeros(...)` were called
   without an explicit device argument. Training never surfaced this —
   `SFTTrainer`/`Accelerate` moves the whole model, buffers included, to
   GPU internally as part of its own setup — but a standalone reload for
   evaluation never gets that same pass, and crashed on the first real
   forward call on the actual training pod. Both fixed by deriving the
   target device explicitly from the model's own parameters instead of
   leaving it implicit, with tests added to document the invariant going
   forward.

Also hit and fixed: this project's rented GPU turned out to be a
Blackwell-generation card, which the originally pinned `torch==2.4.1` +
`bitsandbytes==0.43.3` predate entirely (`no kernel image is available
for execution on the device`). Verified working versions
(`torch==2.11.0+cu128`, `bitsandbytes==0.50.2`) before running anything
for real — see `requirements.txt` for the specifics.

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
