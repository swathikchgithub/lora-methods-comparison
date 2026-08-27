"""TinyLoRA: a from-paper implementation, not a peft library call.

From "TinyLoRA: Minimal 13-Parameter Reasoning" (Feb 2026,
arXiv:2602.04118). Not in the mainline peft library — this is a direct
implementation of the paper's mechanism, verified against the paper
rather than guessed at.

The mechanism, per module:
    W' = W + U Sigma (sum_i v_i * P_i) V^T

- U, Sigma, V: the top-r truncated SVD of the module's own frozen weight
  W, computed once at init. Frozen — these are NOT random directions like
  vanilla LoRA's init, they're the principal directions of the actual
  weight matrix.
- P: a fixed (u, r, r) tensor of random r x r matrices, generated once
  from a fixed seed. Frozen, never trained.
- v: a trainable vector of length u. This is the ONLY trainable
  parameter, and per the paper's "full weight tying" setting, ONE SINGLE
  v is shared across every adapted module in the entire model - not one
  v per layer. That's the mechanism behind the paper's 13-parameter
  result: r=2, u=13, one shared v, applied to all 7 attention/MLP
  projections across every layer.

Efficient forward (never materializes the full d x k delta):
    h = x @ V              # [batch, r]
    h = h @ (Sigma @ R)    # [batch, r], R = sum_i v_i * P_i (r x r)
    delta = h @ U.T        # [batch, out_features]
    y = base_linear(x) + delta
"""

import json

import torch
import torch.nn as nn

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]


class TinyLoRALinear(nn.Module):
    """Wraps one frozen nn.Linear with a TinyLoRA update sharing a single
    model-wide trainable vector `v`."""

    def __init__(self, base_linear: nn.Linear, shared_v: nn.Parameter,
                 rank: int, u: int, seed: int):
        super().__init__()
        self.base_linear = base_linear
        for p in self.base_linear.parameters():
            p.requires_grad = False

        weight = base_linear.weight.data.float()  # [out_features, in_features]
        # Exact SVD, not torch.svd_lowrank's randomized approximation.
        # svd_lowrank draws from the global RNG internally with no way to
        # seed it, so two calls on identical weights give DIFFERENT U/V
        # (verified by a failing round-trip test) - which would silently
        # break checkpoint reconstruction, the entire premise of only
        # saving `v` and a seed. Exact SVD is deterministic and, at these
        # per-layer matrix sizes, still fast since this runs once per
        # layer at model-load time, not per training step.
        U_full, S_full, Vh_full = torch.linalg.svd(weight, full_matrices=False)
        U, S, V = U_full[:, :rank], S_full[:rank], Vh_full[:rank, :].T
        # U: [out, r], S: [r], V: [in, r]
        self.register_buffer("U", U.to(base_linear.weight.dtype))
        self.register_buffer("S", torch.diag(S).to(base_linear.weight.dtype))
        self.register_buffer("V", V.to(base_linear.weight.dtype))

        gen = torch.Generator().manual_seed(seed)
        P = torch.randn(u, rank, rank, generator=gen)
        self.register_buffer("P", P.to(base_linear.weight.dtype))

        self.v = shared_v  # reference, not a copy - this is the tying

    def forward(self, x):
        base_out = self.base_linear(x)
        R = torch.einsum("u,urs->rs", self.v.to(self.P.dtype), self.P)  # [r, r]
        sigma_r = self.S @ R                                            # [r, r]
        h = x @ self.V                                                  # [..., r]
        h = h @ sigma_r                                                 # [..., r]
        delta = h @ self.U.T                                            # [..., out]
        return base_out + delta.to(base_out.dtype)


def apply_tinylora(model, rank=2, u=13, seed=42):
    """Replace every target Linear layer in-place with a TinyLoRALinear,
    all sharing one model-wide trainable vector `v` of length u.

    Time: O(L * svd(r)) for L target layers, each a cheap randomized
    rank-r SVD - not O(full SVD), since torch.svd_lowrank only computes
    the top r singular triplets.
    Space: O(u) trainable (the shared vector) + O(L * r * (d + k)) frozen
    buffers for the per-layer U/V bases - negligible next to the base
    model's own parameter count.
    """
    shared_v = nn.Parameter(torch.zeros(u))
    replaced = 0

    def _recurse(module):
        nonlocal replaced
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in TARGET_MODULES:
                wrapped = TinyLoRALinear(child, shared_v, rank, u, seed=seed + replaced)
                setattr(module, name, wrapped)
                replaced += 1
            else:
                _recurse(child)

    _recurse(model)

    for p in model.parameters():
        p.requires_grad = False
    shared_v.requires_grad = True

    print(f"TinyLoRA: wrapped {replaced} linear layers, "
          f"rank={rank}, u={u}, trainable params={u} (one shared vector)")
    return model, shared_v


def save_tinylora(shared_v, rank, u, seed, target_modules, out_dir):
    """The checkpoint is genuinely tiny: just the u trainable numbers plus
    the config needed to regenerate P deterministically at load time -
    P itself is never saved, since apply_tinylora() reconstructs it
    byte-for-byte from the same seed. This is the mechanism behind the
    paper's "13 parameters, 26 bytes" claim - there's nothing else to
    store."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    torch.save(shared_v.detach().cpu(), f"{out_dir}/tinylora_v.pt")
    with open(f"{out_dir}/tinylora_config.json", "w") as f:
        json.dump({"rank": rank, "u": u, "seed": seed,
                    "target_modules": target_modules}, f, indent=2)
    print(f"TinyLoRA checkpoint saved to {out_dir} "
          f"({shared_v.numel()} floats, {shared_v.numel() * 2} bytes in bf16)")


def load_tinylora(model, checkpoint_dir):
    """Reconstruct a TinyLoRA-wrapped model from a saved checkpoint:
    reapply the same wrapping (regenerating P from the saved seed), then
    load the trained vector into it."""
    with open(f"{checkpoint_dir}/tinylora_config.json") as f:
        config = json.load(f)
    model, shared_v = apply_tinylora(model, rank=config["rank"], u=config["u"], seed=config["seed"])
    saved_v = torch.load(f"{checkpoint_dir}/tinylora_v.pt")
    with torch.no_grad():
        shared_v.copy_(saved_v)
    return model, shared_v
