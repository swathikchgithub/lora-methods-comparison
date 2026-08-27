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
        # Randomized low-rank SVD - only computes the top `rank` singular
        # triplets, far cheaper than a full SVD for these matrix sizes.
        U, S, V = torch.svd_lowrank(weight, q=rank)
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
