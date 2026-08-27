"""Verify the from-paper TinyLoRA implementation before it ever touches a
real model on a GPU: exact trainable parameter count, genuine weight
tying across layers (not per-layer copies), and correct gradient flow.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from tinylora import TARGET_MODULES, TinyLoRALinear, apply_tinylora, save_tinylora, load_tinylora


class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, d * 2)
        self.up_proj = nn.Linear(d, d * 2)
        self.down_proj = nn.Linear(d * 2, d)
        self.norm = nn.LayerNorm(d)  # must NOT be wrapped


class _FakeModel(nn.Module):
    def __init__(self, d=16, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])


def _build(rank=2, u=13, n_layers=3, d=16):
    model = _FakeModel(d=d, n_layers=n_layers)
    model, shared_v = apply_tinylora(model, rank=rank, u=u, seed=42)
    return model, shared_v


def test_trainable_param_count_matches_u_exactly():
    _, shared_v = _build(u=13)
    trainable = sum(p.numel() for p in shared_v.parameters() if p.requires_grad) \
        if hasattr(shared_v, "parameters") else shared_v.numel()
    assert trainable == 13


def test_only_shared_vector_is_trainable():
    model, shared_v = _build()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) == 1
    assert trainable_params[0] is shared_v


def test_target_modules_are_wrapped_others_are_not():
    model, _ = _build(n_layers=1)
    layer = model.layers[0]
    for name in TARGET_MODULES:
        assert isinstance(getattr(layer, name), TinyLoRALinear)
    assert isinstance(layer.norm, nn.LayerNorm)


def test_vector_is_genuinely_shared_not_copied_per_layer():
    model, shared_v = _build(n_layers=3)
    q_projs = [model.layers[i].q_proj.v for i in range(3)]
    for v in q_projs:
        assert v is shared_v


def test_forward_pass_preserves_shape():
    model, _ = _build(d=16)
    x = torch.randn(4, 16)
    out = model.layers[0].q_proj(x)
    assert out.shape == (4, 16)


def test_backward_pass_flows_only_to_shared_vector():
    model, shared_v = _build(d=16)
    x = torch.randn(4, 16)
    out = model.layers[0].q_proj(x)
    out.sum().backward()
    assert shared_v.grad is not None
    assert shared_v.grad.shape == (13,)
    assert model.layers[0].q_proj.base_linear.weight.grad is None


def test_different_u_and_rank_still_produce_correct_shapes():
    model, shared_v = _build(rank=4, u=8, d=32)
    assert shared_v.shape == (8,)
    x = torch.randn(2, 32)
    out = model.layers[0].q_proj(x)
    assert out.shape == (2, 32)


def test_save_and_load_round_trip_produces_identical_outputs(tmp_path):
    # Real usage: reload the SAME pretrained base checkpoint, then
    # reapply TinyLoRA on top - base weights must match for the
    # regenerated SVD (and thus outputs) to match. A fresh random
    # re-init (different base weights) is a different scenario, not a
    # round-trip, so this test constructs the fresh model from the same
    # base weights explicitly, the way a real reload would.
    base_model = _FakeModel(d=16, n_layers=2)
    fresh_model = _FakeModel(d=16, n_layers=2)
    fresh_model.load_state_dict(base_model.state_dict())

    model, shared_v = apply_tinylora(base_model, rank=2, u=13, seed=42)
    with torch.no_grad():
        shared_v.copy_(torch.randn(13))  # simulate a trained vector

    x = torch.randn(3, 16)
    original_out = model.layers[0].q_proj(x).clone()

    save_tinylora(shared_v, rank=2, u=13, seed=42,
                   target_modules=TARGET_MODULES, out_dir=str(tmp_path))

    fresh_model, loaded_v = load_tinylora(fresh_model, str(tmp_path))

    assert torch.allclose(loaded_v, shared_v)
    reloaded_out = fresh_model.layers[0].q_proj(x)
    assert torch.allclose(original_out, reloaded_out, atol=1e-6)
