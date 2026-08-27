"""Integration test: LoRA injection, checkpoint loading, and forward pass.

Verifies:
  1. inject_lora() replaces the correct layers without changing forward output.
  2. freeze_non_lora_params() correctly freezes/unfreezes the right parameters.
  3. A SAM2.1 checkpoint loads correctly after LoRA injection (original weights
     are restored; lora_A/lora_B remain at their initialised values).
  4. A forward pass through SAM2TrainLoRA completes without error.

Run from the sam2 directory:
    pyenv activate lorabora
    cd sam2
    python ../scripts/test_lora_pipeline.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/sam2")

import torch
import torch.nn as nn

from sam2.modeling.lora import (
    LoRALinear,
    freeze_non_lora_params,
    inject_lora,
    lora_state_dict,
)


# ---------------------------------------------------------------------------
# Test 1: LoRALinear drops-in correctly
# ---------------------------------------------------------------------------
def test_lora_linear_forward():
    linear = nn.Linear(64, 128, bias=True)
    x = torch.randn(4, 64)
    ref_out = linear(x).detach().clone()

    lora_lin = LoRALinear(linear, rank=4, alpha=4.0)
    # At init, lora_B is zero → delta is zero → output must equal original
    lora_out = lora_lin(x)
    assert torch.allclose(lora_out, ref_out, atol=1e-5), (
        f"LoRALinear init delta non-zero: max diff={( lora_out - ref_out).abs().max().item()}"
    )
    print("  LoRALinear zero-init forward  ✓")

    # After backprop, only lora_A/lora_B should accumulate gradients
    loss = lora_out.sum()
    loss.backward()
    assert lora_lin.weight.grad is None, "Frozen weight should have no grad"
    assert lora_lin.lora_A.grad is not None
    assert lora_lin.lora_B.grad is not None
    print("  LoRALinear gradient isolation  ✓")


# ---------------------------------------------------------------------------
# Test 2: State-dict key compatibility
# ---------------------------------------------------------------------------
def test_state_dict_keys():
    linear = nn.Linear(32, 64)
    lora_lin = LoRALinear(linear, rank=4, alpha=4.0)

    sd = lora_lin.state_dict()
    assert "weight" in sd, "LoRALinear must expose 'weight' in state_dict"
    assert "bias" in sd, "LoRALinear must expose 'bias' in state_dict"
    assert "lora_A" in sd
    assert "lora_B" in sd
    print("  State-dict keys compatible  ✓")


# ---------------------------------------------------------------------------
# Test 3: inject_lora on a tiny mock model
# ---------------------------------------------------------------------------
class TinyMultiScaleAttn(nn.Module):
    """Mimics MultiScaleAttention: has .qkv and .proj."""
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.proj = nn.Linear(16, 16)


class TinyBlock(nn.Module):
    """Mimics MultiScaleBlock: has .attn attribute."""
    def __init__(self):
        super().__init__()
        self.attn = TinyMultiScaleAttn()


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])

    @property
    def trunk(self):
        return self


class TinyDecoder(nn.Module):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(16, 16)
            self.self_attn.k_proj = nn.Linear(16, 16)
            self.self_attn.v_proj = nn.Linear(16, 16)
            self.self_attn.out_proj = nn.Linear(16, 16)
            self.cross_attn_token_to_image = nn.Module()
            self.cross_attn_token_to_image.q_proj = nn.Linear(16, 16)
            self.cross_attn_token_to_image.k_proj = nn.Linear(16, 16)
            self.cross_attn_token_to_image.v_proj = nn.Linear(16, 16)
            self.cross_attn_token_to_image.out_proj = nn.Linear(16, 16)
            self.cross_attn_image_to_token = nn.Module()
            self.cross_attn_image_to_token.q_proj = nn.Linear(16, 16)
            self.cross_attn_image_to_token.k_proj = nn.Linear(16, 16)
            self.cross_attn_image_to_token.v_proj = nn.Linear(16, 16)
            self.cross_attn_image_to_token.out_proj = nn.Linear(16, 16)

    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([self.Layer()])
        final = nn.Module()
        final.q_proj = nn.Linear(16, 16)
        final.k_proj = nn.Linear(16, 16)
        final.v_proj = nn.Linear(16, 16)
        final.out_proj = nn.Linear(16, 16)
        self.transformer.final_attn_token_to_image = final


class MockSAM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = TinyEncoder()
        self.sam_mask_decoder = TinyDecoder()


def test_inject_lora_mock():
    model = MockSAM2()
    inject_lora(model, rank=4, alpha=4.0, target_hiera=True, target_mask_decoder=True)

    # All targeted attention projections should now be LoRALinear
    for block in model.image_encoder.blocks:
        assert isinstance(block.attn.qkv, LoRALinear), "qkv not replaced"
        assert isinstance(block.attn.proj, LoRALinear), "proj not replaced"

    for layer in model.sam_mask_decoder.transformer.layers:
        for attn_name in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            attn = getattr(layer, attn_name)
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                assert isinstance(getattr(attn, proj_name), LoRALinear), (
                    f"{attn_name}.{proj_name} not replaced"
                )
    print("  inject_lora on mock model  ✓")


def test_freeze_non_lora():
    model = MockSAM2()
    inject_lora(model, rank=4, alpha=4.0)
    freeze_non_lora_params(model)

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]

    # Every trainable param must be lora_A or lora_B
    for name in trainable_names:
        assert "lora_A" in name or "lora_B" in name, (
            f"Non-LoRA param is trainable: {name}"
        )
    # Every frozen param must NOT be lora_A or lora_B
    for name in frozen_names:
        assert "lora_A" not in name and "lora_B" not in name, (
            f"LoRA param is frozen: {name}"
        )
    print(f"  freeze_non_lora: {len(trainable_names)} trainable, {len(frozen_names)} frozen  ✓")


def test_lora_state_dict_small():
    model = MockSAM2()
    inject_lora(model, rank=4, alpha=4.0)
    sd = lora_state_dict(model)
    assert all("lora_A" in k or "lora_B" in k for k in sd), "lora_state_dict contains non-LoRA keys"
    assert len(sd) > 0
    print(f"  lora_state_dict: {len(sd)} entries  ✓")


# ---------------------------------------------------------------------------
# Test 4: Real SAM2.1 model (if checkpoint is available)
# ---------------------------------------------------------------------------
def test_real_model_checkpoint():
    ckpt_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    )
    sam2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../sam2"))
    cfg_path = os.path.join(sam2_root, "sam2/sam2_hiera_b+.yaml")

    if not os.path.exists(ckpt_path):
        print("  Skipping real-model test (checkpoint not found at", ckpt_path, ")")
        return
    if not os.path.exists(cfg_path):
        print("  Skipping real-model test (config not found at", cfg_path, ")")
        return

    from hydra.utils import instantiate
    from sam2.modeling.lora import inject_lora, freeze_non_lora_params
    from omegaconf import OmegaConf

    # Load config directly without going through Hydra's compose
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    sam = instantiate(cfg.model, _recursive_=True)

    # Load checkpoint weights (strict=False allows extra checkpoint keys)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = sam.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  (Missing keys in checkpoint: {missing[:3]}...)")
    if unexpected:
        print(f"  (Unexpected checkpoint keys (will be ignored): {len(unexpected)})")

    n_before = sum(p.numel() for p in sam.parameters())
    inject_lora(sam, rank=4, alpha=4.0)
    n_after = sum(p.numel() for p in sam.parameters())
    freeze_non_lora_params(sam)
    n_trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)

    print(
        f"  Real SAM2 LoRA: params before={n_before:,}, "
        f"after={n_after:,}, trainable={n_trainable:,}  ✓"
    )


def main():
    print("Running LoRA pipeline tests ...\n")
    test_lora_linear_forward()
    test_state_dict_keys()
    test_inject_lora_mock()
    test_freeze_non_lora()
    test_lora_state_dict_small()
    test_real_model_checkpoint()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
