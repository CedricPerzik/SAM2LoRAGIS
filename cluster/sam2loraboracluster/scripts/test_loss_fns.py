"""Unit tests for the favela composite loss functions.

Verifies:
  1. Each loss produces valid gradients on dummy tensors.
  2. SoftDice uses eps=1e-5 and handles all-zero masks without NaN.
  3. FocalTversky output is in [0, 1] for reasonable inputs.
  4. FavelaCompositeLoss integrates into the MultiStepMultiMasksAndIous interface.

Run from the sam2 directory:
    pyenv activate lorabora
    cd sam2
    python ../scripts/test_loss_fns.py
"""

import sys
import os
# Add sam2 to path so we can import training.loss_fns_favela without the
# full Hydra/trainer dependency chain (which requires tensorboard etc.)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "sam2"))

import torch

from training.loss_fns_favela import (
    FavelaCompositeLoss,
    focal_tversky_loss,
    segmentation_bce_loss,
    soft_dice_loss,
)


def _dummy_batch(N=2, M=3, H=64, W=64, device="cpu"):
    """Return (logits, targets, num_objects) dummy tensors."""
    logits = torch.randn(N, M, H, W, device=device, requires_grad=True)
    # Sparse binary targets (rooftops cover ~20% of tile area)
    targets = (torch.rand(N, M, H, W, device=device) > 0.8).float()
    return logits, targets, float(N)


def test_segmentation_bce_gradient():
    logits, targets, n = _dummy_batch()
    loss = segmentation_bce_loss(logits, targets, n, loss_on_multimask=True)
    assert loss.shape == (2, 3), f"Expected (N, M) shape, got {loss.shape}"
    # Reduce to scalar and backprop
    loss.sum().backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "NaN gradient in SegmentationBCE"
    print(f"  SegmentationBCE: loss={loss.mean().item():.4f}  ✓")


def test_soft_dice_gradient():
    logits, targets, n = _dummy_batch()
    loss = soft_dice_loss(logits, targets, n, eps=1e-5, loss_on_multimask=True)
    assert loss.shape == (2, 3)
    loss.sum().backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "NaN gradient in SoftDice"
    print(f"  SoftDice:        loss={loss.mean().item():.4f}  ✓")


def test_soft_dice_empty_mask():
    """SoftDice must not produce NaN when targets are all zero (ε prevents div/0)."""
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.zeros(2, 1, 32, 32)
    loss = soft_dice_loss(logits, targets, 2.0, eps=1e-5, loss_on_multimask=True)
    assert not torch.isnan(loss).any(), "NaN in SoftDice with empty targets"
    loss.sum().backward()
    assert not torch.isnan(logits.grad).any(), "NaN gradient with empty targets"
    print(f"  SoftDice (empty): loss={loss.mean().item():.4f}  ✓")


def test_focal_tversky_gradient():
    logits, targets, n = _dummy_batch()
    loss = focal_tversky_loss(logits, targets, n, loss_on_multimask=True)
    assert loss.shape == (2, 3)
    # Loss should be in [0, 1] range (bounded Tversky-based)
    assert (loss >= 0).all() and (loss <= 1).all(), f"Loss out of [0,1]: {loss}"
    loss.sum().backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "NaN gradient in FocalTversky"
    print(f"  FocalTversky:    loss={loss.mean().item():.4f}  ✓")


def test_composite_loss_interface():
    """FavelaCompositeLoss must integrate with the MultiStepMultiMasksAndIous interface."""
    loss_fn = FavelaCompositeLoss(
        weight_dict={"loss_class": 1.0},
        weight_bce=1.0,
        weight_dice=1.0,
        weight_tversky=1.0,
        weight_iou=1.0,
        pred_obj_scores=False,
        iou_use_l1_loss=True,
        supervise_all_iou=True,
    )

    N, M, H, W = 2, 3, 64, 64
    logits = torch.randn(N, M, H, W, requires_grad=True)
    targets = (torch.rand(N, H, W) > 0.8).float()
    ious_pred = torch.sigmoid(torch.randn(N, M))

    # SAM2Train outputs: list of dicts, one per frame
    frame_out = {
        "multistep_pred_multimasks_high_res": [logits],
        "multistep_pred_ious": [ious_pred],
        "multistep_object_score_logits": [torch.zeros(N, 1)],
    }
    outs_batch = [frame_out]         # 1 frame
    targets_batch = targets.unsqueeze(0)  # [T=1, N, H, W]

    losses = loss_fn(outs_batch, targets_batch)

    core = losses["core_loss"]
    assert isinstance(core, torch.Tensor), "core_loss must be a tensor"
    assert not torch.isnan(core), f"core_loss is NaN: {losses}"
    core.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any(), "NaN gradient in composite loss"
    print(f"  FavelaCompositeLoss: core_loss={core.item():.4f}  ✓")
    print(f"    sub-losses: { {k: f'{float(v):.4f}' for k,v in losses.items()} }")


def main():
    print("Running favela loss function unit tests ...\n")
    test_segmentation_bce_gradient()
    test_soft_dice_gradient()
    test_soft_dice_empty_mask()
    test_focal_tversky_gradient()
    test_composite_loss_interface()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
