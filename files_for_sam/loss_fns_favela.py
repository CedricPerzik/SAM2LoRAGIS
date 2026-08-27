"""Custom composite loss for SAM2 LoRA fine-tuning on favela rooftop segmentation.

Active loss terms applied to the high-resolution mask predictions:
  FocalTversky  - Tversky index with focal exponent; alpha/beta control FP/FN balance,
                  gamma < 1 up-weights hard examples (Abraham & Khan 2019)
  IoUHead       - L1 supervision on the model's predicted-IoU auxiliary output
  ObjScore      - BCE on the object-presence classification logit

BCE and SoftDice are implemented but inactive by default (weight = 0.0).
SoftDice was removed from the active set because it is a symmetric special
case of Tversky (alpha=beta=0.5, gamma=1) and partially cancelled the recall bias
introduced by tversky_beta > 0.5.

FavelaCompositeLoss implements the same forward interface as
MultiStepMultiMasksAndIous so it can be plugged into the SAM2 trainer config
as a drop-in replacement.

Reference: Focal Tversky: https://arxiv.org/abs/1810.07842
"""

from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

# Must match the constant in training/trainer.py.
CORE_LOSS_KEY = "core_loss"


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def segmentation_bce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    loss_on_multimask: bool = False,
) -> torch.Tensor:
    """Standard binary cross-entropy with sigmoid.

    Args:
        inputs:  logits of shape [N, H, W] or [N, M, H, W].
        targets: binary ground-truth of same shape as inputs.
        num_objects: normalisation constant.
        loss_on_multimask: if True treat the M dimension as mask candidates.
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # [N, M]
    return loss.mean(1).sum() / num_objects


def soft_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    eps: float = 1e-5,
    loss_on_multimask: bool = False,
) -> torch.Tensor:
    """Soft (differentiable) Dice loss.

    eps = 1e-5 prevents division by zero on empty masks.
    """
    probs = inputs.sigmoid()
    if loss_on_multimask:
        assert probs.dim() == 4
        p = probs.flatten(2)        # [N, M, H*W]
        t = targets.flatten(2).float()
        inter = (p * t).sum(-1)     # [N, M]
        union = p.sum(-1) + t.sum(-1)
    else:
        p = probs.flatten(1)        # [N, H*W]
        t = targets.flatten(1).float()
        inter = (p * t).sum(1)      # [N]
        union = p.sum(1) + t.sum(1)

    dice = (2.0 * inter + eps) / (union + eps)
    loss = 1.0 - dice
    if loss_on_multimask:
        return loss / num_objects   # [N, M]
    return loss.sum() / num_objects


def focal_tversky_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_objects: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-5,
    loss_on_multimask: bool = False,
) -> torch.Tensor:
    """Focal Tversky Loss for handling class imbalance in dense rooftop regions.

    TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    FTL = (1 - TI)^gamma

    alpha < beta emphasises recall (penalises false negatives more than false positives).
    """
    probs = inputs.sigmoid()
    if loss_on_multimask:
        assert probs.dim() == 4
        p = probs.flatten(2)        # [N, M, H*W]
        t = targets.flatten(2).float()
        tp = (p * t).sum(-1)
        fp = (p * (1.0 - t)).sum(-1)
        fn = ((1.0 - p) * t).sum(-1)
    else:
        p = probs.flatten(1)        # [N, H*W]
        t = targets.flatten(1).float()
        tp = (p * t).sum(1)
        fp = (p * (1.0 - t)).sum(1)
        fn = ((1.0 - p) * t).sum(1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = (1.0 - tversky).pow(gamma)
    if loss_on_multimask:
        return loss / num_objects   # [N, M]
    return loss.sum() / num_objects


def _iou_loss(inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1=False):
    """IoU prediction head loss - standalone copy to avoid trainer import chain."""
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    if use_l1:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def _sigmoid_focal_loss(inputs, targets, num_objects, alpha=-1, gamma=0.0):
    """Focal loss - standalone copy for object-score supervision."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_objects


# ---------------------------------------------------------------------------
# Combined loss class
# ---------------------------------------------------------------------------

class FavelaCompositeLoss(nn.Module):
    """
    Composite segmentation loss for SAM2 LoRA fine-tuning.

    Active terms: FocalTversky + IoUHead + ObjScore.
    BCE (weight_bce) is available but off by default (0.0); it is guarded and
    skipped entirely when zero to avoid wasted compute.
    SoftDice has been removed - it is a symmetric Tversky special case that
    diluted the recall bias controlled by tversky_beta.

    Args:
        weight_dict:    dict for API compatibility; 'loss_class' key controls
                        object-score loss weight.
        weight_bce:     Weight for SegmentationBCE (default 0.0 = off).
        weight_tversky: Weight for FocalTversky term.
        weight_iou:     Weight for IoU-head auxiliary loss.
        tversky_alpha:  FP penalty in Tversky index (default 0.3).
        tversky_beta:   FN penalty in Tversky index (default 0.7).
        tversky_gamma:  Focal exponent; gamma < 1 up-weights hard examples.
        dice_eps:       Smoothing eps shared by FocalTversky (default 1e-5).
        supervise_all_iou: If True, average IoU loss over all mask slots.
        iou_use_l1_loss:   Use L1 instead of MSE for IoU supervision.
        pred_obj_scores:   Whether the model predicts object presence scores.
        focal_gamma_obj_score: Focal gamma for object score loss.
        focal_alpha_obj_score: Focal alpha for object score loss.
    """

    def __init__(
        self,
        weight_dict: dict,
        weight_bce: float = 0.0,
        weight_tversky: float = 1.0,
        weight_iou: float = 1.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 0.75,
        dice_eps: float = 1e-5,
        supervise_all_iou: bool = True,
        iou_use_l1_loss: bool = True,
        pred_obj_scores: bool = True,
        focal_gamma_obj_score: float = 0.0,
        focal_alpha_obj_score: float = -1.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.w_bce = weight_bce
        self.w_tversky = weight_tversky
        self.w_iou = weight_iou
        self.tv_alpha = tversky_alpha
        self.tv_beta = tversky_beta
        self.tv_gamma = tversky_gamma
        self.dice_eps = dice_eps
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1 = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.focal_gamma_obj = focal_gamma_obj_score
        self.focal_alpha_obj = focal_alpha_obj_score

    # ------------------------------------------------------------------
    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor) -> Dict:
        """
        Compute loss over a batch of frames.

        Args:
            outs_batch:    list[dict] of length T (one dict per frame).
            targets_batch: [T, N, H, W] ground-truth binary masks.
        """
        assert len(outs_batch) == len(targets_batch)

        num_objects = torch.tensor(
            targets_batch.shape[1],
            device=targets_batch.device,
            dtype=torch.float,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_objects)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        num_objects = torch.clamp(num_objects / world_size, min=1).item()

        agg = defaultdict(float)
        for outs, targets in zip(outs_batch, targets_batch):
            frame_losses = self._forward_frame(outs, targets, num_objects)
            for k, v in frame_losses.items():
                agg[k] = agg[k] + v

        return dict(agg)

    def _forward_frame(self, outputs: Dict, targets: torch.Tensor, num_objects: float) -> Dict:
        target_masks = targets.unsqueeze(1).float()  # [N, 1, H, W]
        assert target_masks.dim() == 4

        losses = {
            "loss_tversky": 0,
            "loss_iou": 0,
            "loss_class": 0,
        }
        if self.w_bce > 0:
            losses["loss_bce"] = 0
        for src_masks, ious, obj_logits in zip(
            outputs["multistep_pred_multimasks_high_res"],
            outputs["multistep_pred_ious"],
            outputs["multistep_object_score_logits"],
        ):
            self._update_losses(losses, src_masks, target_masks, ious, num_objects, obj_logits)

        losses[CORE_LOSS_KEY] = (
            losses["loss_tversky"] * self.w_tversky
            + losses["loss_iou"] * self.w_iou
            + losses["loss_class"] * self.weight_dict.get("loss_class", 1.0)
        )
        if self.w_bce > 0:
            losses[CORE_LOSS_KEY] = losses[CORE_LOSS_KEY] + losses["loss_bce"] * self.w_bce
        return losses

    def _update_losses(self, losses, src_masks, target_masks, ious, num_objects, obj_logits):
        target_masks = target_masks.expand_as(src_masks)

        loss_ftv = focal_tversky_loss(
            src_masks, target_masks, num_objects,
            alpha=self.tv_alpha, beta=self.tv_beta,
            gamma=self.tv_gamma, eps=self.dice_eps, loss_on_multimask=True,
        )
        loss_iou = _iou_loss(
            src_masks, target_masks, ious, num_objects,
            loss_on_multimask=True, use_l1=self.iou_use_l1,
        )

        # BCE is off by default (w_bce=0); only compute when active.
        if self.w_bce > 0:
            loss_bce = segmentation_bce_loss(
                src_masks, target_masks, num_objects, loss_on_multimask=True
            )

        # Object classification loss
        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=src_masks.dtype, device=src_masks.device)
            target_obj = torch.ones(
                loss_ftv.shape[0], 1, dtype=src_masks.dtype, device=src_masks.device
            )
        else:
            target_obj = torch.any(
                (target_masks[:, 0] > 0).flatten(1), dim=-1
            )[..., None].float()
            loss_class = _sigmoid_focal_loss(
                obj_logits, target_obj, num_objects,
                alpha=self.focal_alpha_obj, gamma=self.focal_gamma_obj,
            )

        # Select best mask candidate per sample by FocalTversky alone.
        if loss_ftv.size(1) > 1:
            best_inds = torch.argmin(loss_ftv, dim=-1)
            batch_inds = torch.arange(loss_ftv.size(0), device=loss_ftv.device)
            loss_ftv = loss_ftv[batch_inds, best_inds].unsqueeze(1)
            if self.w_bce > 0:
                loss_bce = loss_bce[batch_inds, best_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_iou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_iou[batch_inds, best_inds].unsqueeze(1)

        loss_ftv = loss_ftv * target_obj
        loss_iou = loss_iou * target_obj

        if self.w_bce > 0:
            loss_bce = loss_bce * target_obj
            losses["loss_bce"] = losses["loss_bce"] + loss_bce.sum()
        losses["loss_tversky"] = losses["loss_tversky"] + loss_ftv.sum()
        losses["loss_iou"] = losses["loss_iou"] + loss_iou.sum()
        losses["loss_class"] = losses["loss_class"] + loss_class
