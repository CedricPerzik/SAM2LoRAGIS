"""LoRA (Low-Rank Adaptation) injection for SAM2.1 fine-tuning.

Injects trainable low-rank matrices alongside frozen layers in:
  - Hiera image encoder  attention (MultiScaleAttention.qkv and .proj)
  - Hiera image encoder  MLP FFN   (MultiScaleBlock.mlp.layers[0/1])
  - MaskDecoder transformer attention (Attention q/k/v/out_proj)
  - MaskDecoder transformer MLP FFN  (TwoWayAttentionBlock.mlp.layers[0/1])

Checkpoint compatibility: LoRALinear preserves the original 'weight' and 'bias'
parameter names, so SAM2.1 checkpoints load correctly.  The new 'lora_A' and
'lora_B' keys will appear as missing keys and must be ignored via
  ignore_missing_keys: ['*lora_A*', '*lora_B*']
in the trainer checkpoint config.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with Low-Rank Adaptation.

    Forward pass:  out = W x + b  +  scaling * B A x
    where (W, b) are frozen and (A, B) are the trainable LoRA matrices.
    B is zero-initialised so the model output is identical to the original at
    the start of training.

    State-dict compatibility: 'weight' and 'bias' sit at the same path as the
    original nn.Linear so checkpoint loading does not require key remapping.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_f = linear.in_features
        out_f = linear.out_features

        # Clone into new Parameters with the SAME attribute names so the state-
        # dict path (e.g. "blocks.0.attn.qkv.weight") is preserved.
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_f
        self.out_features = out_f

        # A: (rank × in_features),  B: (out_features × rank)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scaling = alpha / rank

        # Kaiming-uniform for A (matches nn.Linear default).  B stays zeros.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + delta

    def extra_repr(self) -> str:
        rank = self.lora_A.shape[0]
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={rank}, alpha={self.scaling * rank:.1f}"
        )


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 4.0,
    target_hiera: bool = True,
    target_mask_decoder: bool = True,
    target_hiera_mlp: bool = False,
    target_mask_decoder_mlp: bool = False,
) -> nn.Module:
    """
    Replace target nn.Linear layers in-place with LoRALinear wrappers.

    Args:
        model: SAM2Base / SAM2Train instance.
        rank: Low-rank dimension (4, 8, or 16 are typical choices).
        alpha: Scaling factor; effective scale = alpha / rank.
        target_hiera: Inject into Hiera image encoder attention (qkv + proj).
        target_mask_decoder: Inject into MaskDecoder transformer attention.
        target_hiera_mlp: Inject into Hiera MultiScaleBlock FFN (mlp.layers[0/1]).
        target_mask_decoder_mlp: Inject into MaskDecoder TwoWayAttentionBlock FFN.

    Returns:
        The same model object with LoRA layers injected.
    """
    hiera_count = 0
    if target_hiera:
        trunk = model.image_encoder.trunk
        for block in trunk.blocks:
            attn = block.attn
            attn.qkv = LoRALinear(attn.qkv, rank, alpha)
            attn.proj = LoRALinear(attn.proj, rank, alpha)
            hiera_count += 2
        logger.info("Injected LoRA into %d Hiera attention projections.", hiera_count)

    hiera_mlp_count = 0
    if target_hiera_mlp:
        trunk = model.image_encoder.trunk
        for block in trunk.blocks:
            mlp = block.mlp
            mlp.layers[0] = LoRALinear(mlp.layers[0], rank, alpha)
            mlp.layers[1] = LoRALinear(mlp.layers[1], rank, alpha)
            hiera_mlp_count += 2
        logger.info("Injected LoRA into %d Hiera MLP projections.", hiera_mlp_count)

    decoder_count = 0
    if target_mask_decoder:
        transformer = model.sam_mask_decoder.transformer
        for layer in transformer.layers:
            for attn_name in (
                "self_attn",
                "cross_attn_token_to_image",
                "cross_attn_image_to_token",
            ):
                attn = getattr(layer, attn_name)
                for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    lin = getattr(attn, proj_name)
                    setattr(attn, proj_name, LoRALinear(lin, rank, alpha))
                    decoder_count += 1
        # Final cross-attention layer
        for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            lin = getattr(transformer.final_attn_token_to_image, proj_name)
            setattr(transformer.final_attn_token_to_image, proj_name, LoRALinear(lin, rank, alpha))
            decoder_count += 1
        logger.info(
            "Injected LoRA into %d MaskDecoder attention projections.", decoder_count
        )

    decoder_mlp_count = 0
    if target_mask_decoder_mlp:
        transformer = model.sam_mask_decoder.transformer
        for layer in transformer.layers:
            mlp = layer.mlp
            mlp.layers[0] = LoRALinear(mlp.layers[0], rank, alpha)
            mlp.layers[1] = LoRALinear(mlp.layers[1], rank, alpha)
            decoder_mlp_count += 2
        logger.info(
            "Injected LoRA into %d MaskDecoder MLP projections.", decoder_mlp_count
        )

    total = hiera_count + hiera_mlp_count + decoder_count + decoder_mlp_count
    logger.info(
        "LoRA injection complete: %d layers, rank=%d, alpha=%.1f, scaling=%.3f",
        total,
        rank,
        alpha,
        alpha / rank,
    )
    return model


def freeze_non_lora_params(model: nn.Module) -> None:
    """
    Freeze all model parameters except LoRA A/B matrices.

    Call AFTER inject_lora().  The checkpoint loader will subsequently write the
    pretrained weights into the frozen LoRALinear.weight tensors without
    re-enabling gradients.
    """
    frozen = trainable = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()
    total = frozen + trainable
    logger.info(
        "Parameter budget: %d trainable / %d total (%.3f%% LoRA)",
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )


def lora_state_dict(model: nn.Module) -> dict:
    """Return only the LoRA A/B parameters for lightweight checkpointing."""
    return {
        k: v
        for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
