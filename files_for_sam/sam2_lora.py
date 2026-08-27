"""SAM2Train subclass that injects LoRA and freezes base weights at init time.

The flow when used with the Hydra trainer:
  1. SAM2TrainLoRA.__init__() builds the full SAM2 model, then injects LoRA and
     freezes all non-LoRA parameters.
  2. The trainer loads the SAM2.1 checkpoint into the model via
     load_state_dict_into_model(..., ignore_missing_keys=['*lora_A*', '*lora_B*']).
     The original weights land correctly in LoRALinear.weight (same state-dict
     path as the original nn.Linear.weight).  lora_A / lora_B are not in the
     checkpoint and remain with their random / zero initialisation.
  3. The optimizer sees only the trainable lora_A / lora_B parameters.
"""

import logging

from training.model.sam2 import SAM2Train
from sam2.modeling.lora import freeze_non_lora_params, inject_lora, lora_state_dict

logger = logging.getLogger(__name__)


class SAM2TrainLoRA(SAM2Train):
    """
    SAM2Train with Low-Rank Adaptation injected into image encoder and mask decoder.

    Extra constructor arguments (all others are forwarded to SAM2Train):
        lora_rank (int): rank of LoRA matrices (default 4).
        lora_alpha (float): scaling alpha; effective scale = alpha/rank (default = rank).
        lora_target_hiera (bool): inject into Hiera attention (qkv + proj) [default True].
        lora_target_mask_decoder (bool): inject into MaskDecoder attention [default True].
        lora_target_hiera_mlp (bool): inject into Hiera FFN layers [default False].
        lora_target_mask_decoder_mlp (bool): inject into MaskDecoder FFN layers [default False].
    """

    def __init__(
        self,
        *args,
        lora_rank: int = 4,
        lora_alpha: float = 4.0,
        lora_target_hiera: bool = True,
        lora_target_mask_decoder: bool = True,
        lora_target_hiera_mlp: bool = False,
        lora_target_mask_decoder_mlp: bool = False,
        **kwargs,
    ):
        # Build the full SAM2 model first.  freeze_image_encoder=False because we
        # will manage freezing ourselves via freeze_non_lora_params().
        kwargs.pop("freeze_image_encoder", None)
        super().__init__(*args, freeze_image_encoder=False, **kwargs)

        # Store LoRA hyperparams so the trainer banner can introspect them.
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_hiera = lora_target_hiera
        self.lora_target_mask_decoder = lora_target_mask_decoder
        self.lora_target_hiera_mlp = lora_target_hiera_mlp
        self.lora_target_mask_decoder_mlp = lora_target_mask_decoder_mlp

        # Inject LoRA matrices into targeted layers.
        inject_lora(
            self,
            rank=lora_rank,
            alpha=lora_alpha,
            target_hiera=lora_target_hiera,
            target_mask_decoder=lora_target_mask_decoder,
            target_hiera_mlp=lora_target_hiera_mlp,
            target_mask_decoder_mlp=lora_target_mask_decoder_mlp,
        )

        # Freeze everything except lora_A / lora_B.
        freeze_non_lora_params(self)

        n_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info("SAM2TrainLoRA ready - trainable params: %d", n_trainable)

    def save_lora_checkpoint(self, path: str) -> None:
        """Save only the LoRA weights (small file, sufficient for inference)."""
        import torch

        torch.save({"lora_state_dict": lora_state_dict(self)}, path)
        logger.info("Saved LoRA checkpoint to %s", path)
