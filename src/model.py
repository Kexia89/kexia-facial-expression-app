"""Model architecture and stage configuration for Kexia Facial Expression V4B.

Extracted verbatim in layer topology and parameter names from the experimental
notebook `Kexia_Facial_Expression_V4B.ipynb` to guarantee exact state_dict
compatibility with trained checkpoints (`best.pt`, `best_original.pt`, `last.pt`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchvision import models

CLASSES: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

CLASS_ID: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}


class HybridFER(nn.Module):
    """Hybrid MobileNetV2 + ViT-B/16 architecture with adaptive gated feature fusion.

    Attributes:
        mode: Operating mode ('hybrid', 'cnn', or 'vit').
        fusion: Fusion mechanism when mode == 'hybrid' ('gated' or 'concat').
        cnn: MobileNetV2 backbone with classifier replaced by Identity (1,280-D output).
        vit: ViT-B/16 backbone with heads replaced by Identity (768-D output).
        pc: 1,280 -> 512 projection block for the CNN branch.
        pv: 768 -> 512 projection block for the ViT branch.
        gate: Learnable sigmoid gate mapping concatenated 1,024-D features to 512-D weights.
        head: Classification head mapping fused features (512-D or 1,024-D) to 7 logits.
    """

    def __init__(
        self,
        pretrained: bool = False,
        mode: str = "hybrid",
        fusion: str = "gated",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.fusion = fusion
        assert mode in ("cnn", "vit", "hybrid") and fusion in ("gated", "concat"), (
            f"Invalid mode={mode!r} or fusion={fusion!r}"
        )

        self.cnn = (
            models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            )
            if mode != "vit"
            else None
        )
        self.vit = (
            models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
            )
            if mode != "cnn"
            else None
        )

        if self.cnn is not None:
            self.cnn.classifier = nn.Identity()
        if self.vit is not None:
            self.vit.heads = nn.Identity()

        def project(d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 512),
                nn.GELU(),
                nn.Dropout(0.25),
            )

        self.pc = project(1280) if self.cnn is not None else None
        self.pv = project(768) if self.vit is not None else None
        self.gate = (
            nn.Sequential(nn.Linear(1024, 512), nn.Sigmoid())
            if mode == "hybrid" and fusion == "gated"
            else None
        )
        d = 1024 if mode == "hybrid" and fusion == "concat" else 512
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 7),
        )

    def forward(
        self,
        x: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        return_gate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass through the dual-branch architecture.

        Args:
            x: Tensor of shape (B, 3, 160, 160) for MobileNetV2.
            y: Tensor of shape (B, 3, 224, 224) for ViT-B/16.
            return_gate: If True, returns a tuple `(logits, gate_activation)`.

        Returns:
            Logits tensor of shape (B, 7), or `(logits, gate)` if `return_gate=True`.
        """
        c = self.pc(self.cnn(x)) if self.cnn is not None else None
        v = self.pv(self.vit(y)) if self.vit is not None else None
        g: Optional[torch.Tensor] = None
        if self.mode == "hybrid":
            assert c is not None and v is not None
            z = torch.cat([c, v], 1)
            if self.gate is not None:
                g = self.gate(z)
                z = g * c + (1 - g) * v
        else:
            z = c if c is not None else v
        logits = self.head(z)
        return (logits, g) if return_gate else logits


def configure_stage(model: HybridFER, stage: str) -> None:
    """Configure parameter trainability for progressive fine-tuning stages A, B, or C.

    - Stage A: Only projections (`pc`, `pv`), `gate`, and `head` are trainable.
    - Stage B: Unfreezes the last 4 blocks of MobileNetV2 and last 2 encoder blocks
      (plus final LayerNorm) of ViT-B/16.
    - Stage C: Unfreezes the last 8 blocks of MobileNetV2 and last 4 encoder blocks
      (plus final LayerNorm) of ViT-B/16.
    - BatchNorm2d parameters in MobileNetV2 remain frozen across all stages.
    """
    assert stage in ("A", "B", "C"), f"Unknown stage: {stage!r}"
    for p in model.parameters():
        p.requires_grad_(True)
    for b in [model.cnn, model.vit]:
        if b is not None:
            for p in b.parameters():
                p.requires_grad_(False)
    if stage != "A":
        if model.cnn is not None:
            for block in list(model.cnn.features)[-(4 if stage == "B" else 8) :]:
                for p in block.parameters():
                    p.requires_grad_(True)
        if model.vit is not None:
            for block in list(model.vit.encoder.layers)[-(2 if stage == "B" else 4) :]:
                for p in block.parameters():
                    p.requires_grad_(True)
            for p in model.vit.encoder.ln.parameters():
                p.requires_grad_(True)
    if model.cnn is not None:
        for m in model.cnn.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad_(False)


def set_train_mode(model: HybridFER) -> None:
    """Set model to training mode while keeping frozen backbones and BatchNorm2d in eval mode."""
    model.train()
    for b in [model.cnn, model.vit]:
        if b is not None and not any(p.requires_grad for p in b.parameters()):
            b.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
