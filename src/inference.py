"""Reproducible inference interface for Kexia Facial Expression V4B.

IMPORTANT SCIENTIFIC DISCLAIMER:
1. Input expectation: The model operates on pre-cropped facial images. It does
   not include an integrated face detector or facial landmark alignment stage.
2. Interpretation of outputs: Softmax values represent an uncalibrated conditional
   probability distribution over the seven visual expression labels defined in the
   training corpus (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).
   They do NOT represent calibrated confidence, statistical certainty, or a person's
   internal, subjective, clinical, or psychological emotional state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image

from .model import CLASSES, HybridFER
from .preprocessing import CNN_SIZE, VIT_SIZE, prepare_dual_inputs

INTERPRETATION_NOTE = (
    "Softmax outputs represent uncalibrated scores over the 7 dataset visual expression "
    "classes for a cropped facial image. They do not measure model certainty or infer "
    "internal emotional, psychological, or clinical states."
)


@dataclass(frozen=True)
class PredictionResult:
    """Structured output of a single-image facial expression prediction."""

    predicted_class: str
    predicted_index: int
    top_probability: float
    probabilities: Dict[str, float]
    mean_cnn_gate: Optional[float]
    checkpoint_stage: Optional[str]
    checkpoint_epoch: Optional[int]
    note: str = INTERPRETATION_NOTE

    def to_series(self) -> pd.Series:
        """Return class probabilities sorted in descending order (matches notebook API)."""
        return pd.Series(self.probabilities).sort_values(ascending=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prediction result to a JSON-compatible dictionary."""
        return {
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "top_probability": self.top_probability,
            "probabilities": self.probabilities,
            "mean_cnn_gate": self.mean_cnn_gate,
            "checkpoint_stage": self.checkpoint_stage,
            "checkpoint_epoch": self.checkpoint_epoch,
            "note": self.note,
        }


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[HybridFER, Dict[str, Any]]:
    """Load a trained HybridFER checkpoint and verify architectural and class consistency.

    Args:
        checkpoint_path: Path to `.pt` checkpoint (`best.pt`, `best_original.pt`, or `last.pt`).
        device: Target torch device (defaults to CUDA if available, else CPU).

    Returns:
        Tuple `(model, metadata)` where `model` is in eval mode and `metadata` holds
        checkpoint config, stage, epoch, and validation metrics.
    """
    target_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ck = torch.load(path, map_location=target_device, weights_only=True)
    if "config" not in ck or "state_dict" not in ck:
        raise KeyError("Checkpoint must contain 'config' and 'state_dict' keys.")

    cfg = ck["config"]
    if list(cfg.get("classes", [])) != CLASSES:
        raise ValueError(
            f"Checkpoint class order {cfg.get('classes')} does not match expected {CLASSES}."
        )

    mode = cfg.get("mode", "hybrid")
    fusion = cfg.get("fusion", "gated")
    model = HybridFER(pretrained=False, mode=mode, fusion=fusion)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(target_device)
    model.eval()

    metadata = {
        "config": cfg,
        "stage": ck.get("stage"),
        "epoch": ck.get("epoch"),
        "val": ck.get("val"),
    }
    return model, metadata


class ExpressionPredictor:
    """Reusable inference wrapper that loads a checkpoint once and scores cropped face images."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.metadata = load_checkpoint(checkpoint_path, device=self.device)
        cfg = self.metadata["config"]
        self.cnn_size: int = int(cfg.get("cnn_size", CNN_SIZE))
        self.vit_size: int = int(cfg.get("vit_size", VIT_SIZE))
        self.classes: List[str] = list(cfg.get("classes", CLASSES))

    @torch.no_grad()
    def predict(
        self, image: Union[str, Path, Image.Image]
    ) -> PredictionResult:
        """Run inference on a single cropped facial image (path or PIL.Image)."""
        self.model.eval()
        x_cnn, y_vit = prepare_dual_inputs(
            image,
            cnn_size=self.cnn_size,
            vit_size=self.vit_size,
            device=self.device,
        )
        logits, gate = self.model(x_cnn, y_vit, return_gate=True)
        probs_arr = logits.float().softmax(dim=1)[0].cpu().numpy()
        pred_idx = int(probs_arr.argmax())
        probs_dict = {
            cls_name: float(probs_arr[idx])
            for idx, cls_name in enumerate(self.classes)
        }
        sorted_probs = dict(
            sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
        )
        mean_gate = (
            float(gate.float().mean(dim=1)[0].cpu().item())
            if gate is not None
            else None
        )
        return PredictionResult(
            predicted_class=self.classes[pred_idx],
            predicted_index=pred_idx,
            top_probability=float(probs_arr[pred_idx]),
            probabilities=sorted_probs,
            mean_cnn_gate=mean_gate,
            checkpoint_stage=self.metadata.get("stage"),
            checkpoint_epoch=self.metadata.get("epoch"),
        )


@torch.no_grad()
def predict_image(
    path: Union[str, Path, Image.Image],
    checkpoint: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> pd.Series:
    """Notebook-compatible convenience function returning sorted softmax probabilities.

    Note:
        Softmax probabilities reflect alignment with the training labels on cropped
        facial images; they are neither calibrated confidence scores nor indicators
        of internal emotional states.
    """
    predictor = ExpressionPredictor(checkpoint_path=checkpoint, device=device)
    result = predictor.predict(path)
    return result.to_series()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kexia Facial Expression V4B inference on a cropped face image."
    )
    parser.add_argument("--image", required=True, help="Path to a cropped facial image.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint.")
    parser.add_argument(
        "--device",
        default=None,
        help="Computation device ('cuda' or 'cpu'). Defaults to auto-detection.",
    )
    args = parser.parse_args()

    predictor = ExpressionPredictor(args.checkpoint, device=args.device)
    result = predictor.predict(args.image)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
