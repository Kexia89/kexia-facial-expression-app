"""Kexia Facial Expression V4B package.

Hybrid MobileNetV2 + ViT-B/16 architecture with gated feature fusion for
seven-class visual facial expression classification.
"""

from .model import CLASSES, CLASS_ID, HybridFER, configure_stage, set_train_mode
from .preprocessing import (
    CNN_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIT_SIZE,
    build_eval_transform,
    build_train_augment,
    prepare_dual_inputs,
    read_image,
)
from .inference import ExpressionPredictor, load_checkpoint, predict_image

__all__ = [
    "CLASSES",
    "CLASS_ID",
    "CNN_SIZE",
    "VIT_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "HybridFER",
    "configure_stage",
    "set_train_mode",
    "read_image",
    "build_eval_transform",
    "build_train_augment",
    "prepare_dual_inputs",
    "load_checkpoint",
    "predict_image",
    "ExpressionPredictor",
]
