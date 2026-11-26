"""Image loading, hashing, and dual-resolution preprocessing for Kexia Facial Expression V4B.

Preserves the exact input dimensions (160x160 for MobileNetV2, 224x224 for ViT-B/16),
interpolation settings (`antialias=True`), and ImageNet normalization parameters
used during training and validation.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from .model import CLASS_ID

CNN_SIZE: int = 160
VIT_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def norm_label_string(s: str) -> str:
    """Normalize a string by case-folding and stripping diacritics and non-alphanumerics."""
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", s.casefold())
        if c.isalnum() and not unicodedata.combining(c)
    )


def get_label_aliases() -> Dict[str, str]:
    """Return normalized label aliases in English and Spanish mapped to canonical 7 classes."""
    aliases: Dict[str, str] = {}
    raw_map = {
        "angry": ["anger", "angry", "enojo", "ira"],
        "disgust": ["disgust", "disgusted", "asco", "disgusto"],
        "fear": ["fear", "fearful", "miedo"],
        "happy": ["happy", "happiness", "felicidad", "alegria"],
        "neutral": ["neutral", "neutrality", "neutro"],
        "sad": ["sad", "sadness", "tristeza"],
        "surprise": ["surprise", "surprised", "sorpresa"],
    }
    for cls_name, names in raw_map.items():
        aliases.update({norm_label_string(x): cls_name for x in names})
    return aliases


def read_image(path: Union[str, Path]) -> Image.Image:
    """Read an image file, apply EXIF orientation transposition, and convert to RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def compute_image_hashes(path: Union[str, Path], im: Image.Image) -> Tuple[str, str, str]:
    """Compute file SHA-256, decoded RGB pixel SHA-256, and 64-bit horizontal dHash.

    Args:
        path: Path to the image file on disk.
        im: Decoded RGB PIL Image.

    Returns:
        Tuple of `(file_sha256, pixel_sha256, dhash_hex)`.
    """
    p = Path(path)
    arr = np.asarray(im)
    pixel_hash = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
    gray = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    dhash = np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes().hex()
    file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    return file_hash, pixel_hash, dhash


def build_train_augment() -> T.Compose:
    """Build the training data augmentation pipeline used in the notebook."""
    return T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomAffine(12, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.ColorJitter(0.15, 0.15, 0.08),
        ]
    )


def build_eval_transform(
    size: int,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Build deterministic resize + tensor + ImageNet normalization transform."""
    return T.Compose(
        [
            T.Resize((size, size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


cnn_eval_transform = build_eval_transform(CNN_SIZE)
vit_eval_transform = build_eval_transform(VIT_SIZE)


def prepare_dual_inputs(
    image: Union[str, Path, Image.Image],
    cnn_size: int = CNN_SIZE,
    vit_size: int = VIT_SIZE,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare dual-resolution batch tensors `(x_cnn, y_vit)` for a single cropped face image.

    Args:
        image: File path or PIL.Image instance representing a cropped face.
        cnn_size: Spatial resolution for the CNN branch (default: 160).
        vit_size: Spatial resolution for the ViT branch (default: 224).
        device: Optional target device for the output tensors.

    Returns:
        Tuple `(x_cnn, y_vit)` with shapes `(1, 3, cnn_size, cnn_size)` and
        `(1, 3, vit_size, vit_size)`.
    """
    if isinstance(image, (str, Path)):
        im = read_image(image)
    elif isinstance(image, Image.Image):
        im = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")

    cnn_tf = cnn_eval_transform if cnn_size == CNN_SIZE else build_eval_transform(cnn_size)
    vit_tf = vit_eval_transform if vit_size == VIT_SIZE else build_eval_transform(vit_size)

    x_cnn = cnn_tf(im).unsqueeze(0)
    y_vit = vit_tf(im).unsqueeze(0)
    if device is not None:
        x_cnn = x_cnn.to(device)
        y_vit = y_vit.to(device)
    return x_cnn, y_vit


class FERDataset(Dataset):
    """Dataset returning dual-resolution tensors `(x_cnn, y_vit, label, index)`."""

    def __init__(self, frame, training: bool = False) -> None:
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.augment = build_train_augment() if training else None
        self.cnn_tf = cnn_eval_transform
        self.vit_tf = vit_eval_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        r = self.frame.iloc[i]
        im = read_image(Path(r.path))
        if self.training and self.augment is not None:
            im = self.augment(im)
        label = int(r.label) if "label" in r else CLASS_ID[r.cls]
        return self.cnn_tf(im), self.vit_tf(im), label, i
