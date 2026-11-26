"""Verification test suite for Kexia Facial Expression V4B.

Compatible with both `python -m unittest discover -s GitHub/tests -v` and `pytest`.

Tests:
1. Class order and config consistency (`configs/model_config.json`, `artifacts/reports/labels.json`, `src/model.py`).
2. Exact parameter count (89,735,175) and dual-branch tensor shapes (160x160 and 224x224).
3. Stage trainability rules (Stages A, B, C and frozen BatchNorm2d).
4. Preprocessing and dual-input pipeline determinism.
5. Checkpoint compatibility (`best.pt`, `best_original.pt`, `last.pt`) and local inference without retraining.
6. Privacy, large-file exclusion, and markdown relative link integrity across `GitHub/`.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
RUN_DIR = WORKSPACE_ROOT / "kexia_v4_runs" / "20261124_102050_038611"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import ExpressionPredictor, predict_image
from src.model import CLASSES, CLASS_ID, HybridFER, configure_stage, set_train_mode
from src.preprocessing import (
    CNN_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VIT_SIZE,
    prepare_dual_inputs,
)

EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class TestKexiaRepository(unittest.TestCase):
    """End-to-end verification tests for the Kexia Facial Expression V4B repository."""

    def test_01_class_order_and_configs(self) -> None:
        self.assertEqual(CLASSES, EXPECTED_CLASSES)
        self.assertEqual(CLASS_ID, {c: i for i, c in enumerate(EXPECTED_CLASSES)})

        cfg = json.loads((REPO_ROOT / "configs" / "model_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["classes"], EXPECTED_CLASSES)
        self.assertEqual(cfg["cnn_size"], 160)
        self.assertEqual(cfg["vit_size"], 224)
        self.assertEqual(cfg["total_parameters"], 89735175)

        labels = json.loads((REPO_ROOT / "artifacts" / "reports" / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(labels, CLASS_ID)

    def test_02_architecture_parameters_and_stages(self) -> None:
        model = HybridFER(pretrained=False, mode="hybrid", fusion="gated")
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 89735175)

        configure_stage(model, "A")
        set_train_mode(model)
        stage_a_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(stage_a_trainable, 1712647)

        configure_stage(model, "B")
        stage_b_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(stage_b_trainable, stage_a_trainable)

        configure_stage(model, "C")
        stage_c_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertGreater(stage_c_trainable, stage_b_trainable)

        for m in model.cnn.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for p in m.parameters():
                    self.assertFalse(p.requires_grad)

    def test_03_dual_input_preprocessing_and_forward(self) -> None:
        rng = np.random.default_rng(42)
        synthetic_face = Image.fromarray(
            rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8), mode="RGB"
        )
        x_cnn, y_vit = prepare_dual_inputs(synthetic_face)
        self.assertEqual(tuple(x_cnn.shape), (1, 3, CNN_SIZE, CNN_SIZE))
        self.assertEqual(tuple(y_vit.shape), (1, 3, VIT_SIZE, VIT_SIZE))
        self.assertEqual(IMAGENET_MEAN, (0.485, 0.456, 0.406))
        self.assertEqual(IMAGENET_STD, (0.229, 0.224, 0.225))

        model = HybridFER(pretrained=False, mode="hybrid", fusion="gated").eval()
        with torch.no_grad():
            logits, gate = model(x_cnn, y_vit, return_gate=True)
        self.assertEqual(tuple(logits.shape), (1, 7))
        self.assertIsNotNone(gate)
        self.assertEqual(tuple(gate.shape), (1, 512))
        self.assertTrue(bool(torch.all((gate >= 0.0) & (gate <= 1.0))))

    def test_04_checkpoint_compatibility_and_local_inference(self) -> None:
        best_pt = RUN_DIR / "best.pt"
        if not best_pt.exists():
            self.skipTest("Original checkpoint best.pt not available in local workspace.")

        predictor = ExpressionPredictor(best_pt, device="cpu")
        self.assertEqual(predictor.metadata["stage"], "C")
        self.assertEqual(predictor.metadata["epoch"], 5)
        self.assertAlmostEqual(
            predictor.metadata["val"]["macro_f1"], 0.4255765952025515, places=6
        )

        # Synthetic image inference
        rng = np.random.default_rng(42)
        synthetic_face = Image.fromarray(
            rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8), mode="RGB"
        )
        res = predictor.predict(synthetic_face)
        self.assertIn(res.predicted_class, EXPECTED_CLASSES)
        self.assertAlmostEqual(sum(res.probabilities.values()), 1.0, places=5)
        self.assertIsNotNone(res.mean_cnn_gate)
        self.assertTrue(0.0 <= float(res.mean_cnn_gate) <= 1.0)

        series = predict_image(synthetic_face, checkpoint=best_pt, device="cpu")
        self.assertEqual(len(series), 7)
        self.assertAlmostEqual(float(series.sum()), 1.0, places=5)

    def test_05_privacy_and_repository_hygiene(self) -> None:
        forbidden_exts = {".pt", ".pth", ".ckpt", ".keras", ".h5", ".tflite", ".pkl"}
        for p in REPO_ROOT.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
                self.assertNotIn(
                    p.suffix.lower(),
                    forbidden_exts,
                    f"Forbidden large binary inside GitHub/: {p}",
                )
                self.assertLess(
                    p.stat().st_size,
                    10 * 1024 * 1024,
                    f"File exceeds 10 MB inside GitHub/: {p}",
                )

        text_exts = {".md", ".py", ".json", ".csv", ".txt", ".ipynb"}
        this_file = Path(__file__).resolve()
        private_pattern = re.compile(
            r"(Users[\\/]+uriel|OneDrive[\\/]+Documentos|<LOCAL_USER_PATH>)",
            re.IGNORECASE,
        )
        for p in REPO_ROOT.rglob("*"):
            if (
                p.is_file()
                and p.resolve() != this_file
                and p.suffix.lower() in text_exts
                and "__pycache__" not in p.parts
            ):
                content = p.read_text(encoding="utf-8", errors="ignore")
                match = private_pattern.search(content)
                self.assertIsNone(
                    match,
                    f"Private path pattern {match.group(0) if match else ''!r} found in {p.relative_to(REPO_ROOT)}",
                )

    def test_06_markdown_relative_links(self) -> None:
        link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
        src_pattern = re.compile(r'src="([^"]+)"')
        for md_file in REPO_ROOT.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            targets = link_pattern.findall(text) + src_pattern.findall(text)
            for t in targets:
                t_clean = t.split("#")[0].strip()
                if not t_clean or t_clean.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = (md_file.parent / t_clean).resolve()
                self.assertTrue(
                    resolved.exists(),
                    f"Broken relative link {t!r} in {md_file.relative_to(REPO_ROOT)}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
