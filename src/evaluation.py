"""Evaluation metrics, reporting, and scientific visualization utilities for Kexia Facial Expression V4B.

Provides functions to compute classification metrics from model outputs or saved
artifacts and to generate publication-grade figures using the sober white-background
palette (deep blue, slate grey, and turquoise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import CLASSES, HybridFER

# Sober scientific palette: deep blue, turquoise, slate grey
COLOR_BLUE = "#1F4E79"
COLOR_BLUE_LIGHT = "#4A7BB0"
COLOR_TURQUOISE = "#1B9E8C"
COLOR_TURQUOISE_LIGHT = "#65C3B6"
COLOR_GREY_DARK = "#3B4252"
COLOR_GREY = "#7B8898"
COLOR_GREY_LIGHT = "#D8DEE9"
COLOR_BG = "#FFFFFF"


def apply_scientific_style() -> None:
    """Configure Matplotlib rcParams for sober scientific figures on white background."""
    plt.rcParams.update(
        {
            "figure.facecolor": COLOR_BG,
            "axes.facecolor": COLOR_BG,
            "savefig.facecolor": COLOR_BG,
            "axes.edgecolor": "#B0B8C4",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#E6EAF0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": COLOR_GREY_DARK,
            "ytick.color": COLOR_GREY_DARK,
            "text.color": COLOR_GREY_DARK,
            "axes.labelcolor": COLOR_GREY_DARK,
            "legend.frameon": True,
            "legend.facecolor": COLOR_BG,
            "legend.edgecolor": "#D0D7DE",
            "legend.fontsize": 9,
        }
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loss: Optional[float] = None,
    gates: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute validation/test aggregate metrics matching the notebook specification."""
    metrics: Dict[str, float] = {}
    if loss is not None:
        metrics["loss"] = float(loss)
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    )
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    if gates:
        metrics["mean_cnn_gate"] = float(np.mean(gates))
    return metrics


@torch.no_grad()
def evaluate_dataloader(
    model: HybridFER,
    loader: DataLoader,
    device: Union[str, torch.device],
    desc: str = "eval",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, List[int], List[float]]:
    """Evaluate a HybridFER model over a DataLoader returning `(x_cnn, y_vit, target, idx)`."""
    target_device = torch.device(device)
    model.eval()
    amp = target_device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    ys: List[int] = []
    probs: List[np.ndarray] = []
    ids: List[int] = []
    gates: List[float] = []
    total_loss = 0.0
    count = 0

    for x, y, t, i in tqdm(loader, desc=desc, leave=False):
        x, y, t = x.to(target_device), y.to(target_device), t.to(target_device)
        with torch.autocast(device_type=target_device.type, dtype=amp_dtype, enabled=amp):
            logits, g = model(x, y, return_gate=True)
        total_loss += nn.functional.cross_entropy(logits.float(), t, reduction="sum").item()
        count += len(t)
        ys.extend(t.cpu().tolist())
        probs.extend(logits.float().softmax(1).cpu().numpy())
        ids.extend(i.tolist() if hasattr(i, "tolist") else list(i))
        if g is not None:
            gates.extend(g.float().mean(1).cpu().tolist())

    ys_arr = np.array(ys)
    probs_arr = np.array(probs)
    pred_arr = probs_arr.argmax(1)
    metrics = compute_classification_metrics(
        ys_arr, pred_arr, loss=total_loss / max(1, count), gates=gates
    )
    return metrics, ys_arr, pred_arr, probs_arr, ids, gates


def build_epoch_labels(df: pd.DataFrame) -> List[str]:
    """Build stage-epoch labels like 'A1', 'B5', 'C5', 'C12'."""
    return [f"{row.stage}{int(row.epoch)}" for _, row in df.iterrows()]


def _add_stage_boundaries(ax: plt.Axes, df: pd.DataFrame, best_idx: Optional[int] = None) -> None:
    """Add vertical stage transition markers and recovered/selected checkpoint annotations."""
    stages = df["stage"].tolist()
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            ax.axvline(i + 0.5, color=COLOR_GREY, linestyle=":", linewidth=1.1, alpha=0.85)
    # Mark B5 resumption boundary if present
    b5_matches = df.index[(df["stage"] == "B") & (df["epoch"] == 5)].tolist()
    if b5_matches:
        b5_x = b5_matches[0] + 1
        ax.axvline(b5_x, color=COLOR_GREY_DARK, linestyle="--", linewidth=1.0, alpha=0.7)
    if best_idx is not None:
        ax.axvline(best_idx + 1, color=COLOR_TURQUOISE, linestyle="-.", linewidth=1.2, alpha=0.9)


def generate_all_scientific_figures(
    metrics_dir: Union[str, Path],
    reports_dir: Union[str, Path],
    figures_dir: Union[str, Path],
) -> List[Path]:
    """Generate all 8 required scientific visualizations strictly from real experimental artifacts."""
    apply_scientific_style()
    metrics_path = Path(metrics_dir)
    reports_path = Path(reports_dir)
    out_path = Path(figures_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(metrics_path / "metrics_history.csv")
    df_res = pd.read_csv(metrics_path / "metrics_resumed.csv")
    df_report = pd.read_csv(metrics_path / "validation_classification_report.csv", index_col=0)
    df_cm = pd.read_csv(metrics_path / "validation_confusion_matrix.csv", index_col=0)
    df_dist = pd.read_csv(reports_path / "distribution.csv", index_col=0)
    val_metrics = json.loads((metrics_path / "validation_metrics.json").read_text(encoding="utf-8"))

    labels_res = build_epoch_labels(df_res)
    x_res = np.arange(1, len(df_res) + 1)
    best_idx = int(df_res["val_macro_f1"].idxmax())
    b5_idx = int(df_res.index[(df_res["stage"] == "B") & (df_res["epoch"] == 5)][0])
    # Interrupted original B6-B8 branch
    df_orig_tail = df_orig[df_orig["stage"] == "B"].iloc[4:]  # B5..B8
    x_orig_tail = np.arange(b5_idx + 1, b5_idx + 1 + len(df_orig_tail))

    created: List[Path] = []

    # 1. Training vs Validation Accuracy
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.plot(x_res, df_res["train_accuracy"] * 100, color=COLOR_BLUE, linewidth=2.2, label="Train Accuracy (Resumed branch)")
    ax.plot(x_res, df_res["val_accuracy"] * 100, color=COLOR_TURQUOISE, linewidth=2.2, label="Validation Accuracy (Resumed branch)")
    ax.plot(
        x_orig_tail,
        df_orig_tail["val_accuracy"] * 100,
        color=COLOR_GREY,
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        label="Validation Accuracy (Initial interrupted B6–B8)",
    )
    _add_stage_boundaries(ax, df_res, best_idx=best_idx)
    ax.scatter(
        [b5_idx + 1, best_idx + 1],
        [df_res.loc[b5_idx, "val_accuracy"] * 100, df_res.loc[best_idx, "val_accuracy"] * 100],
        color=[COLOR_GREY_DARK, COLOR_TURQUOISE],
        zorder=5,
        s=45,
    )
    ax.annotate(
        f"Recovered B5\n({df_res.loc[b5_idx, 'val_accuracy']*100:.2f}%)",
        xy=(b5_idx + 1, df_res.loc[b5_idx, "val_accuracy"] * 100),
        xytext=(b5_idx - 2.5, df_res.loc[b5_idx, "val_accuracy"] * 100 - 7.5),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREY_DARK, lw=0.9),
        fontsize=8.5,
    )
    ax.annotate(
        f"Selected C5\n({df_res.loc[best_idx, 'val_accuracy']*100:.2f}%)",
        xy=(best_idx + 1, df_res.loc[best_idx, "val_accuracy"] * 100),
        xytext=(best_idx - 3.5, df_res.loc[best_idx, "val_accuracy"] * 100 + 5.5),
        arrowprops=dict(arrowstyle="->", color=COLOR_TURQUOISE, lw=0.9),
        fontsize=8.5,
        fontweight="bold",
    )
    ax.set_xticks(x_res[::2])
    ax.set_xticklabels([labels_res[i] for i in range(0, len(labels_res), 2)])
    ax.set_xlabel("Progressive Training Epoch (Stage A → Stage B → Stage C)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Kexia V4B — Training and Validation Accuracy Across Progressive Stages")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p1 = out_path / "training_validation_accuracy.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    created.append(p1)

    # 2. Macro F1 Evolution
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.plot(
        x_res,
        df_res["val_macro_f1"] * 100,
        color=COLOR_BLUE,
        linewidth=2.3,
        marker="o",
        markersize=3.8,
        label="Validation Macro F1 (Resumed run)",
    )
    ax.plot(
        x_res,
        df_res["val_balanced_accuracy"] * 100,
        color=COLOR_TURQUOISE,
        linewidth=1.8,
        linestyle="-.",
        label="Validation Balanced Accuracy",
    )
    ax.plot(
        x_orig_tail,
        df_orig_tail["val_macro_f1"] * 100,
        color=COLOR_GREY,
        linestyle="--",
        linewidth=1.5,
        marker="s",
        markersize=3.5,
        label="Initial interrupted B6–B8 Macro F1",
    )
    _add_stage_boundaries(ax, df_res, best_idx=best_idx)
    ax.scatter(
        [b5_idx + 1, best_idx + 1],
        [df_res.loc[b5_idx, "val_macro_f1"] * 100, df_res.loc[best_idx, "val_macro_f1"] * 100],
        color=[COLOR_GREY_DARK, COLOR_BLUE],
        s=55,
        zorder=6,
    )
    ax.annotate(
        f"Recovered B5: {df_res.loc[b5_idx, 'val_macro_f1']*100:.2f}%",
        xy=(b5_idx + 1, df_res.loc[b5_idx, "val_macro_f1"] * 100),
        xytext=(b5_idx - 4.0, df_res.loc[b5_idx, "val_macro_f1"] * 100 + 2.2),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREY_DARK, lw=0.9),
        fontsize=8.5,
    )
    ax.annotate(
        f"Selected Best C5: {df_res.loc[best_idx, 'val_macro_f1']*100:.2f}%",
        xy=(best_idx + 1, df_res.loc[best_idx, "val_macro_f1"] * 100),
        xytext=(best_idx - 6.5, df_res.loc[best_idx, "val_macro_f1"] * 100 + 1.8),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=1.0),
        fontsize=9,
        fontweight="bold",
    )
    ax.set_xticks(x_res[::2])
    ax.set_xticklabels([labels_res[i] for i in range(0, len(labels_res), 2)])
    ax.set_xlabel("Progressive Training Epoch (Stage A → Stage B → Stage C)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Kexia V4B — Validation Macro F1 Evolution & Checkpoint Selection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p2 = out_path / "macro_f1_evolution.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    created.append(p2)

    # 3. Loss Curves
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=200)
    ax.plot(x_res, df_res["train_loss"], color=COLOR_BLUE, linewidth=2.2, label="Training Weighted Cross-Entropy")
    ax.plot(x_res, df_res["val_loss"], color=COLOR_TURQUOISE, linewidth=2.2, label="Validation Cross-Entropy")
    ax.plot(
        x_orig_tail,
        df_orig_tail["val_loss"],
        color=COLOR_GREY,
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        label="Validation Loss (Initial interrupted B6–B8)",
    )
    _add_stage_boundaries(ax, df_res, best_idx=best_idx)
    ax.scatter(
        [b5_idx + 1, best_idx + 1, len(df_res)],
        [df_res.loc[b5_idx, "val_loss"], df_res.loc[best_idx, "val_loss"], df_res.iloc[-1]["val_loss"]],
        color=[COLOR_GREY_DARK, COLOR_TURQUOISE, COLOR_BLUE],
        s=45,
        zorder=5,
    )
    ax.annotate(
        f"B5 ({df_res.loc[b5_idx, 'val_loss']:.4f})",
        xy=(b5_idx + 1, df_res.loc[b5_idx, "val_loss"]),
        xytext=(b5_idx - 2.5, df_res.loc[b5_idx, "val_loss"] - 0.14),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREY_DARK, lw=0.9),
        fontsize=8.5,
    )
    ax.annotate(
        f"C5 Selected ({df_res.loc[best_idx, 'val_loss']:.4f})",
        xy=(best_idx + 1, df_res.loc[best_idx, "val_loss"]),
        xytext=(best_idx - 4.0, df_res.loc[best_idx, "val_loss"] + 0.12),
        arrowprops=dict(arrowstyle="->", color=COLOR_TURQUOISE, lw=0.9),
        fontsize=8.5,
        fontweight="bold",
    )
    ax.annotate(
        f"C12 Early Stop ({df_res.iloc[-1]['val_loss']:.4f})",
        xy=(len(df_res), df_res.iloc[-1]["val_loss"]),
        xytext=(len(df_res) - 7.5, df_res.iloc[-1]["val_loss"] + 0.04),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=0.9),
        fontsize=8.5,
    )
    ax.set_xticks(x_res[::2])
    ax.set_xticklabels([labels_res[i] for i in range(0, len(labels_res), 2)])
    ax.set_xlabel("Progressive Training Epoch (Stage A → Stage B → Stage C)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Kexia V4B — Training and Validation Cross-Entropy Loss Curves")
    ax.legend(loc="lower left")
    fig.tight_layout()
    p3 = out_path / "loss_curves.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    created.append(p3)

    # 4. Per-class Precision, Recall, and F1-score on Validation (C5)
    cls_rows = df_report.loc[CLASSES]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    idx = np.arange(len(CLASSES))
    width = 0.26
    bars_p = ax.bar(idx - width, cls_rows["precision"] * 100, width, color=COLOR_GREY, label="Precision")
    bars_r = ax.bar(idx, cls_rows["recall"] * 100, width, color=COLOR_TURQUOISE, label="Recall")
    bars_f = ax.bar(idx + width, cls_rows["f1-score"] * 100, width, color=COLOR_BLUE, label="F1-Score")
    ax.axhline(
        val_metrics["macro_f1"] * 100,
        color=COLOR_BLUE,
        linestyle="--",
        linewidth=1.2,
        label=f"Macro F1 ({val_metrics['macro_f1']*100:.2f}%)",
    )
    for b in bars_f:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 1.1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=COLOR_BLUE,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{c.capitalize()}\n(n={int(cls_rows.loc[c, 'support']):,})" for c in CLASSES])
    ax.set_ylim(0, 90)
    ax.set_ylabel("Metric (%)")
    ax.set_title("Kexia V4B (Checkpoint C5) — Per-Class Precision, Recall, and F1-Score (Validation)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p4 = out_path / "per_class_f1.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    created.append(p4)

    # 5 & 6. Confusion Matrices (Absolute and Row-Normalized)
    cm_abs = df_cm.loc[CLASSES, CLASSES].to_numpy(dtype=int)
    cm_norm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

    from matplotlib.colors import LinearSegmentedColormap

    scientific_cmap = LinearSegmentedColormap.from_list(
        "kexia_scientific", ["#FFFFFF", "#D7ECE9", "#65C3B6", "#2B7A9B", "#1F4E79"]
    )

    # 5. Absolute Confusion Matrix
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=200)
    ax.grid(False)
    im = ax.imshow(cm_abs, cmap=scientific_cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Validation Image Count")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=35, ha="right")
    ax.set_yticklabels([c.capitalize() for c in CLASSES])
    thresh = cm_abs.max() * 0.52
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_abs[i, j]
            ax.text(
                j,
                i,
                f"{val:,}",
                ha="center",
                va="center",
                color="white" if val > thresh else COLOR_GREY_DARK,
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xlabel("Predicted Expression Label")
    ax.set_ylabel("True Expression Label")
    ax.set_title("Kexia V4B (Checkpoint C5) — Validation Confusion Matrix (Absolute Counts)")
    fig.tight_layout()
    p5 = out_path / "confusion_matrix_absolute.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    created.append(p5)

    # 6. Row-Normalized Confusion Matrix
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=200)
    ax.grid(False)
    im = ax.imshow(cm_norm * 100, cmap=scientific_cmap, vmin=0, vmax=100, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-Normalized Recall (%)")
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels([c.capitalize() for c in CLASSES], rotation=35, ha="right")
    ax.set_yticklabels([c.capitalize() for c in CLASSES])
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_norm[i, j] * 100
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color="white" if val > 48 else COLOR_GREY_DARK,
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
            )
    ax.set_xlabel("Predicted Expression Label")
    ax.set_ylabel("True Expression Label")
    ax.set_title("Kexia V4B (Checkpoint C5) — Validation Confusion Matrix (Row-Normalized %)")
    fig.tight_layout()
    p6 = out_path / "confusion_matrix_normalized.png"
    fig.savefig(p6, dpi=200)
    plt.close(fig)
    created.append(p6)

    # 7. Class Distribution Across Partitions
    df_dist_ord = df_dist.loc[CLASSES, ["train", "validation", "test"]]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    idx = np.arange(len(CLASSES))
    width = 0.27
    b_tr = ax.bar(idx - width, df_dist_ord["train"], width, color=COLOR_BLUE, label="Train (n=115,288)")
    b_va = ax.bar(idx, df_dist_ord["validation"], width, color=COLOR_TURQUOISE, label="Validation (n=17,017)")
    b_te = ax.bar(idx + width, df_dist_ord["test"], width, color=COLOR_GREY, label="Test (n=24,090)")
    for b in b_tr:
        h = int(b.get_height())
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 450,
            f"{h:,}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=COLOR_BLUE,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylim(0, 39000)
    ax.set_ylabel("Retained Image Count")
    ax.set_title("Unified Facial Expression Corpus — Class Distribution Across Partitions (Total: 156,395)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    p7 = out_path / "class_distribution.png"
    fig.savefig(p7, dpi=200)
    plt.close(fig)
    created.append(p7)

    # 8. Generalization Gap Evolution (Accuracy Gap & Loss Gap)
    acc_gap = (df_res["train_accuracy"] - df_res["val_accuracy"]) * 100
    loss_gap = df_res["val_loss"] - df_res["train_loss"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=200)

    ax1.plot(x_res, acc_gap, color=COLOR_BLUE, linewidth=2.2, marker="o", markersize=3.2)
    ax1.fill_between(x_res, 0, acc_gap, color=COLOR_BLUE_LIGHT, alpha=0.16)
    _add_stage_boundaries(ax1, df_res, best_idx=best_idx)
    ax1.scatter(
        [b5_idx + 1, best_idx + 1, len(df_res)],
        [acc_gap.iloc[b5_idx], acc_gap.iloc[best_idx], acc_gap.iloc[-1]],
        color=[COLOR_GREY_DARK, COLOR_TURQUOISE, COLOR_BLUE],
        s=45,
        zorder=5,
    )
    ax1.annotate(
        f"B5: {acc_gap.iloc[b5_idx]:.2f} pp",
        xy=(b5_idx + 1, acc_gap.iloc[b5_idx]),
        xytext=(b5_idx - 3.5, acc_gap.iloc[b5_idx] - 5.0),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREY_DARK, lw=0.8),
        fontsize=8,
    )
    ax1.annotate(
        f"C5: {acc_gap.iloc[best_idx]:.2f} pp",
        xy=(best_idx + 1, acc_gap.iloc[best_idx]),
        xytext=(best_idx - 5.5, acc_gap.iloc[best_idx] + 3.2),
        arrowprops=dict(arrowstyle="->", color=COLOR_TURQUOISE, lw=0.8),
        fontsize=8.5,
        fontweight="bold",
    )
    ax1.annotate(
        f"C12: {acc_gap.iloc[-1]:.2f} pp",
        xy=(len(df_res), acc_gap.iloc[-1]),
        xytext=(len(df_res) - 7.5, acc_gap.iloc[-1] - 4.5),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=0.8),
        fontsize=8.5,
    )
    ax1.set_xticks(x_res[::3])
    ax1.set_xticklabels([labels_res[i] for i in range(0, len(labels_res), 3)])
    ax1.set_xlabel("Progressive Epoch")
    ax1.set_ylabel("Train − Validation Accuracy (percentage points)")
    ax1.set_title("Accuracy Generalization Gap")

    ax2.plot(x_res, loss_gap, color=COLOR_TURQUOISE, linewidth=2.2, marker="s", markersize=3.2)
    ax2.fill_between(x_res, 0, loss_gap, color=COLOR_TURQUOISE_LIGHT, alpha=0.18)
    _add_stage_boundaries(ax2, df_res, best_idx=best_idx)
    ax2.scatter(
        [b5_idx + 1, best_idx + 1, len(df_res)],
        [loss_gap.iloc[b5_idx], loss_gap.iloc[best_idx], loss_gap.iloc[-1]],
        color=[COLOR_GREY_DARK, COLOR_TURQUOISE, COLOR_BLUE],
        s=45,
        zorder=5,
    )
    ax2.annotate(
        f"B5: {loss_gap.iloc[b5_idx]:.3f}",
        xy=(b5_idx + 1, loss_gap.iloc[b5_idx]),
        xytext=(b5_idx - 3.5, loss_gap.iloc[b5_idx] - 0.18),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREY_DARK, lw=0.8),
        fontsize=8,
    )
    ax2.annotate(
        f"C5: {loss_gap.iloc[best_idx]:.3f}",
        xy=(best_idx + 1, loss_gap.iloc[best_idx]),
        xytext=(best_idx - 5.5, loss_gap.iloc[best_idx] - 0.22),
        arrowprops=dict(arrowstyle="->", color=COLOR_TURQUOISE, lw=0.8),
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.annotate(
        f"C12: {loss_gap.iloc[-1]:.3f}",
        xy=(len(df_res), loss_gap.iloc[-1]),
        xytext=(len(df_res) - 7.5, loss_gap.iloc[-1] - 0.14),
        arrowprops=dict(arrowstyle="->", color=COLOR_BLUE, lw=0.8),
        fontsize=8.5,
    )
    ax2.set_xticks(x_res[::3])
    ax2.set_xticklabels([labels_res[i] for i in range(0, len(labels_res), 3)])
    ax2.set_xlabel("Progressive Epoch")
    ax2.set_ylabel("Validation − Training Loss")
    ax2.set_title("Cross-Entropy Divergence (Overfitting Progression)")

    fig.suptitle("Kexia V4B — Generalization Gap and Overfitting Analysis Across Stages", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    p8 = out_path / "generalization_gap.png"
    fig.savefig(p8, dpi=200)
    plt.close(fig)
    created.append(p8)

    # 9. Hybrid Architecture Diagram
    p9 = generate_architecture_diagram(out_path / "hybrid_architecture_diagram.png")
    created.append(p9)

    return created


def generate_architecture_diagram(output_file: Union[str, Path]) -> Path:
    """Render a publication-grade schematic diagram of the HybridFER V4B architecture."""
    apply_scientific_style()
    out_file = Path(output_file)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=220)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str,
        edge_color: str,
        fill_color: str,
    ) -> None:
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            linewidth=1.6,
            edgecolor=edge_color,
            facecolor=fill_color,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color=COLOR_GREY_DARK,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color="#4C566A",
        )

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, color: str = COLOR_GREY_DARK) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color=color, mutation_scale=12),
        )

    ax.text(
        6.0,
        5.62,
        "Kexia Facial Expression V4B — Hybrid MobileNetV2 + ViT-B/16 with Gated Feature Fusion",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=COLOR_BLUE,
    )
    ax.text(
        6.0,
        5.25,
        "Total Parameters: 89,735,175   |   Fusion Rule:  g = $\\sigma$(W_g [c ; v] + b_g),   z = g $\\odot$ c + (1 - g) $\\odot$ v",
        ha="center",
        va="center",
        fontsize=9,
        color=COLOR_GREY_DARK,
    )

    # Input node
    draw_box(
        0.35,
        2.25,
        1.75,
        1.25,
        "Cropped Face RGB",
        "EXIF Transpose\nImageNet Norm",
        COLOR_GREY_DARK,
        "#F4F6F9",
    )

    # CNN Branch (top)
    draw_box(
        2.55,
        3.55,
        2.15,
        1.15,
        "MobileNetV2",
        "160 × 160 × 3 input\n1,280-D pooled features",
        COLOR_BLUE,
        "#EBF2FA",
    )
    draw_box(
        5.1,
        3.55,
        2.0,
        1.15,
        "CNN Projection (pc)",
        "LayerNorm(1280) → Linear(512)\nGELU → Dropout(0.25) → c",
        COLOR_BLUE,
        "#EBF2FA",
    )

    # ViT Branch (bottom)
    draw_box(
        2.55,
        1.05,
        2.15,
        1.15,
        "ViT-B/16",
        "224 × 224 × 3 input (16×16)\n768-D CLS representation",
        COLOR_TURQUOISE,
        "#E8F7F5",
    )
    draw_box(
        5.1,
        1.05,
        2.0,
        1.15,
        "ViT Projection (pv)",
        "LayerNorm(768) → Linear(512)\nGELU → Dropout(0.25) → v",
        COLOR_TURQUOISE,
        "#E8F7F5",
    )

    # Gated Feature Fusion (center right)
    draw_box(
        7.5,
        2.1,
        2.05,
        1.55,
        "Gated Feature Fusion",
        "Concat [c ; v] (1,024-D)\ng = $\\sigma$(Linear(1024 -> 512))\nz = g $\\odot$ c + (1 - g) $\\odot$ v (512-D)",
        COLOR_BLUE,
        "#F0F7F9",
    )

    # Classification Head
    draw_box(
        9.9,
        2.1,
        1.8,
        1.55,
        "Classification Head",
        "LN(512) → Linear(256)\nGELU → Drop(0.4) → Linear(7)\n7 Expression Logits",
        COLOR_GREY_DARK,
        "#F4F6F9",
    )

    # Arrows
    draw_arrow(2.1, 3.1, 2.55, 4.1, COLOR_BLUE)
    draw_arrow(2.1, 2.65, 2.55, 1.65, COLOR_TURQUOISE)
    draw_arrow(4.7, 4.12, 5.1, 4.12, COLOR_BLUE)
    draw_arrow(4.7, 1.62, 5.1, 1.62, COLOR_TURQUOISE)
    draw_arrow(7.1, 4.12, 7.5, 3.15, COLOR_BLUE)
    draw_arrow(7.1, 1.62, 7.5, 2.55, COLOR_TURQUOISE)
    draw_arrow(9.55, 2.88, 9.9, 2.88, COLOR_GREY_DARK)

    ax.text(
        6.0,
        0.35,
        "Class Order (0–6): angry, disgust, fear, happy, neutral, sad, surprise   |   Val Mean CNN Gate (C5): 0.4043",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4C566A",
    )

    fig.tight_layout()
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return out_file
