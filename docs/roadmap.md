# Research Roadmap — Kexia Facial Expression V4B

This roadmap outlines prioritized, falsifiable engineering and scientific steps to address the empirical limitations documented in run `20261124_102050_038611`. All future experiments must preserve fixed evaluation splits and refrain from tuning hyperparameters on the 24,090-image independent test partition.

---

## Priority 1 — Complete Independent Test Evaluation on Checkpoint C5

- **Context:** In the final notebook, evaluation with `RUN_TEST=True` was interrupted at 4% of the validation loader (`87/2128` batches), leaving the 24,090-image test split unevaluated.
- **Action:** Run [`src/evaluation.py`](../src/evaluation.py) using the frozen selected checkpoint `best.pt` (Stage C, Epoch 5) on the held-out `test` partition without any further training or threshold tuning.
- **Deliverables:** `test_metrics.json`, `test_classification_report.csv`, and `test_confusion_matrix.csv` (excluding raw file paths).

---

## Priority 2 — Perceptual Duplicate Resolution and Identity Clustering

- **Context:** While exact `pixel_hash` duplicates (`7,172` files) were removed, **4,865** cross-partition 64-bit `dhash` candidates remain (`1,750` train, `1,069` validation, `2,046` test).
- **Action:**
  1. Compute Hamming distances over 64-bit `dhash` and 256-bit `pHash` (DCT-based perceptual hash) across partitions to cluster near-duplicate images.
  2. Use a pretrained face recognition embedding (e.g., ArcFace / InsightFace) to cluster probable subject identities and quantify cross-split identity overlap.
  3. Construct a strictly subject-disjoint and near-duplicate-free manifest (`manifest_v4c_clean.csv`) prioritizing `test > validation > train`.
- **Success Criterion:** Zero cross-split pairs within perceptual Hamming distance $\le 4$ and documented identity-cluster separation.

---

## Priority 3 — Controlled Architectural Ablation Study

- **Context:** `HybridFER` already supports `mode in {'hybrid', 'cnn', 'vit'}` and `fusion in {'gated', 'concat'}`, with a mean validation CNN gate of $\bar{g} = 0.4043$ at C5, but comparative runs on the identical manifest have not been executed.
- **Action:** Train four controlled configurations under the exact same seed (`42`), manifest, augmentation pipeline, and three-stage schedule:
  1. `mode='cnn'` (MobileNetV2 $160 \times 160$ + 512-D projection + head)
  2. `mode='vit'` (ViT-B/16 $224 \times 224$ + 512-D projection + head)
  3. `mode='hybrid', fusion='concat'` (1,024-D concatenation + head)
  4. `mode='hybrid', fusion='gated'` (V4B baseline)
- **Evaluation Criteria:** Validation Macro F1, Balanced Accuracy, parameter count, peak VRAM, and single-image inference latency (ms).

---

## Priority 4 — Minority-Class Learning and Overfitting Mitigation

- **Context:** At C5, `disgust` F1 is `13.44%`, `sad` F1 is `31.60%`, and the train–validation accuracy gap reaches `19.19 pp` (`25.99 pp` at C12).
- **Action:**
  1. **Loss Formulations:** Compare square-root weighted Cross-Entropy against Class-Balanced Focal Loss ($\gamma \in \{1.5, 2.0\}$) and Logit-Adjusted Cross-Entropy (Menon et al., 2021) to reduce `angry` over-prediction and `neutral` absorption without destabilizing calibration.
  2. **Regularization in Stages B & C:** Introduce `DropPath` (stochastic depth) in unfrozen ViT blocks, increase weight decay from `0.01` to `0.05` during Stage C, apply label smoothing ($\varepsilon = 0.05$), and add `RandomErasing` / `Mixup` ($\alpha = 0.2$).
  3. **Deterministic Checkpoint Continuity:** Ensure every stage transition and interruption loads full `optimizer`, `scheduler`, and `scaler` state dictionaries from `last.pt`.
- **Success Criterion:** Reduction of the Stage C train–validation accuracy gap below `10 pp` and improvement of `disgust` and `sad` F1-scores without degrading Macro F1.

---

## Priority 5 — Face Alignment, Probability Calibration, and Distribution Readiness

- **Action:**
  1. **Standardized Face Detection & Similarity Alignment:** Document and integrate an optional upstream 5-point facial landmark alignment step (e.g., RetinaFace / MTCNN similarity transform) prior to dual $160 \times 160$ and $224 \times 224$ resizing.
  2. **Post-Hoc Calibration:** Fit temperature scaling ($T > 0$) on the validation logits and report Expected Calibration Error (ECE) and multiclass Brier score before and after calibration.
  3. **Provenance & Licensing Audit:** Trace original dataset sources before publishing any model weights via Git LFS, GitHub Releases, or Hugging Face Hub.
