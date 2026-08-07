# Tinea → Psoriasis Misclassification — Diagnosis Report

**Date:** 2026-08-05
**Scope:** Known-misclassified Tinea images in `analytics/misclassified/true_Tinea_pred_Psoriasis/`
**Method:** `diagnose_tinea_sample.py` (AI verification force-disabled — `GEMINI_API_KEY=""`)
**Models:** MobileNetV2, EfficientNetB0, DenseNet121 (4-class: Acne, Psoriasis, Tinea, Vitiligo)
**Preprocessing:** 160×160 RGB, 0–255 float32, model-internal Rescaling layer.

> **No code was modified.** This is a pure diagnostic report.

---

## 1. Small-Sample Diagnosis (5 images)

### Image-by-image

| Image (true=Tinea) | MobileNetV2 | EfficientNetB0 | DenseNet121 | Majority | Soft Vote | FINAL `predict_disease` | Origin |
|---|---|---|---|---|---|---|---|
| `13tineaCApitis98-GP3.jpeg` | Tinea 65.5% | Vitiligo 73.4% | Tinea 98.9% | **Tinea** | Tinea 57.9% | **Tinea** (82.2%) | ✅ correct |
| `13TineaIncognito2.jpeg` | Tinea 48.7% | Acne 34.3% | Tinea 74.7% | **Tinea** | Tinea 48.8% | **Tinea** (61.7%) | ✅ correct |
| `tinea-beard-18.jpeg` | **Psoriasis 52.2%** | Vitiligo 40.9% | Tinea 88.8% | Psoriasis (tie→1st) | Tinea 38.4% | **Tinea** (38.4%) | ✅ correct (soft vote saved it) |
| `tinea-body-104.jpeg` | **Psoriasis 74.6%** | Acne 36.2% | **Psoriasis 67.5%** | **Psoriasis** | Psoriasis 53.3% | ❌ **Psoriasis** (71.1%) | **Individual models + Majority voting** |
| `tinea-body-107.jpeg` | **Psoriasis 53.1%** | Acne 36.1% | Tinea 73.9% | Psoriasis (tie) | Tinea 46.1% | **Tinea** (46.1%) | ✅ correct (soft vote saved it) |

### Small-sample aggregate
- **Per-model:** MobileNetV2 → Tinea 2 / Psoriasis 3; EfficientNetB0 → Vitiligo 2 / Acne 3; DenseNet121 → Tinea 4 / Psoriasis 1
- **Majority vote:** Tinea 2 / Psoriasis 3
- **Soft vote:** Tinea 4 / Psoriasis 1
- **Final `predict_disease`:** Tinea 4 / Psoriasis 1

### Root cause (small sample)
Of the 5 `true_Tinea` images, **only 1 (`tinea-body-104.jpeg`) was misclassified as Psoriasis**.
Origin: **1 = Individual models + 2 = Majority voting** — MobileNetV2 (74.6%) and DenseNet121 (67.5%)
*both* independently predicted Psoriasis, so majority voting selected it and soft voting agreed.
The core failure is **individual-model bias toward Psoriasis**, amplified by majority voting.

---

## 2. Full-Set Diagnosis (all 113 images in the folder)

### Aggregate confusion counts

| Stage | Tinea (correct) | Psoriasis | Acne | Vitiligo | Total |
|---|---|---|---|---|---|
| **MobileNetV2** | 44 | **67** | 1 | 1 | 113 |
| **EfficientNetB0** | 0 | 0 | **87** | 26 | 113 |
| **DenseNet121** | 63 | **40** | 10 | 0 | 113 |
| **Majority vote** | 38 | **64** | 10 | 1 | 113 |
| **Soft vote** | 60 | **47** | 6 | 0 | 113 |
| **FINAL `predict_disease`** | **55** | **46** | 11 | 1 | 113 |

> Note: EfficientNetB0 predicts **Tinea for 0/113** images — it essentially never outputs Tinea.
> That is a severe per-class sensitivity failure.

### Summary statistics (n = 113, all ground-truth Tinea)
- **% predicted as Psoriasis (final):** `46/113 = 40.7%`
- **% predicted correctly as Tinea (final):** `55/113 = 48.7%`
- **Remaining errors:** Acne 11 (9.7%), Vitiligo 1 (0.9%)

### Breakdown of the 46 Psoriasis final errors
Root cause attribution across the 46 misclassified-as-Psoriasis images:
- **Individual model bias** is the primary driver:
  - MobileNetV2 predicted Psoriasis on **67** Tinea images (59.3% of all images).
  - DenseNet121 predicted Psoriasis on **40** Tinea images (35.4%).
- **Majority voting** converted many of those into finals (final Psoriasis = 46), because it is a
  hard class-count vote with no tie-breaking weight toward Tinea.
- **Soft voting** (probability averaging) is notably *better* — it recovers Tinea on 60 images
  (Tinea 60 vs. Psoriasis 47), because EfficientNetB0's non-Tinea mass is spread across
  Acne/Vitiligo rather than Psoriasis, so averaging dilutes the Psoriasis signal.
  - Final (55 Tinea) sits between majority (38) and soft (60) because the production logic
    prefers **majority voting when a majority exists** and only falls back to soft voting on ties.

### Where the misclassification originates
Across the production pipeline for Tinea→Psoriasis:
1. **Individual model** — YES, primary. MobileNetV2 and DenseNet121 carry a strong Psoriasis bias
   on Tinea images (scaly, hyperpigmented lesions overlap appearance).
2. **Majority voting** — YES, secondary amplifier. Hard class vote has no confidence/class weighting,
   so a Psoriasis majority (even low-confidence) is accepted.
3. **Soft voting** — MINOR. It actually *reduces* Psoriasis errors vs. majority; when it goes wrong
   it is because the raw models already favor Psoriasis.
4. **AI verification** — NOT the cause. It was disabled for this run, and would only *override* the
   ML result if it were ≥10 pts more confident. No evidence it produces the Psoriasis errors.
5. **Other backend issue** — NO. Preprocessing, class-index mapping, and rebuilding of the vote logic
   are consistent with production (verified against `predict_disease`).

---

## 3. Conclusion

- The Tinea→Psoriasis problem is **reproduced consistently** across the full set (~41% of true Tinea
  images are returned as Psoriasis).
- **Root cause = individual model bias (primarily MobileNetV2 and DenseNet121) toward Psoriasis,
  amplified by hard majority voting.**
- EfficientNetB0 is effectively broken for Tinea (0/113), acting as a "spoiler" that can push tied
  votes toward the majority/wrong class.
- Soft voting recovers ~16 more correct Tinea predictions than majority voting (60 vs 38).

## 4. Recommended next steps (for approval — NOT yet applied)

1. **Prefer soft voting** over hard majority voting as the default ensemble decision
   (it has significantly higher Tinea recall: 60 vs 38).
2. **Add a confidence-weighted vote** (weight each model vote by its softmax confidence) instead of
   the current equal-weight hard vote.
3. **Investigate EfficientNetB0** — it never predicts Tinea; consider retraining or dropping it from
   the ensemble (or treating it as non-voting for Tinea).
4. **Calibrate / rebalance training data** for the Tinea vs Psoriasis boundary (both appear as scaly,
   hyperpigmented patches) — the individual models are the ultimate source of the bias.
5. **Only then** consider enabling AI verification as a safety net, with the current +10 confidence
   override rule.

These are recommendations only. No code or model changes have been made.
