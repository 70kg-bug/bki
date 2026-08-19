# Pulsemind — MIMIC-IV pipeline rebuild: findings

**Date:** 2026-08-04 · **Source data:** MIMIC-IV 3.1 (97.2 GB, local) · **Code:** `pipeline/`
**Raw results:** `reports/s12_final_evaluation.json`, `reports/s11_bakeoff.json`,
`reports/s13_calibration.json`, `reports/verify.json`

---

> ## ⚠️ SUPERSEDED IN PART — the target changed on 2026-08-06
>
> **This document describes the model trained on `warning`.** That is no longer the target.
> The pipeline now trains on **`y_resp_6h`** — the respiratory arm of composite
> deterioration at a 6 h look-ahead. Every performance figure below (AP 0.6395, ROC-AUC
> 0.9512, the 0.1959 operating point, the 84%/95% documentation shares) is about the OLD
> label and is retained as the evidence base for the switch, not as a description of what
> ships.
>
> **What is still true:** the scale story (286 → 39,319 admissions), the imputation parity
> result, the feature-representation ablation, the faithfulness verification, the
> calibration mechanics (`scale_pos_weight` breaks the level; Platt fixes it at zero
> ranking cost), and the CatBoost hardware exclusion. All of these reproduced on the new
> label.
>
> **What is superseded:** every headline metric, the operating point and its thresholds, the
> documentation-share figures, and the sections arguing that `warning` should be replaced —
> that argument was accepted and acted on.
>
> **Current numbers live in:**
> `reports/tool_gate_pivot.json` (the eight-step ablation that drove the switch),
> `reports/s15_target_comparison.json` (head-to-head on one common row set),
> `reports/s13_calibration.json` and `models/operating_point_y_resp_6h.json` (rewritten for the
> new label), and `.claude/rules/results.md` (the interpretation).
>
> Headline replacement figures, for orientation — physiology-attributable skill,
> patient-level bootstrap, identical rows and features:
>
> | Target | ΔAP | ΔROC-AUC | doc-share (AP) |
> |---|---|---|---|
> | `warning` | +0.1187 | +0.0301 | 79.2% |
> | **`y_resp_6h`** | **+0.2439** | **+0.1504** | **18.5%** |
>
> This file needs a full rewrite against the new label; it has no generator, so that is a
> manual job and has not been done.

---

## Summary

The training set went from **286 ICU admissions / 3,288 rows** to **39,319 admissions /
4,200,041 rows** across **33,045 distinct patients**, and model performance on held-out
patients improved **7.5×** (average precision 0.0856 → 0.6395).

Three findings matter more than the headline number:

1. **The feature representation was the binding constraint — not the data volume, and not
   PCA.** Holding rows and algorithm fixed at exactly what the current model uses,
   swapping the 11 raw values for the 55 derived columns moves AP from 0.1290 to **0.5015**.
   That single change is worth ~10× removing PCA and ~20× the 109× data increase (§3).
2. **The prediction target is mostly a documentation-behaviour signal.** A model given
   only *which measurements were charted* — no physiological values at all — reaches 84%
   of full performance. Physiology adds a real, significant, but minority +0.1012 (§4).
   Findings 1 and 2 are the same fact seen twice: what unlocked the performance was
   exposing the charting pattern to the model.
3. **`warning` is the wrong target, and the replacement is now measured, not argued.**
   Trained head-to-head on identical patients, rows, features and params, composite
   deterioration (D) delivers **2.4× more physiology-attributable ranking skill and 5.8×
   more physiology-attributable discrimination**, with non-overlapping bootstrap
   intervals. Its documentation share falls from 84.5% to 27.2%. Raw ROC-AUC does drop
   (0.947 → 0.749) — because the old number was mostly a model of the nursing chart, not
   of the patient. Cohen's κ between the two labels is **0.076**: they are not the same
   event (§9).
4. **The scores rank well but are not probabilities.** Every model here scores *worse than
   a constant forecast* on Brier, because `scale_pos_weight` trains against a reweighted
   prior that is never undone — the shipped model predicts a mean risk of 0.166 against an
   observed 0.053. This matters because the product thresholds the raw number at 0.70.
   Rescaling fixes it completely and costs **exactly zero** average precision (§5).

None of these is a reason to stop. They are reasons to be precise about what the number
means, to fix the scale before anything reads it as a probability, and to change the target
before building on it.

---

## 1. Scale

| | Admissions | Rows | Patients |
|---|---|---|---|
| Current model (`data/training_calinerating.ipynb`) | 286 | 3,288 | — |
| Truncated BigQuery export (`data/raw-query.csv`) | 7,393 | 1,048,575 | — |
| **This pipeline** | **39,319** | **4,200,041** | **33,045** |
| Improvement | **137×** | **1,277×** | — |

### Where the data had been lost

| Stage | What happened | Admissions left |
|---|---|---|
| Full database on disk | — | 94,458 ICU stays |
| BigQuery export | Hit the 1,048,575-row export ceiling (2²⁰−1) and stopped silently | 7,393 |
| Cleaning notebook | `load_data` hardcodes `[:40000]` | ~286 |
| Training notebook | Deletes most negatives to balance classes (`iloc[:n_pos*2]`) | 286 |

The model was seeing **0.9%** of available admissions.

### Cohort construction

Invasive-ventilation procedure records (`procedureevents` itemid 225792) give **31,969**
admissions. Procedure documentation is incomplete, so a second detection pass added
admissions that carry invasive-specific evidence — a charted ventilator mode, or an
endotracheal/tracheostomy tube:

| Detection | Admissions | New vs. strict |
|---|---|---|
| Ventilation procedure record (strict) | 31,969 | — |
| Ventilator mode charted | 27,164 | +3,711 |
| Endotracheal / tracheostomy tube | 37,063 | +5,801 |
| **Adopted: strict ∪ invasive evidence** | **39,394** | **+7,425 (+23.2%)** |

Every added admission is still an invasively ventilated patient — only the detection is
more sensitive. PEEP alone was measured (39,364 admissions) but **not** used to admit a
stay, because PEEP also appears with non-invasive BiPAP.

---

## 2. Results on held-out patients

**6,609 patients (~839,000 rows), no patient shared with training.** Positive rate 5.26%,
so a random model scores ≈ 0.053. Average precision is the primary metric — at this
prevalence ROC-AUC is flattering.

| Model | AP | ROC-AUC | Brier | Train | Inference |
|---|---|---|---|---|---|
| **XGBoost** (CUDA) | **0.6395** | 0.9512 | 0.0804 | 97 s | **3.76 µs/row** |
| **LightGBM** (CPU) | 0.6393 | 0.9506 | **0.0770** | 401 s | 32.65 µs/row |
| Documentation-only diagnostic | 0.5384 | 0.9278 | 0.1089 | — | — |
| Current method @ 359,148 rows | 0.0920 | 0.6625 | 0.2345 | 275 s | — |
| Current method @ 3,288 rows (today) | 0.0856 | 0.6461 | 0.2399 | 5 s | — |

### Is the difference real? Bootstrap over patients

Resampling **patients**, not rows — rows within a patient are correlated, so a row-level
bootstrap would give misleadingly tight intervals. 200 resamples.

| Comparison | AP difference | 95% CI | Verdict |
|---|---|---|---|
| XGBoost − LightGBM | +0.0002 | [−0.0010, +0.0014] | **tied** |
| New − current method | +0.5541 | [+0.5478, +0.5609] | significant |
| New − current method, scaled up | +0.5478 | [+0.5415, +0.5548] | significant |
| New − documentation-only | +0.1012 | [+0.0970, +0.1057] | significant |

### Generalisation

Cross-validation AP **0.6400** vs held-out test AP **0.6393** — a gap of 0.0007. With
33,045 patients and patient-grouped splits there is no meaningful overfitting left. This
directly answers the concern that motivated the scale-up.

---

## 3. Finding: the feature representation was the constraint

The bottleneck was isolated by changing **one thing at a time**, every variant trained on
the same patients and scored on the same held-out patients.
Reproduce: `python -m pipeline.tools.ablate_method` → `reports/tool_method_ablation.json`.

| Variant | Rows | AP | Δ vs current |
|---|---|---|---|
| **Current method, exactly as it is** | 3,288 | **0.0856** | — |
| scale *before* PCA ("correct" order) | 3,288 | 0.0767 | **−0.0089** |
| drop the 3-stage chain, use the classifier directly | 3,288 | 0.0833 | −0.0024 |
| PCA 5 components | 3,288 | 0.0848 | −0.0008 |
| PCA 8 components | 3,288 | 0.0974 | +0.0117 |
| PCA 11 — rotation only, **no** reduction | 3,288 | 0.0998 | +0.0142 |
| **no PCA at all** | 3,288 | 0.1239 | **+0.0383** |
| no PCA + no chain | 3,288 | 0.1290 | +0.0433 |
| no PCA + no chain, 50k rows | 50,000 | 0.1437 | +0.0581 |
| no PCA + no chain, 359k rows | 359,148 | 0.1467 | +0.0611 |
| **no PCA + no chain + the 55 derived columns** | **3,288** | **0.5015** | **+0.4159** |
| same, 50k rows | 50,000 | 0.5609 | +0.4752 |
| Full pipeline (102 features, XGBoost) | 3,359,846 | **0.6395** | +0.5539 |

### Ranking the levers

| Lever | Worth |
|---|---|
| **Feature representation** (11 raw values → 55 derived columns) | **+0.372** |
| Model class + Tables 1/3 (GBC → XGBoost, 55 → 102 cols) | +0.079 |
| Removing PCA | +0.038 |
| Data volume (3,288 → 359,148 rows) | +0.018 |
| Dropping the 3-stage calibration chain | +0.005 |

**The decisive result:** at the *exact same 3,288 rows the current model uses*, and with
the *same* GradientBoostingClassifier, swapping the 11 raw values for the 55 derived
columns moves AP from **0.1290 to 0.5015**. No extra data, no different algorithm — the
representation alone is worth roughly ten times what removing PCA is worth, and twenty
times what the 109× data increase is worth.

### Two corrections to earlier claims

**PCA was not the main culprit.** It is a real cost (+0.038 to remove) but an order of
magnitude smaller than the feature representation. An earlier note in this project
attributed the gap primarily to `PCA(n_components=2)`; the ablation does not support that.

**Scaling after PCA is not a defect.** The "correct" order — scaling first — actually
scores *worse* here (0.0767 vs 0.0856). Unscaled, PCA is dominated by the high-variance
columns (tidal volume ~450 mL, minute volume ~8 L/min); scaling equalises every column's
contribution and, for this target, spreads the signal across components less usefully.
Unconventional, but not harmful.

### Why PCA hurts at all

Note that **PCA with 11 components — a pure rotation that discards no information —
still scores 0.0998 against 0.1239 with no PCA.** The loss is not dimensionality
reduction; it is the rotation itself. Decision trees split on one feature at a time, so
axis-aligned structure is cheap for them to express. Rotating the space smears that
structure across every component and forces many more splits to recover it. PCA is
generally counterproductive in front of a tree model, reduction or not.

### The sting

That +0.372 from "derived columns" is largely the model gaining access to `_observed`
and `_delta_t_min` — *was this charted* and *how stale is it*. §4 shows those carry 69.7%
of attribution. So "the feature representation was the bottleneck" and "the target is
mostly documentation behaviour" are the same finding seen from two angles: what unlocked
the performance was exposing the charting pattern to the model.

### Still-real defects fixed in the new pipeline

- PCA and the scaler were fit on the **entire** dataset before splitting, leaking test
  information into training.
- The split was a plain row-level `train_test_split` with no patient grouping, so the same
  patient appeared on both sides. (`bki/train/train_bki_models_advanced.py` had already
  established patient-level splitting — that safeguard was lost.)
- Row-deletion class balancing discarded most negatives and, with them, most patients.

The scale-up still matters — 33,045 patients is what makes patient-grouped validation
meaningful and drives the train/test gap to 0.0007 — but it was not the bottleneck.

---

## 4. Finding: the target is mostly documentation behaviour

`warning` is a native column of MIMIC's `chartevents`, defined as *"a warning for this
observation was manually documented by the care provider."* It is a caregiver
documentation flag, not a physiological outcome.

Three independent lines of evidence agree on how much of the model's performance comes
from charting patterns rather than physiology.

**a. Ablation** (LightGBM, fixed parameters, validation fold):

| Feature set | AP |
|---|---|
| `documentation_only` — 33 columns: *was it charted*, *how stale*, *never measured* | **0.5455** |
| `final_only` — the 11 physiological values the current model uses | **0.1637** |
| `t2_all` — 55 time-series columns | 0.6176 |
| `full` — 102 columns, all three tables | 0.6364 |

Knowing only which measurements were charted outscores knowing what they said by **3.3×**.

**b. Held-out ablation:** documentation-only reaches **0.5384 of the full 0.6395 — 84%**.
Physiology's contribution is the significant but minority **+0.1012 (16%)**.

**c. Feature attribution (SHAP):** **69.7%** of total attribution sits on `_observed` /
`_delta_t_min` / `_structurally_missing` columns. Six of the top eight features are
"was it charted" or "how stale is it":

| Rank | Feature | Kind |
|---|---|---|
| 1 | `fio2_observed` | was it charted |
| 2 | `peep_observed` | was it charted |
| 3 | `fio2_delta_t_min` | staleness |
| 4 | `minute_volume_delta_t_min` | staleness |
| 5 | `spo2_locf` | **physiological value** |
| 6 | `first_careunit` | context |
| 7 | `pip_delta_t_min` | staleness |
| 8 | `ventilator_mode` | context |

By source table: Table 2 (time-series) **82.3%**, Table 1 (patient background) 10.0%,
Table 3 (interventions) 7.7%.

**How to read this.** The model is *not* purely an artifact — physiology adds genuine,
statistically significant signal. But predicting `warning` is predominantly a
documentation-behaviour task. A defensible presentation says so, and reports the
documentation-only diagnostic beside the headline. The pipeline computes it automatically
so it always travels with the result.

### The conclusion does not depend on average precision

`warning` was called a weak label using AP. AP is the metric most *generous* to physiology
here, so the conclusion survives changing it. Measured on the clean split (§5), as a share
of each metric's own achievable skill — normalised against its floor, so the numbers are
comparable:

| Metric | Floor | Share reachable from charting pattern alone |
|---|---|---|
| Average precision | prevalence 0.0526 | **84.6%** |
| ROC-AUC | 0.5 | **95.2%** |
| Brier skill (after calibration) | base-rate constant | **84.3%** |

Under ROC-AUC the label looks *worse*, not better. Switching metrics does not rescue it.

---

## 5. Finding: the scores rank well and are not probabilities

`pipeline/stages/s13_calibrate.py` → `reports/s13_calibration.json`. Same held-out 6,609 patients.

Everything in §2 measures **rank** — AP and ROC-AUC ask whether sicker patients sort above
healthier ones, and both are invariant under *any* monotone rescaling of the output. The
product does not consume a rank. `bki/backend/main.py:85` applies `risk_prob > 0.70`, which
is a test on the **level**. Nothing had measured the level.

The level was broken. A constant forecast of the base rate scores Brier
`p̄(1−p̄) = 0.0499`; Brier skill `BSS = 1 − BS/0.0499` is negative when a model is worse
than that constant.

| Head | AP | Brier | BSS | ECE | mean predicted |
|---|---|---|---|---|---|
| LightGBM + Platt | **0.6347** | 0.0290 | **+0.418** | 0.0028 | 0.0534 |
| XGBoost, no `scale_pos_weight` | 0.6338 | 0.0290 | **+0.418** | **0.0011** | 0.0518 |
| XGBoost **regressor** on 0/1 | 0.6322 | 0.0292 | +0.414 | 0.0079 | 0.0603 |
| XGBoost + `scale_pos_weight`, raw | 0.6277 | 0.0716 | **−0.437** | 0.1135 | **0.1662** |
| XGBoost + Platt | 0.6277 | 0.0293 | +0.412 | 0.0016 | 0.0532 |
| XGBoost + isotonic | 0.6221 | 0.0293 | +0.412 | 0.0012 | 0.0533 |
| `bki` 3-stage chain, modern base | 0.6243 | 0.0295 | +0.408 | 0.0051 | 0.0575 |
| `bki` 3-stage chain, literal sklearn † | 0.5574 | 0.0332 | +0.333 | 0.0188 | 0.0539 |
| LightGBM raw | 0.6347 | 0.0751 | **−0.507** | 0.1106 | 0.1632 |
| documentation-only | 0.5393 | 0.1079 | −1.164 | 0.1729 | 0.2255 |
| *base-rate constant* | *0.0526* | *0.0499* | *0.000* | *0* | *0.0526* |

† capped at 50,000 rows and numeric features only — sklearn's `GradientBoostingClassifier`
does not scale to 2.5M × 102. Its AP is not comparable; its calibration is.

**Every raw score is worse than a constant.** The shipped XGBoost predicts a mean risk of
**0.166 against an observed rate of 0.053** — 3.2× too high. The cause is
`scale_pos_weight ≈ 18` (`s11_train.py:96`, `:113`): it trains against a reweighted prior
and nothing converts the output back.

The reliability curve makes it concrete. Top decile of the raw score: model says **0.899**,
reality is **0.513**. The bin at predicted 0.448 has an observed rate of **0.072** — six
times overstated. After Platt those become 0.520 vs 0.513, and 0.0721 vs 0.0724.

### Rescaling costs exactly nothing

Murphy's decomposition (Brier = reliability − resolution + uncertainty) separates the two:

| | reliability (miscalibration ↓) | resolution (information ↑) |
|---|---|---|
| XGBoost raw | 0.03993 | 0.01648 |
| XGBoost + Platt | **0.00001** | **0.01648** |

Platt cut miscalibration by ~4,000× and left resolution **bit-for-bit identical**. The raw
model's real information (0.01648) was more than cancelled by its miscalibration (0.03993)
— it had the signal and threw it away on the scale.

Patient-level bootstrap, 95% CI, Platt vs raw:

| | difference | 95% CI |
|---|---|---|
| Brier | **−0.0423** | [−0.0437, −0.0412] |
| ECE | **−0.1118** | [−0.1135, −0.1102] |
| Average precision | **+0.00000** | [+0.00000, +0.00000] |

AP was identical in **every single resample**. Platt is strictly monotone, so this is a
theorem, not a coincidence — and it is asserted in the code, so a future change that breaks
it fails the stage.

### Is a regressor emitting [0,1] better than a ranked classifier?

They are not alternatives — AP is *computed from* the score. But the underlying question is
real, because squared-error regression on a 0/1 label **is** Brier minimisation, so it asks
whether optimising level directly beats optimising rank then rescaling. Measured:

| Comparison | ΔAP | ΔBrier | Verdict |
|---|---|---|---|
| Platt − regressor | −0.0045 [−0.0062, −0.0025] | +0.00007 [−0.00002, +0.00015] | regressor ranks slightly better, **tied on calibration** |
| Platt − `bki` chain | +0.0034 [+0.0019, +0.0048] | −0.00025 [−0.00031, −0.00017] | chain **worse on both** |
| Platt − isotonic | +0.0056 [+0.0053, +0.0058] | −0.0000 [−0.0000, +0.0000] | Platt ranks better, same calibration |

**The regressor head is a legitimate choice** — statistically tied on calibration and
marginally better at ranking. It is not, however, *better than the simple fix*: dropping
`scale_pos_weight` gives a higher AP (0.6338 vs 0.6322) and a lower ECE (0.0011 vs 0.0079)
with no extra machinery at all.

**The `bki` three-stage chain is the one option that is strictly worse.** Even rebuilt with
a modern base learner, orientation corrected and the calibrator fitted on training data, it
loses on ranking *and* on calibration, both CIs excluding zero. Its second-stage regressor
re-approximates an already-calibrated probability and can only add error.

### What `risk > 0.70` actually does

709,114 monitored patient-hours across 7,823 held-out admissions. Alerts are de-duplicated
to one per admission-hour, since rows sit on an irregular grid.

| Score | Rows flagged | Sensitivity | PPV | Alert-hours per patient-day |
|---|---|---|---|---|
| raw, threshold 0.70 | 8.2% | 0.701 | 0.450 | 2.2 |
| calibrated, threshold 0.70 | 2.0% | 0.315 | 0.832 | 0.6 |

**The same constant means different things on the two scores** — a 3.7× swing in alert
volume. On the raw score 0.70 is a moderate-sensitivity setting; on a calibrated score it is
conservative.

With calibration the threshold can be chosen against an alert budget instead of picked round:

| Budget | Threshold | Sensitivity | PPV |
|---|---|---|---|
| 1 alert-hour / patient-day | 0.4432 | 0.479 | 0.718 |
| **2 alert-hours / patient-day** | **0.1959** | **0.672** | **0.486** |
| 4 alert-hours / patient-day | 0.0873 | 0.847 | 0.304 |

Persisted to `models/operating_point.json` alongside `xgboost_warning.ubj` and
`calibrator.joblib`. **Nothing in `bki/` was modified** — it is a separate repository kept as
historical reference.

### A smaller defect, fixed in passing

`s12_baselines.py:140` passes the test set in as the early-stopping eval set, so the tree
count was chosen on test. Reproducing that protocol scores AP 0.6308 against 0.6277 for the
clean four-way split — but the leaky run also trains on 67% more patients, so **+0.0031
bounds the leak rather than isolating it**. It is small and does not affect any conclusion.
Stage 13 uses a dedicated early-stopping fold; the test set is scored once and nothing else.

---

## 6. Algorithm selection

Budget-matched: identical folds, identical features, identical metric, 600 s tuning
wall-clock per candidate. Matching wall-clock rather than trial count is the fair
comparison when one candidate has a working GPU backend and another does not.

| Algorithm | CV AP | Trials in budget | Tuning | 5-fold CV |
|---|---|---|---|---|
| **XGBoost** (CUDA) | **0.6407 ± 0.0044** | 10/12 | 10.1 min | **7.1 min** |
| **LightGBM** (CPU) | 0.6400 ± 0.0051 | 8/12 | 10.9 min | 24.3 min |
| CatBoost | *excluded* | 1/12 | 31.4 min/trial | — |

**LightGBM and XGBoost are statistically tied on accuracy** (bootstrap CI straddles zero).
The original choice of LightGBM was sound. Decide on other grounds:

- **Throughput → XGBoost.** 4× faster training, **8.7× faster inference** (3.76 vs
  32.65 µs/row ≈ 266,000 vs 30,600 rows/s). For a real-time bedside monitor this matters.
- **Calibration → LightGBM.** Brier 0.0770 vs 0.0804. If a clinician acts on a displayed
  probability, calibration is not a footnote. XGBoost can close this with Platt or
  isotonic calibration on a validation fold.

**Hyperparameter tuning is not the lever here** — tuned versus fixed parameters differ by
less than 0.002 AP. The ceiling is set by what the features can express.

### CatBoost: excluded on hardware grounds

- **GPU unusable.** `CUDA error 218: a PTX JIT compilation failed` (`ptxas fatal: memory
  allocation failure`). CatBoost 1.2.10 ships no `sm_120` (Blackwell) kernels and the
  runtime PTX JIT fails on the RTX 5060. Its GPU build also cannot compute PRAUC.
- **CPU prohibitive.** ~1,883 s per fit *already capped at 800 iterations*, versus ~82 s
  (LightGBM) and ~61 s (XGBoost) — 23–31× slower. Completing 5-fold CV would have cost
  ~2.6 h to refine a candidate whose single-trial AP (0.6248) already trailed both.

### Transformers: deliberately not run

FT-Transformer was scoped but not executed, on evidence rather than assumption: tuning
moves AP by <0.002, and documentation features carry ~70% of attribution. That places the
ceiling on *what the features express*, not on model class. A transformer given the same
isolated per-timestamp snapshots would optimise the same limited signal. The lever that
would move the number is temporal structure (§10), currently out of scope.

---

## 7. Data-quality defects found and fixed

Each was found by checking rather than assuming, and each would have silently degraded
results.

| # | Defect | Evidence | Resolution |
|---|---|---|---|
| 1 | **`peep` sourced from the wrong item.** | On all **14,288** rows where PEEP-set-only disagreed with the BigQuery export, the export matched `Total PEEP Level` (224700) exactly; **none** matched `PEEP set` (220339). | Frozen definition is `COALESCE(Total PEEP, PEEP set)`. Agreement **89.27% → 99.81%**. Clinically also the better measure — set PEEP plus auto-PEEP, i.e. the pressure actually in the lung. **Cost no rescan**: 224700 was already in the superset cache. |
| 2 | **FiO₂ unit heterogeneity.** | 154 values on the 0.21–1.0 fraction scale alongside 1,142,898 on the 21–100 percentage scale. | Fractions rescaled ×100. Left alone they would read as 0.5% oxygen instead of 50%. |
| 3 | **Corticosteroids absent from `inputevents`.** | `icu/inputevents` has 133 medication items, none steroids — IV steroids live in `hosp/prescriptions` / `emar`. | Group removed rather than shipped as an all-zero column. `s08` now **raises** if any drug group resolves to zero itemids. |
| 4 | **No plausibility range checking at all.** | e.g. 3,873 FiO₂ values (0.34%) outside 21–100. | Per-item ranges; out-of-range → NULL so bad values flow into the missingness machinery instead of corrupting medians. |
| 5 | **DuckDB CSV parse failure.** | MIMIC contains RFC4180 doubled quotes such as `"1"" Packing"` (an inch mark); the scan died at line 121,491. | `escape='"'` set explicitly. |
| 6 | **Target column silently dropped.** | The original `result[ordered]` reorder omits `warning`, which is why the training notebook re-reads the raw CSV to recover it. | Label preserved through imputation. |

### ⚠️ 7.1 OPEN — `vent_hours` is future information. Flagged for removal.

Found on 2026-08-17 while building the serving-time feature assembly, and **not fixed**:
training is finalised, so this is recorded for the next retrain rather than acted on.

`s01_cohort_strict.py:50` defines it as `date_diff('hour', vent_start, vent_end)` — the
length of the **completed** ventilation episode — and `s02_table1_static.py:169` broadcasts
it per admission. Every reading in a stay therefore carries how long that stay will
ultimately last, including the first, before any of it has elapsed.

`build/features.json` documents `full` as "109 features, all at or before *t*". This one is
not, and the existing leakage guards do not catch it: they check for `leaky_` prefixes,
identifiers and forward-label columns, none of which describe a whole-stay aggregate.

Measured (`python -m pipeline.tools.serving_parity` → `reports/tool_causal_parity.json`):

| | |
|---|---|
| Constant within the stay | **39,319 of 39,319 stays (100%)** |
| Total ≥ the span already observed | **94.7%** of stays with >5 readings |
| In the stored top-8 | 1,851 of 12,578 sampled readings (**14.72%**); ranked 1st in 92 |
| Share of \|attribution\| when present | mean 0.059, max 0.234 |

The top-8 row is a share of the 120-stay sample the harness draws, not of the 58,765-record
file — `records_sampled` is persisted in the JSON so the two bases cannot be confused.

**Why it matters beyond tidiness.** Ventilation lasts longer for patients who deteriorate,
so the feature is partly an outcome. Serving it causally — "hours since ventilation started",
the only form obtainable at time *t* — moves the **displayed band on 7.95% of readings**.
The eleven `*_structurally_missing_in_stay` features, redefined the same way, move 0.85%.
Nearly the whole train/serve gap is this one column.

AUROC 0.7860 and AP 0.4155 in §2 were measured with it in the feature set.

**Recommended fix at the next retrain:** replace with elapsed hours at *t*
(`charttime - vent_start`), which is causal and is almost certainly what was intended, then
re-measure. The serving path already uses that form; `core/features.py` marks it.

### ⚠️ 7.2 OPEN — grounding checks numbers, not the units attached to them

Found on 2026-08-18 while verifying the 7B end to end against a live serving stack. The
generator produced, and the checker **passed**, this sentence:

> "The patient's PEEP was carried forward from **0.45 hours** ago … **Minute volume** and
> tidal volume were also **carried forward** from 0.45 hours ago, measured at this reading."

Two false claims in one sentence, on a screen whose stated purpose is that every value
carries where it came from:

| Claim | Record |
|---|---|
| PEEP charted "0.45 hours" ago | `age_min = 0.45` — 27 **seconds**, not 27 minutes. Out by **60×** |
| Minute volume "carried forward" | `source = measured`, `age_min = 0`. PEEP and tidal volume were the carried-forward ones |

**Why the checker let it through.** `grounding.py` re-derives the allowed-number set from the
record and asks whether every number in the text is a member of it. `0.45` is a member, so
the claim passes — the check has no opinion about the **unit** written beside a number, nor
about a **provenance** word attached to a parameter. Its independence from `explain.py` is
what makes it valuable, and this is a gap in what it covers rather than a defect in that
design.

**Why the generator reached for the raw number.** `build_payload` gives each telemetry entry
BOTH a rendered phrase and the raw figure:

```python
"age_min": v.get("age_min"),          # 0.45
"age":     age_phrase(v.get("age_min"))   # "measured at this reading"
```

`age_phrase` is correct — under a minute it returns "measured at this reading" precisely so
no unit has to be chosen. The model ignored it, took the bare `0.45`, supplied its own unit,
and then appended the correct phrase to the end of the same sentence, producing a
self-contradiction that no numeric check can see.

**Two candidate fixes, and they are not equivalent.** Withholding `age_min` from the prompt
removes the bare number the model misused, but it changes every generation and so
invalidates any generation already verified; note also that the staleness filter later in
the same module reads `t["age_min"]` off the payload. Extending the checker to verify units
and provenance words leaves generation untouched and turns the bad sentence into a rejection
that falls back to the template floor — which is the behaviour the floor exists for. The
second is the smaller change and the safer one to make first.

**Not a blocker for the local demo.** The template floor produces correct prose for the same
record, and the service takes `use_llm: false` to select it. It is a blocker for anything a
clinician reads.

---

## 8. Verification

### Faithfulness to the frozen BigQuery definition

Local extract compared column-by-column against `data/raw-query.csv` on the 6,028
admissions and 672,618 `(stay_id, charttime)` keys both cover:

| Column | Agreement |
|---|---|
| spo2, fio2, flow_rate, pip, respiratory_rate_total, minute_volume, tidal_volume_observed, etco2, expiratory_ratio | **100.00%** |
| **warning (target)** | **100.00%** |
| peep | 99.81% |
| inspiratory_ratio | 99.67% |

Residual differences (251 peep rows, 116 I-ratio rows, <0.4%) come from tie-breaking when
a parameter is charted more than once at the same timestamp.

### Imputation parity

The vectorised rewrite against the original row-by-row implementation on an identical
40,000-row slice:

- **44 of 44 columns byte-identical** (`_observed`, `_locf`, `_structurally_missing_in_stay`, `_final`)
- **0.01 s vs 48.4 s — 7,235× faster.** Extrapolated to the full dataset: ~1 s versus ~1.4 h.
- `_delta_t_min` differs **by design**: the rewrite reports the age of the value actually
  in use (0 on a freshly measured row) rather than the gap since the previous reading.

### Leakage and integrity (assert-enforced, not hoped for)

- No patient appears in both train and test (26,436 / 6,609, overlap 0)
- No identifier, timestamp or outcome column in any feature set
- Every Table 3 intervention began **strictly before** its row's timestamp
- Joins preserved row and admission counts exactly
- No constant or all-null feature columns

---

## 9. Alternative targets for the risk metric — measured

`warning` was chosen because it was convenient and sounded plausible. §4 shows what it
actually is. These are the realistic alternatives, all measured on the built cohort
(39,319 admissions / 4,200,041 rows). Full numbers: `reports/tool_target_candidates.json`,
reproducible via `python -m pipeline.tools.explore_targets`.

Time-varying targets are evaluated on a **6-hour forward look-ahead** over 3,548,733
hourly bins — "will this happen in the next 6 h?"

| # | Candidate | Kind | Positive rate | Admissions with ≥1 | Extra ETL |
|---|---|---|---|---|---|
| — | `warning` (current) | per-timestamp | 5.33% of rows | 85.5% | none |
| **D** | **Composite deterioration** | per-timestamp, forward | **7.98% of bins** | **63.0%** | none |
| C | Desaturation (SpO₂ <88%) | per-timestamp, forward | 2.92% of bins | 20.8% | none |
| B | Ventilator support escalation | per-timestamp, forward | 2.10% of bins | 45.9% | none |
| A | NHSN VAC | per-admission, day-anchored | 46.6% of assessable ⚠️ | 12,213 assessable | none |
| E1 | In-hospital mortality | per-admission | 19.16% | — | none (built) |
| E2 | Prolonged ventilation (>7 d) | per-admission | 15.59% | — | none (built) |
| E3 | Extubation failure (<48 h) | per-admission | 4.82% | — | none (built) |
| E4 | ICU readmission | per-admission | 9.84% | — | none (built) |

### D versus `warning` — measured head-to-head

`pipeline/stages/s14_forward_targets.py` + `pipeline/stages/s15_target_compare.py` →
`reports/s15_target_comparison.json`. D was previously a recommendation from construct
validity. It has now been trained and measured.

**Definition.** D is positive if, within the next 6 hours, **any** of: FiO₂ rises ≥20
points above the setting in force, PEEP rises ≥3 cmH₂O, SpO₂ falls below 88%, or a new
vasopressor infusion starts.

**Strict-delta protocol.** Identical patients, folds, the 102-column feature set, XGBoost
params, and the clean 4-way split from §5 — and, critically, **identical rows**. D is
undefined where its forward window overruns the end of the stay, so every label is scored
on D's observable row set: **3,945,823 rows, 6.05% dropped as unobservable**. Held-out
test: 788,405 rows / 5,953 patients.

> ⚠️ **Average precision is not comparable down this table.** Random-guess AP equals the
> positive rate, so `warning` starts from a floor of 0.050 and D from 0.137. Compare
> **AP-skill**, **ROC-AUC** and **documentation share** — never raw AP.

| Target | Prevalence | AP *(floor)* | AP-skill | ROC-AUC | Brier skill | Doc-share (AP / AUC) | Admissions covered |
|---|---|---|---|---|---|---|---|
| `warning` | 5.04% | 0.6142 *(0.050)* | 0.5937 | **0.9467** | +0.401 | **84.5% / 95.0%** | 78.5% |
| **D composite** | 13.72% | 0.4222 *(0.137)* | 0.3303 | 0.7488 | +0.159 | **27.2% / 48.3%** | 76.5% |
| D strict † | 11.33% | 0.4214 *(0.113)* | 0.3474 | 0.7770 | +0.178 | 29.8% / 57.5% | 76.4% |

† escalation judged only against a baseline measured within 60 min — a robustness check on
whether "a rise" is an acute change or slow drift against a stale setting.

### The trade, stated plainly

**`warning` is far easier to predict** — ROC-AUC 0.9467 against 0.7488, AP-skill 0.5937
against 0.3303. Anyone presenting D has to own that number rather than hide it.

**But it was easy for the wrong reason.** 95.0% of `warning`'s ROC-AUC skill is reachable
from charting pattern alone. Strip the documentation signal out and ask how much
discrimination is attributable to the patient's physiology:

| Target | Physiology adds (AP) | 95% CI | Physiology adds (ROC-AUC) | 95% CI |
|---|---|---|---|---|
| `warning` | +0.0870 | [+0.0820, +0.0920] | +0.0225 | [+0.0207, +0.0247] |
| **D composite** | **+0.2070** | [+0.1931, +0.2195] | **+0.1292** | [+0.1182, +0.1424] |
| D strict | +0.2156 | [+0.2038, +0.2278] | +0.1177 | [+0.1097, +0.1267] |

**D delivers 2.4× more physiology-attributable ranking skill and 5.8× more
physiology-attributable discrimination than `warning`.** Patient-level bootstrap;
the confidence intervals do not come close to overlapping.

The apparent paradox resolves cleanly: `warning`'s 0.9467 is mostly a model of the nursing
chart. D's 0.7488 is mostly a model of the patient.

### They are not the same event

On identical rows, **Cohen's κ = 0.076**, Jaccard 0.078.

| | D positive | D negative |
|---|---|---|
| **`warning` positive** | 10,669 | 29,104 |
| **`warning` negative** | 97,527 | 651,105 |

`warning` and D disagree far more often than they agree. Switching targets is not a
relabelling of the same underlying event — it is a change of question.

### Why 6 hours

| Horizon | Prevalence | Admissions with ≥1 |
|---|---|---|
| 2 h | 6.96% | 68.3% |
| **6 h** | **13.76%** | **68.7%** |
| 12 h | 21.29% | 68.9% |

Coverage is flat from 2 h to 12 h (68.3% → 68.9%) while prevalence triples. Beyond 6 h the
label buys density, not reach, at the cost of a vaguer clinical claim. 6 h is the knee.

### Reconciling with the 7.98% published above

The row-grain build gives 13.76%, not 7.98%. The cause is measured, not tuned away:
`explore_targets.py` required a **same-hour measured** escalation baseline, and FiO₂ is
charted on only 22.4% of rows against 85.5% for the LOCF value. Its FiO₂ and PEEP
components therefore could not fire on roughly four bins in five. **The 7.98% was an
undercount driven by charting sparsity.** Using the setting actually in force is both
clinically faithful and ~4× more eligible. Full reconciliation in
`reports/s14_forward_targets.json`.

### Recommendation and honest caveats

**Switch to D composite.** It is the variant with the lowest documentation share on both
metrics, and reducing that confound is the entire reason for switching. D strict reaches
the same verdict from a different construction (2.5× / 5.2×), so the conclusion does not
depend on the baseline-freshness choice.

Three things to say out loud when presenting it:

1. **Raw discrimination genuinely falls** (0.947 → 0.749). The honest framing is that the
   old number was inflated by an artifact, not that the new model is worse.
2. **D is still 74% clinician-action driven.** Only 26.0% of D's positives involve the
   desaturation component, and just **19.4% are triggered by it alone**; the rest involve
   a FiO₂/PEEP change or a vasopressor start — things a clinician *does because* the
   patient is deteriorating. D predicts treatment decisions as well as patient states.
   A pure-physiology variant (desaturation only) is available at 3.58% prevalence.
3. **D still calibrates** (Brier skill +0.159, ECE 0.0037), so the [0,1] output remains
   usable as a probability under §5's banding.

### On the others

- **B (support escalation)** is the cleaner, narrower version of D — pure ventilator
  mechanics, and effectively the NHSN VAC trigger applied continuously rather than daily.
  Use it if you want the tightest clinical story; use D if you want more signal.
- **E1 (in-hospital mortality, 19.16%)** is the safest benchmark: a large literature, easy
  to defend, and directly comparable to the Gen-1 `in_hospital_death` work in
  `bki/README.md`. Weakness: it is a whole-stay verdict and only weakly ventilator-specific.
- **E3 (extubation failure, 4.82%)** is highly actionable but only meaningful around
  extubation — a decision-support tool, not a continuous risk metric.
- ⚠️ **A (NHSN VAC)** is the right *standard* but the number above is **not trustworthy
  yet**. My implementation flags 46.6% of assessable admissions against a published VAC
  rate of roughly 5–10%, so it is clearly over-triggering: the NHSN rule requires a
  baseline period of *stable or decreasing* support before the rise, which my
  `least(lag1, lag2)` approximation does not enforce. Treat it as evidence the definition
  is computable here, not as a prevalence estimate. Transcribe the protocol properly (or
  port MIT-LCP's implementation) before using it.

**A reasonable path:** train on **D** as the primary risk metric, keep **E1** as a
secondary head or benchmark, and implement **A** properly if a published, externally
recognisable definition is needed for clinical or regulatory framing.

---

## 10. Recommendations

**1. Switch the target.** See §9 — composite deterioration (D) is the recommended primary,
with in-hospital mortality (E1) as a benchmark. No additional ETL is required for either.

**2. Temporal features — the largest remaining lift.** Every row is currently an isolated
snapshot, so the model can see a *level* but never a *trajectory*. Rolling means, standard
deviations and slopes over 1 h / 4 h / 12 h windows, and physiological ratios (S/F =
spo2÷fio2, driving-pressure proxy = pip−peep, dynamic compliance, RSBI), are all
computable **from the frozen eleven columns alone** — no new parameter. Windows must be
backward-looking within each admission or they leak the future.

**3. Drop `scale_pos_weight`, and treat the displayed number as a probability.** See §5.
The reweighting bought nothing measurable and cost twice: it lowered AP (0.6338 → 0.6277)
*and* made the output worse than a constant forecast (BSS −0.437). If it is kept for some
other reason, Platt scaling on a held-out fold undoes the damage at **exactly zero** cost to
ranking. The 0.70 alert constant in `bki/backend/main.py:85` should then be replaced by a
threshold chosen against an alert budget — 0.1959 for 2 alert-hours per patient-day, already
written to `models/operating_point.json`. Do **not** adopt the legacy three-stage
classifier→calibrator→regressor chain: it is worse on ranking and on calibration, both
significant.

**4. Tier-2 enrichment**, if the feature freeze is ever relaxed: `hosp/labevents`
(WBC, lactate, procalcitonin — the last is already requested by the legacy gRPC contract),
and `hosp/prescriptions` for corticosteroids. Both need a second ETL path.

**5. Additional chartevents columns**, available at **no extra scanning cost** because
they are already in the cache: Plateau Pressure (224696 → true driving pressure), Total
PEEP (224700 → auto-PEEP), Mean Airway Pressure (224697), Compliance (229661), set Tidal
Volume (224684), spontaneous Vt/RR (224421/224422 → RSBI).

---

## 11. Cohort description

Sanity checks that the cohort behaves like a real ventilated ICU population.

**Demographics and comorbidity** (Charlson, mapped across **both** ICD-9 (45.7%) and
ICD-10 (54.3%) — an ICD-9-only mapping would have silently dropped half the records):

median age 66 · median Charlson index 5 · COPD 25.3% · heart failure 28.1% ·
renal disease 19.9% · malignancy 10.6% · complicated diabetes 10.1%

**Outcomes:** in-hospital mortality 19.16% · ICU mortality 14.88% · 28-day 20.83% ·
90-day 26.06% · prolonged ventilation (>7 d) 15.59% · extubation failure 4.82% ·
ICU readmission 9.84%

**Interventions** (share of rows with an infusion running): sedatives 42.41% ·
opioids 29.41% · vasopressors 27.18% · paralytics 2.74%

Ventilated duration: median 24 h, IQR 8–93 h.

---

## 12. Architecture and performance

### Build once, reuse forever

The only expensive operation is reading the 42 GB `chartevents.csv`. It happens **once**
and produces a compact cache that every later stage reads instead.

| | Size | Rebuild cost |
|---|---|---|
| `chartevents.csv` (source) | 41.94 GB | 33 s per full scan |
| `ts_long.parquet` (cache) | **166 MB** | never — read in seconds |

**253× reduction.** Three properties make this durable:

1. **The cache is filtered by itemid only, not by cohort** — changing the cohort
   definition costs seconds, not a rescan.
2. **It stores a superset** (~30 itemids, of which 11 are the frozen model inputs). Extra
   codes are free during a pass that already reads every byte; adding one later would cost
   a full rescan. *This paid for itself immediately* — the PEEP correction needed 224700,
   which was already cached.
3. **Each table is an independent Parquet artifact with a manifest** (source fingerprints,
   config hash, counts). Stages skip themselves when nothing moved.

### Stage timings

| Stage | Output | Time |
|---|---|---|
| Cohort (strict) | 31,969 admissions | <1 s |
| Table 1 — patient background + Charlson | 47 cols | 2 s |
| **Cache — the only 42 GB scan** | 44,338,438 rows, 166 MB | **33 s** |
| Cohort (final) | 39,394 admissions | <1 s |
| Table 2 — pivot | 4,200,041 rows | 4 s |
| Table 2 — imputation | 70 cols | **2 s** |
| Table 3 — interventions | 4,200,041 rows | 4 s |
| Table 4 — outcomes | 39,394 rows | 1 s |
| Assemble | 102 features | 1 s |

Total build artifacts: **400 MB**. DuckDB never spilled to disk.

### Hardware notes

- **RTX 5060 Laptop GPU**, Blackwell `sm_120`, 8 GB VRAM (~7.4 GB usable), driver 610.62 / CUDA 13.3
- **XGBoost CUDA: works.** **PyTorch 2.11.0+cu128: works.** **CatBoost GPU: does not** (see §6)
- LightGBM's Windows pip wheel is CPU-only by design — it uses 30 of 32 threads
- Host: 33.5 GB RAM, 32 cores

---

## Appendix — running it

```
python -m pipeline.run_all                  # build data; skips what is current
python -m pipeline.run_all --force          # rebuild from the 42 GB source
python -m pipeline.tools.check_env                # verify GPU + library versions
python -m pipeline.tools.verify                   # faithfulness, parity, leakage checks
python -m pipeline.stages.s11_train --phase a      # feature-set ablation
python -m pipeline.stages.s11_train --phase b --time-budget 600
python -m pipeline.stages.s12_baselines            # held-out evaluation + baselines
```

Environment: `uv venv` on Python 3.12, pinned in `pipeline/requirements.lock.txt`.
Stack: DuckDB (out-of-core CSV), Polars (dataframes), LightGBM / XGBoost / CatBoost,
Optuna, SHAP.

### Glossary

- **MIMIC-IV** — public database of de-identified US intensive-care records.
- **Admission (`stay_id`)** — one ICU stay. **Patient (`subject_id`)** — one person, who
  may have several admissions. Splits group on *patient*.
- **Invasive ventilation** — breathing supported by a machine through a tube into the windpipe.
- **LOCF** — "last observation carried forward": reusing the most recent real reading to
  fill a gap, here only while under four hours old, since a stale setting is not current.
- **Average precision (AP)** — how well a model finds rare positives. Must be read against
  the base rate: at 5.26% positives, random ≈ 0.053, so 0.6395 is ~12× better than chance.
  **ROC-AUC** looks flatteringly high when positives are rare.
- **Brier score** — calibration: whether a predicted "70% risk" happens about 70% of the time.
- **Leakage** — information reaching the model that it would not have at prediction time.
- **SHAP** — attributing a prediction to individual input features.
