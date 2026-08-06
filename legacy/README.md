# Legacy — the previous modelling generations

Everything in this directory predates the MIMIC-IV pipeline in [`../pipeline/`](../pipeline).
It is kept because it is the reason the current work exists, not because it still runs.
**Nothing here is on the current path. Do not extend it.**

The serving stack (`../backend`, `../edge`, `../engine_fcp`, `../proxy`, `../certs`,
`../docker-compose.yml`) deliberately stays at the repo root — it is still the only runnable
end-to-end demo — but it consumes the artifacts stored here.

---

## What is here

### Generation 1 — 7-patient waveform models (`train/`, `data/old_training_data/`)

PhysioNet-style `.xlsx` waveforms (Flow, Pao, Pes) for **seven** patients, labelled
`in_hospital_death`, with P02 and P07 positive. Clinical notes are in Chinese.

| Script | What it did |
|---|---|
| `train/train_bki_models.py` | GradientBoosting + MLP on shuffled rows. Reached >99% accuracy — **data leakage**: a random split across a continuous per-patient time series put 80% of a patient's timeline in training and the rest in test. |
| `train/train_bki_models_advanced.py` | The fix: patient-level split (train `[1,3,4,5,7]`, val `[6]`, test `[2]`), 100-step / stride-20 sliding windows, a 1D-CNN, early stopping, `pos_weight` balancing. |
| `train/train_ventilator_regressor.py` | Unrelated to the clinical goal — the Kaggle *Google Brain* ventilator-pressure dataset. Its input (`data/train.csv`) was never in this repo. |

**The finding that motivates everything downstream**, recorded in the original README and
still correct: once the leakage was removed, *the honest model completely failed to generalise
to the unseen patient*. Seven patients is far too small. Its stated next step was "a massive
scale-up in the dataset" — that is precisely what `../pipeline/` is.

> ⚠️ Do not "fix" this by reverting to a random split. The failure is the result.

### Generation 1.5 — LLM explainability (`generate_lora_dataset.py`, `lora_adapters/`)

`generate_lora_dataset.py` runs the 1D-CNN over the same 7-patient windows and writes
`lora_adapters/cot_dataset.jsonl` — **synthetic, template-generated** chain-of-thought
rationales built from four hardcoded TP/FP/FN/TN paragraphs, formatted with Gemma
`<start_of_turn>` tags. `train/train_lora.py` LoRA-tunes Qwen2-7B on it (r=16, alpha=32,
`q_proj`/`v_proj`, `max_steps=100`, explicitly "short run for demo").

The adapter shipped in `lora_adapters/clinical-lora/` is that demo artifact. It is mounted by
the `vllm` service in `../docker-compose.yml`.

### Generation 2 notebooks (`notebooks/`)

The first MIMIC-IV attempt, superseded by `../pipeline/`:

- `cleaning_and_filling_mising_values_claude.ipynb` — a careful imputation pipeline in a single
  152-line cell. Its `_observed` / `_delta_t_min` / `_locf` / `_structurally_missing_in_stay` /
  `_final` design carried directly into `pipeline/s07_impute.py`, which reproduces it
  **identically on 44/44 columns and thousands of times faster**.
- `training_calinerating.ipynb` — GradientBoosting → LogisticRegression calibrator →
  GradientBoosting regressor. Trained on 3,288 rows from 286 admissions.

### Trained outputs (`output_from_*/`)

Checkpoints from the above. `output_from_advanced_training/` is mounted at `/model` by the
`fcp-former` service.

---

## Known-broken, recorded rather than repaired

These are documented so nobody loses an afternoon to them:

1. **The training scripts' data paths do not resolve.** They read
   `data/waveform_data/P0{i}Waveform.xlsx`, but the seven workbooks actually live at
   `data/old_training_data/waveform_data/`. The folder was nested one level deeper at some
   point and the scripts were never updated — **this was already broken before the
   reorganisation**; moving the files under `legacy/` did not cause it. Run from this
   directory, the literal that would resolve is
   `data/old_training_data/waveform_data/P0{i}Waveform.xlsx`. Left unchanged deliberately:
   this generation is frozen, and an untested edit would misrepresent it as maintained.
2. **The served model is random weights.** `../engine_fcp/server.py` defines a 6-input MLP
   (`feature_extractor.*`) but loads `bki_classifier_advanced_cnn.pt`, whose state dict is
   `conv_blocks.*` / `fc.*` — a 3-channel × 100-timestep 1D-CNN. `load_state_dict` raises, a
   bare `except` swallows it, and the service answers from an **uninitialised** network.
3. **Feature-space mismatch.** The gRPC contract asks for
   `peep, pip, fio2, hrv, procalcitonin, p_f_ratio`. `hrv` and `procalcitonin` appear in no
   training set anywhere in this repository.
4. **Scripts write to the working directory**, not to `output_from_*/`. The checked-in outputs
   were moved there by hand.

---

## A note on `data/updated_cleaned_training_data/`

Those files are per-patient extracts derived from **MIMIC-IV**, which is distributed under a
PhysioNet Data Use Agreement and may not be redistributed. They remain on disk but are **no
longer tracked** by git, and the path is in `../.gitignore`.

Untracking stops further distribution; it does **not** remove them from existing history. If
that matters for your use, that is a separate, deliberate history rewrite.

---

## What replaced this

| | Generation 1 | Generation 2 notebooks | `../pipeline/` |
|---|---|---|---|
| Patients | 7 | 286 admissions | **33,045 patients / 39,319 admissions** |
| Rows | waveform windows | 3,288 | **4,200,041** |
| Split | patient-level (after the fix) | row-level, leaking | patient-level, assert-enforced |
| Result | did not generalise | not measured against a clinical metric | measured, with bootstrap intervals |

See [`../reports/FINDINGS.md`](../reports/FINDINGS.md).
