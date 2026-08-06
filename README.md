# Pulsemind — ICU ventilation risk

Clinical decision-support prototype for **mechanical-ventilation risk monitoring** in
intensive care: stream ventilator telemetry → score it with a model → have an LLM produce a
plain-language rationale → surface a prioritised risk board. Read-only and
clinician-in-the-loop; it never controls a ventilator and never recommends treatment.

This repository holds two things, kept deliberately separate:

| | |
|---|---|
| [`pipeline/`](pipeline) + [`reports/`](reports) | **Current work.** A DuckDB/Polars build over MIMIC-IV 3.1 and the measured results. |
| [`legacy/`](legacy) | **The previous generations**, kept as the reason the current work exists. Not on the current path — see [`legacy/README.md`](legacy/README.md). |

The serving stack (`backend/`, `edge/`, `engine_fcp/`, `proxy/`, `docker-compose.yml`) sits at
the root because it is still the only runnable end-to-end demo. It has known defects, listed
in `legacy/README.md`.

---

## Layout

```
pipeline/            the MIMIC-IV pipeline, stages s01 → s15
reports/             FINDINGS.md and the result JSONs — the deliverable
backend/ edge/       FastAPI orchestrator and edge aggregator
engine_fcp/ proxy/   gRPC model server, nginx mTLS termination
certs/               demo certificates
rag document/        source PDFs for the RAG branch (designed, not built)
legacy/              Gen-1 waveform models, LoRA work, superseded notebooks, old data
```

Large and credentialed things live **outside** this repo, in the workspace beside it:

```
<workspace>/data     MIMIC-IV 3.1 (~97 GB) and the BigQuery exports
<workspace>/build    pipeline artifacts (~420 MB) — regenerable
<workspace>/models   fitted models (~152 MB) — regenerable
```

Each is overridable via `PM_DATA_ROOT`, `PM_BUILD_ROOT`, `PM_MODELS_ROOT`, or all three at
once with `PM_WORKSPACE`. `pipeline/config.py` is the single place paths are defined; no stage
hardcodes one.

## Running the pipeline

Python 3.12, PyTorch cu128 (the dev box is an RTX 5060, Blackwell `sm_120`). From this
directory:

```powershell
..\.venv\Scripts\python.exe -m pipeline.run_all                 # build data, skip what is current
..\.venv\Scripts\python.exe -m pipeline.run_all --force         # rebuild from the 42 GB source
..\.venv\Scripts\python.exe -m pipeline.run_all --with-training # add the bake-off and evaluation
..\.venv\Scripts\python.exe -m pipeline.verify                  # faithfulness, parity, leakage checks
```

Every stage writes a manifest of its input fingerprints and config hash and **skips itself
when nothing moved**, so a re-run recomputes only what actually changed. Only `s03` reads the
42 GB `chartevents.csv`; it caches a compact superset that every later stage reads instead, so
changing the cohort or a column definition costs seconds rather than another full scan.

Without the MIMIC tree present the package still imports — the code and reports are readable
on their own — and `run_all` fails immediately with a message telling you which tables are
missing and which variable to set.

## What the pipeline established

Training data went from **286 admissions / 3,288 rows** to **39,319 admissions / 4,200,041
rows** across **33,045 patients**, evaluated on 6,609 held-out patients sharing none of them.
Four findings matter more than that headline, all in [`reports/FINDINGS.md`](reports/FINDINGS.md):

1. **The feature representation was the binding constraint** — not data volume, and not PCA.
   At identical rows and model, swapping 11 raw values for 55 derived columns moves average
   precision 0.1290 → 0.5015.
2. **The `warning` label is mostly documentation behaviour.** A model shown only *which*
   measurements were charted and how stale they are — no physiological values — reaches 84.6%
   of achievable ranking skill and 95.2% of achievable discrimination.
3. **The scores rank well but are not probabilities.** Every raw model scores worse than a
   constant base-rate forecast on Brier, because `scale_pos_weight` trains against a
   reweighted prior nothing undoes. Rescaling fixes it at *exactly zero* cost to ranking.
4. **Composite deterioration is the better target, measured.** Against `warning` on identical
   patients, rows, features and parameters, it delivers **2.4× more physiology-attributable
   ranking skill and 5.8× more physiology-attributable discrimination**, with non-overlapping
   bootstrap intervals. Cohen's κ between the two labels is 0.076 — they are not the same
   event renamed.

## Data licensing

**MIMIC-IV is credentialed data under a PhysioNet Data Use Agreement and must not be
committed here.** That covers the raw tables, BigQuery exports, and per-patient extracts
derived from them. `.gitignore` enforces this; `legacy/data/updated_cleaned_training_data/`
was untracked for exactly this reason and remains only on local disk.
