"""Stage 10 -- assemble the model matrix.

Joins Table 1 (broadcast per admission), Table 2 (native grain) and Table 3
(already resolved per timestamp) onto one row per (stay_id, charttime).

Also writes features.json: which columns exist, which are categorical, which
source table each came from, and the named feature sets the bake-off compares.
Column *names* are left alone so the like-for-like diff against raw-query.csv
still works; provenance is recorded in the manifest instead of by renaming.

Guards enforced here, not hoped for:
  * every leaky_/outcome column is excluded from the feature list
  * no identifier reaches the model
  * Table 3 really did use strictly-before intervals
"""
from __future__ import annotations

import json

import polars as pl

from . import config as C
from .common import account_parquet, cached_stage, log

FEATURES_JSON = C.BUILD / "features.json"

# Never model inputs: identifiers, timestamps, split bookkeeping, the target.
NON_FEATURES = {"stay_id", "subject_id", "hadm_id", "charttime", C.TARGET,
                "split", "fold"}

CATEGORICAL = ["gender", "admission_type", "admission_location", "insurance",
               "language", "marital_status", "race", "first_careunit",
               "ventilator_mode"]


def main(force: bool = False) -> None:
    sources = [C.T2_IMPUTED_PQ, C.T1_STATIC_PQ, C.T3_INTERV_PQ, C.FOLDS_PQ]
    with cached_stage("s10_assemble", sources=sources, output=C.MODEL_MATRIX_PQ,
                      force=force) as ran:
        if not ran:
            return

        t2 = pl.read_parquet(C.T2_IMPUTED_PQ)
        n_base, stays_base = t2.height, t2["stay_id"].n_unique()
        log(f"Table 2 base grain: {n_base:,} rows / {stays_base:,} admissions")

        t3 = pl.read_parquet(C.T3_INTERV_PQ)
        t1 = pl.read_parquet(C.T1_STATIC_PQ).drop(["subject_id", "hadm_id"])
        folds = pl.read_parquet(C.FOLDS_PQ).select("subject_id", "split", "fold")

        # Table 1 columns that describe an outcome or a post-discharge fact.
        leaky = [c for c in t1.columns if c.startswith("leaky_")]
        t1_feat = t1.drop(leaky + ["vent_start", "vent_end"])
        log(f"Table 1: {t1_feat.width - 1} feature columns "
            f"({len(leaky)} leaky_ columns dropped: {', '.join(leaky[:4])}...)")

        df = (t2
              .join(t3, on=["stay_id", "charttime"], how="left")
              .join(t1_feat, on="stay_id", how="left")
              .join(folds, on="subject_id", how="left"))

        assert df.height == n_base, (
            f"join changed the row count: {n_base:,} -> {df.height:,}")
        assert df["stay_id"].n_unique() == stays_base, "join changed the admission count"
        log(f"[green]join integrity: {df.height:,} rows, "
            f"{df['stay_id'].n_unique():,} admissions -- unchanged[/green]")

        # --- reverse-causation guard on Table 3 -------------------------------
        for g in ("vasopressor", "sedative", "opioid", "paralytic"):
            col = f"{g}_minutes_since_start"
            if col in df.columns:
                bad = df.filter((pl.col(col) != -1) & (pl.col(col) <= 0)).height
                assert bad == 0, (
                    f"{col}: {bad:,} rows have an infusion starting at or after the "
                    f"row timestamp -- the strictly-before guard failed")
        log("[green]reverse-causation guard: every intervention began strictly "
            "before its row[/green]")

        df.write_parquet(C.MODEL_MATRIX_PQ, compression="zstd")

        # ------------------------------------------------------------------
        # Feature manifest
        # ------------------------------------------------------------------
        all_cols = [c for c in df.columns if c not in NON_FEATURES]
        t2_cols, t1_cols, t3_cols = [], [], []
        t1_names = set(t1_feat.columns) - {"stay_id"}
        t3_names = set(t3.columns) - {"stay_id", "charttime"}
        for c in all_cols:
            if c in t1_names:
                t1_cols.append(c)
            elif c in t3_names:
                t3_cols.append(c)
            else:
                t2_cols.append(c)

        final_only = [f"{p}_final" for p in C.FROZEN_ORDER]
        doc_only = ([f"{p}_observed" for p in C.FROZEN_ORDER]
                    + [f"{p}_delta_t_min" for p in C.FROZEN_ORDER]
                    + [f"{p}_structurally_missing_in_stay" for p in C.FROZEN_ORDER])
        # Raw (pre-imputation) columns duplicate _locf/_final information and are
        # mostly null by construction -- keep them out of the modelling sets.
        t2_model = [c for c in t2_cols if c not in C.FROZEN_ORDER]

        manifest = {
            "target": C.TARGET,
            "group_key": C.GROUP_KEY,
            "categorical": [c for c in CATEGORICAL if c in df.columns],
            "groups": {"t1_static": t1_cols, "t2_timeseries": t2_cols,
                       "t3_interventions": t3_cols},
            "sets": {
                "final_only": final_only,
                "t2_all": t2_model,
                "full": t2_model + t1_cols + t3_cols,
                "documentation_only": doc_only,
            },
        }
        FEATURES_JSON.write_text(json.dumps(manifest, indent=2))

        log(f"columns by source -- Table 1: {len(t1_cols)}  Table 2: {len(t2_cols)}  "
            f"Table 3: {len(t3_cols)}")
        for name, cols in manifest["sets"].items():
            log(f"  feature set '{name}': {len(cols)} columns")

        leaked = [c for c in manifest["sets"]["full"]
                  if c.startswith("leaky_") or c in NON_FEATURES]
        assert not leaked, f"leaky or identifier columns reached the feature set: {leaked}"
        log("[green]feature guard: no identifier, outcome or leaky_ column in any "
            "feature set[/green]")

    account_parquet("model matrix", C.MODEL_MATRIX_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
