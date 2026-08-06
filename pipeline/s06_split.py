"""Stage 6 -- train/test split and cross-validation folds, grouped by PATIENT.

Runs before imputation on purpose: the fill values in stage 7 must be computed
from training data only, which is impossible if the split comes afterwards.

Grouping is on subject_id, not stay_id. One patient can have several ICU
admissions, so grouping by admission would still place the same person on both
sides of the split -- the model would score well partly because it had already
met that patient. This is the safeguard the current notebook lost by using a
plain row-level train_test_split.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C
from .common import account_parquet, cached_stage, console, log

TEST_FRACTION = 0.20


def main(force: bool = False) -> None:
    with cached_stage("s06_split", sources=[C.T2_WIDE_PQ], output=C.FOLDS_PQ,
                      force=force) as ran:
        if not ran:
            return

        # Patient-level summary: each patient is one indivisible unit.
        subj = (
            pl.scan_parquet(C.T2_WIDE_PQ)
            .group_by("subject_id")
            .agg(
                pl.len().alias("n_rows"),
                pl.col(C.TARGET).sum().alias("n_pos"),
                pl.col("stay_id").n_unique().alias("n_stays"),
            )
            .collect()
        )
        log(f"patients={subj.height:,}  admissions={subj['n_stays'].sum():,}  "
            f"rows={subj['n_rows'].sum():,}")

        # Stratify on how positive a patient is, so folds are balanced on the
        # target rather than only on patient count.
        rate = (subj["n_pos"] / subj["n_rows"]).to_numpy()
        strat = np.digitize(rate, [1e-12, 0.02, 0.10, 0.30])  # 0 = never positive
        subj = subj.with_columns(pl.Series("strat", strat))
        for b in sorted(set(strat.tolist())):
            m = strat == b
            log(f"  stratum {b}: {m.sum():>6,} patients  "
                f"mean positive rate {rate[m].mean():.4f}")

        ids = subj["subject_id"].to_numpy()
        tr_idx, te_idx = train_test_split(
            np.arange(len(ids)), test_size=TEST_FRACTION,
            random_state=C.RANDOM_SEED, stratify=strat)

        split = np.array(["train"] * len(ids), dtype=object)
        split[te_idx] = "test"
        fold = np.full(len(ids), -1, dtype=int)

        # Inner CV folds, within train only.
        skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True,
                              random_state=C.RANDOM_SEED)
        for k, (_, va) in enumerate(skf.split(tr_idx, strat[tr_idx])):
            fold[tr_idx[va]] = k

        out = subj.select("subject_id", "n_rows", "n_pos", "n_stays").with_columns(
            pl.Series("split", split.astype(str)),
            pl.Series("fold", fold),
        )
        out.write_parquet(C.FOLDS_PQ, compression="zstd")

        tr = out.filter(pl.col("split") == "train")
        te = out.filter(pl.col("split") == "test")
        log(f"train: {tr.height:,} patients  {tr['n_stays'].sum():,} admissions  "
            f"{tr['n_rows'].sum():,} rows  positive {100*tr['n_pos'].sum()/tr['n_rows'].sum():.2f}%")
        log(f"test : {te.height:,} patients  {te['n_stays'].sum():,} admissions  "
            f"{te['n_rows'].sum():,} rows  positive {100*te['n_pos'].sum()/te['n_rows'].sum():.2f}%")
        for k in range(C.N_FOLDS):
            f = out.filter(pl.col("fold") == k)
            log(f"  fold {k}: {f.height:,} patients  {f['n_rows'].sum():,} rows  "
                f"positive {100*f['n_pos'].sum()/max(f['n_rows'].sum(),1):.2f}%")

        # Hard guarantee, not a hope.
        overlap = set(tr["subject_id"].to_list()) & set(te["subject_id"].to_list())
        assert not overlap, f"patient leakage: {len(overlap)} subjects in both splits"
        log("[green]leakage check: no patient appears in both train and test[/green]")

    account_parquet("folds", C.FOLDS_PQ, stay_col="subject_id", subject_col="subject_id")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
