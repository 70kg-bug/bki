"""Run the whole pipeline end to end.

Stages skip themselves when their inputs and config are unchanged, so a re-run
only recomputes what actually moved. Pass --force to rebuild everything.

    python -m pipeline.run_all                 # build data, skip what is current
    python -m pipeline.run_all --force         # rebuild from the 42 GB source
    python -m pipeline.run_all --with-training # also run the bake-off (slow)
"""
from __future__ import annotations

import argparse
import time

from . import config as C
from .common import accounting_table, console, log
from . import (s01_cohort_strict, s02_table1_static, s03_extract_cache,
               s04_cohort_final, s05_pivot, s06_split, s07_impute,
               s08_table3_interventions, s09_table4_outcomes, s10_assemble,
               s14_forward_targets)

DATA_STAGES = [
    ("cohort (strict)", s01_cohort_strict),
    ("cache -- THE ONLY 42 GB SCAN", s03_extract_cache),
    ("cohort (final)", s04_cohort_final),
    # AFTER s04, not before it. Table 1 must be built on the same cohort every
    # later stage uses: when it ran second, against the strict cohort, the
    # 7,423 admissions s04 adds arrived at the model matrix with all 33 static
    # features NULL -- and that null pattern identified the cohort arm exactly,
    # handing the model a stratum flag it had no business seeing.
    ("Table 1 -- patient background", s02_table1_static),
    ("Table 2 -- pivot", s05_pivot),
    ("splits (before imputation, on purpose)", s06_split),
    ("Table 2 -- imputation", s07_impute),
    ("Table 3 -- interventions", s08_table3_interventions),
    ("Table 4 -- outcomes", s09_table4_outcomes),
    ("assemble model matrix", s10_assemble),
    # Candidate target D. Kept out of the model matrix on purpose: a forward
    # label must never be reachable as a feature.
    ("forward targets (candidate D)", s14_forward_targets),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-training", action="store_true")
    ap.add_argument("--trials", type=int, default=12)
    # CatBoost is excluded by default on HARDWARE grounds, not accuracy: its
    # GPU build ships no sm_120 kernels (CUDA error 218) and cannot compute
    # PRAUC, and on CPU it takes ~1,883 s per fit even capped at 800
    # iterations -- a 5-fold CV would cost about 2.6 hours. s11's own default
    # still lists it, so this passthrough exists to stop `run_all
    # --with-training` quietly committing to that.
    ap.add_argument("--algos", default="lightgbm,xgboost",
                    help="comma-separated; catboost is excluded by default")
    a = ap.parse_args()

    # Fail on a misconfigured data root now, not as a DuckDB IO error six
    # minutes into the 42 GB scan.
    C.require_mimic()

    t0 = time.time()
    for i, (label, mod) in enumerate(DATA_STAGES, 1):
        console.print(f"[bold white on blue] {i}/{len(DATA_STAGES)} [/] {label}")
        mod.main(force=a.force)

    accounting_table()
    log(f"data pipeline finished in {time.time() - t0:.0f}s")

    if a.with_training:
        from . import (s11_train, s12_baselines, s13_calibrate, s15_target_compare,
                       s16_bands, s17_records, s18_explain)
        import sys
        sys.argv = ["s11", "--phase", "all", "--trials", str(a.trials),
                    "--algos", a.algos]
        s11_train.main()
        sys.argv = ["s12"]
        s12_baselines.main()
        # s13 needs a fitted model, so it only runs alongside training. It scores
        # the LEVEL of the output -- s11/s12 only ever scored the ranking.
        s13_calibrate.main(force=a.force)
        # s15 refits on each candidate target, so it belongs here too.
        s15_target_compare.main(force=a.force)
        # s16 re-scores from the artifacts s13 persisted, so it must follow it.
        s16_bands.main(force=a.force)
        # s17 consumes s16's band table as its contract, so it must follow that.
        s17_records.main(force=a.force)
        # s18 decides what may be said about each record, so it follows s17. It
        # runs its adversarial suite against the grounding checker before it
        # emits anything, and fails the run if a corruption goes uncaught.
        s18_explain.main(force=a.force)


if __name__ == "__main__":
    main()
