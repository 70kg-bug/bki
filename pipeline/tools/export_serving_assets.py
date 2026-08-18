"""Export what a serving process needs and models/ does not already hold.

The calibrator carries `feature_order`. Two things are still missing, and both
fail silently:

  CATEGORY LEVELS  s11_train.load() does `pdf[c].astype("category")`, so the
      code behind "MICU" is its position in the sorted unique values OF THE ROW
      SET IT LOADED. XGBoost splits on the code, so a level set from a different
      row set is a different model that still returns a probability. Captured
      here from the same censored row set training used.

  COHORT MEDIANS  `{p}_final` is `{p}_locf` filled with the TRAIN-SPLIT median
      (s07_impute.py:86).

Neither is patient data: 147 category strings and 11 medians.

Run from bki/:  ..\\.venv\\Scripts\\python.exe -m pipeline.tools.export_serving_assets
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import polars as pl

from .. import config as C
from ..common import align_forward_labels, console, log

# The nine columns s10_assemble declares categorical. Listed rather than
# imported because core/ must not depend on a stage; asserted against
# features.json below, so a drift cannot pass.
CATEGORICAL = ["gender", "admission_type", "admission_location", "insurance",
               "language", "marital_status", "race", "first_careunit",
               "ventilator_mode"]


def categorical_levels() -> dict[str, list[str]]:
    """The exact `category` levels the booster was fitted against.

    s11_train.load() for the nine categorical columns only, over the same
    censored row set -- that identity is the part that matters.
    """
    n_rows = pl.read_parquet(C.MODEL_MATRIX_PQ, columns=["stay_id"]).height
    observed = align_forward_labels(n_rows)[C.TARGET].is_not_null().to_numpy()
    log(f"model matrix {n_rows:,} rows, {int((~observed).sum()):,} censored "
        f"({100 * (~observed).mean():.2f}%) -- levels taken from the rest")

    frame = (pl.read_parquet(C.MODEL_MATRIX_PQ, columns=CATEGORICAL)
               .filter(pl.Series(observed))
               .to_pandas())
    levels = {}
    for column in CATEGORICAL:
        series = frame[column].astype("category")
        levels[column] = [str(v) for v in series.cat.categories]
        nulls = int(frame[column].isna().sum())
        log(f"  {column:<20} {len(levels[column]):>3} levels, {nulls:>9,} null")
    return levels


def main() -> None:
    console.rule("[bold cyan]Export serving assets")

    manifest = json.loads((C.BUILD / "features.json").read_text())
    declared = [c for c in manifest["categorical"]]
    assert declared == CATEGORICAL, (
        f"categorical list drifted from features.json: {declared} != {CATEGORICAL}")

    # From the calibrator, not features.json: load_scorer() asserts against the
    # calibrator, so that is what decides whether a frame is scoreable.
    import joblib
    payload = joblib.load(C.CALIBRATOR_PKL)
    feature_order = [str(c) for c in payload["feature_order"]]
    # Element-wise, not by length: two 109-element lists in a different order
    # both pass a length check, and order decides every categorical's code.
    if feature_order != list(manifest["sets"]["full"]):
        raise RuntimeError(
            "calibrator feature_order and features.json disagree; the first "
            "difference is at index "
            f"{next(i for i, (a, b) in enumerate(zip(feature_order, manifest['sets']['full'])) if a != b)}")
    log(f"feature order: {len(feature_order)} columns, from {C.CALIBRATOR_PKL.name}")

    reference = json.loads(C.REFERENCE_STATS_JSON.read_text())
    assert set(reference) == set(C.FROZEN_ORDER), (
        "reference_stats.json does not cover the frozen parameters")

    assets = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "model": C.MODEL_XGB.name,
            "calibrator": C.CALIBRATOR_PKL.name,
            "band_table": C.BAND_TABLE_JSON.name,
            "target": C.TARGET,
            "levels_from": "model_matrix.parquet, censored to the labelled row set",
            "reference_from": C.REFERENCE_STATS_JSON.name,
        },
        "feature_order": feature_order,
        "categorical": categorical_levels(),
        "reference_medians": reference,
        "frozen_params": list(C.FROZEN_ORDER),
        "locf_cutoff_min": C.LOCF_CUTOFF_MIN,
        # Fingerprinted config serving would otherwise hardcode. Flip either
        # and rebuild: the column NAMES stay the same while the values change,
        # which no column check can see.
        "contrib_top_k": C.CONTRIB_TOP_K,
        "charlson_hierarchy": C.CHARLSON_HIERARCHY,
        # From features.json, not a name pattern: this is the set every stage
        # measures doc-share against (s17_records.feature_kinds).
        "documentation_features": list(manifest["sets"]["documentation_only"]),
        "feature_group": {c: g for g, cs in manifest["groups"].items() for c in cs},
    }

    out = C.MODELS / "serving_assets.json"
    out.write_text(json.dumps(assets, indent=2))
    log(f"[green]wrote {out} "
        f"({out.stat().st_size / 1024:.1f} KB)[/green]")


if __name__ == "__main__":
    main()
