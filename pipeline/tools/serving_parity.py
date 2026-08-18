"""Does the serving-time feature assembly reproduce the pipeline's own numbers?

`core/features.py` restates s07/s08/s02 as a per-stay state machine. A restated
definition that is subtly wrong still produces a probability, so this replays
real stays through it and diffs against what the pipeline actually emitted.

Four checks, in increasing order of what they can tell you:

  A  TIME SERIES   Replay each reading's measured values through StayFeatures and
     diff the 55 s07 columns against the model matrix. Everything except
     `_structurally_missing_in_stay` must match exactly.

  B  FRAME         Assemble a row the serving way -- pd.Categorical with the
     exported level set, float32 everywhere else -- from the matrix's own values,
     score it, and diff against `risk_records.jsonl.gz`. This is the check that
     catches a wrong category level set, which is otherwise silent.

  C  CAUSAL COST   Swap in the causal forms of the twelve unservable features and
     re-score. The band-disagreement rate here is the real cost of serving this
     model live, and it is the number the demo has to be honest about.

  D  BANDS         Push both score series through BandStepper and compare the
     displayed band, not just the probability -- a score shift only matters if it
     moves a patient across a cut.

  G  PUSH          `tools.push_parity`: every one of the 109 columns, through
     the real `push()`, against real ICD codes, infusions and mode events. A
     and B between them exercise 55 columns and none of the assembly, which is
     how a wrong `ventilator_mode` survived both.

Sections C, D and F measure; A, B, E and G assert. **Exits non-zero when one of
those four fails** -- an earlier version could only print.

Run from bki/:  ..\\.venv\\Scripts\\python.exe -m pipeline.tools.serving_parity
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from .. import config as C
from ..common import console, log
from ..core import bands as B
from ..core import records as R
from ..core.features import PatientContext, Reading, ServingAssets, StayFeatures
from ..core.scoring import load_scorer
from . import push_parity

STAYS = 120         # enough rows for the disagreement rate to settle
TOLERANCE = 1e-9    # same device, same booster: this should be exact
SHARE_ROWS = 400    # TreeSHAP per row is the slow part; 400 settles the deltas


def stored_records(stay_ids: set[int]) -> dict[tuple[int, str], dict]:
    """The pipeline's own output for these stays, keyed by (stay_id, charttime)."""
    out = {}
    with gzip.open(C.RECORDS_JSONL, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["stay_id"] in stay_ids:
                out[(r["stay_id"], r["charttime"])] = r
    return out


def replay_timeseries(rows: pl.DataFrame, assets: ServingAssets,
                      context: PatientContext) -> list[dict]:
    """Feed one stay's measured values through StayFeatures, reading by reading.

    The raw parameter columns survive into the model matrix (s07 keeps them in
    `ordered`), so the measured values are recoverable exactly -- no need to
    invert `_locf`.
    """
    state = StayFeatures(context, assets)
    produced = []
    for row in rows.iter_rows(named=True):
        values = {p: row[p] for p in assets.frozen_params if row[p] is not None}
        reading = Reading(observed_at=row["charttime"], values=values)
        produced.append(state._timeseries_row(reading))
    return produced


def main() -> None:
    console.rule("[bold cyan]Serving parity")

    assets = ServingAssets.load(C.MODELS / "serving_assets.json")
    scorer = load_scorer()
    log(f"scoring device: [bold]{scorer.device}[/bold]  "
        f"features: {len(assets.feature_order)}")

    # --- pick stays that the record file actually covers ------------------
    covered = []
    with gzip.open(C.RECORDS_JSONL, "rt") as fh:
        for line in fh:
            covered.append(json.loads(line)["stay_id"])
            if len(covered) > 40_000:
                break
    stay_ids = sorted(set(covered))[:STAYS]
    records = stored_records(set(stay_ids))
    log(f"{len(stay_ids)} stays, {len(records):,} stored records")

    ts_cols = [f"{p}{s}" for p in assets.frozen_params for s in C.SUFFIXES]
    raw_cols = list(assets.frozen_params)
    other = [c for c in assets.feature_order if c not in ts_cols]
    cols = list(dict.fromkeys(["stay_id", "charttime"] + raw_cols + ts_cols + other
                              + ["height_cm", "weight_kg", "gender"]))
    matrix = (pl.read_parquet(C.MODEL_MATRIX_PQ, columns=cols)
                .filter(pl.col("stay_id").is_in(stay_ids))
                .sort("stay_id", "charttime"))
    log(f"matrix slice: {matrix.height:,} rows")

    # ------------------------------------------------------------------ A
    console.rule("[cyan]A -- time-series restatement")
    mismatch = {c: 0 for c in ts_cols}
    compared = 0
    for stay_id in stay_ids:
        rows = matrix.filter(pl.col("stay_id") == stay_id)
        if rows.is_empty():
            continue
        context = PatientContext(
            sex=rows["gender"][0] or "F", age_at_icu=60.0,
            admission_type="", admission_location="", race="", first_careunit="",
            ventilation_start=rows["charttime"][0],
            height_cm=rows["height_cm"][0], weight_kg=rows["weight_kg"][0])
        produced = replay_timeseries(rows, assets, context)
        for i, got in enumerate(produced):
            compared += 1
            for c in ts_cols:
                want = rows[c][i]
                have = got[c]
                if want is None and have is None:
                    continue
                if want is None or have is None or abs(float(want) - float(have)) > 1e-4:
                    mismatch[c] += 1

    bad = {c: n for c, n in mismatch.items() if n}
    log(f"compared {compared:,} rows x {len(ts_cols)} columns")
    expected = {c for c in ts_cols if c.endswith("_structurally_missing_in_stay")}
    unexpected = {c: n for c, n in bad.items() if c not in expected}
    for c, n in sorted(bad.items(), key=lambda kv: -kv[1])[:8]:
        tag = "[yellow]causal, expected[/yellow]" if c in expected else "[red]UNEXPECTED[/red]"
        log(f"  {c:<52} {n:>8,} ({100*n/compared:5.2f}%)  {tag}")
    if not unexpected:
        log("[green]every causal s07 column reproduces exactly[/green]")
    else:
        log(f"[red]{len(unexpected)} columns disagree that should not[/red]")

    # ------------------------------------------------------------------ B
    console.rule("[cyan]B -- frame construction and scoring")
    keys = [(int(s), t) for s, t in zip(matrix["stay_id"], matrix["charttime"])]
    have_record = [i for i, (s, t) in enumerate(keys)
                   if (s, _iso(t)) in records]
    log(f"{len(have_record):,} of {matrix.height:,} rows have a stored record "
        f"({matrix.height - len(have_record):,} censored)")

    frame = _as_model_frame(matrix.select(list(assets.feature_order)), assets)
    calibrated = scorer.score(frame)
    stored = np.array([records[(keys[i][0], _iso(keys[i][1]))]["risk"]["calibrated"]
                       for i in have_record])
    mine = calibrated[have_record]
    delta = np.abs(mine - stored)
    log(f"score delta vs stored records: max {delta.max():.3e}  "
        f"mean {delta.mean():.3e}  (tol {TOLERANCE:.0e})")
    if delta.max() <= TOLERANCE:
        log("[green]frame construction, category levels and dtypes are exact[/green]")
    else:
        log("[red]the serving frame does not reproduce the pipeline's score[/red]")

    # ------------------------------------------------------------------ C
    console.rule("[cyan]C -- what the twelve causal features cost")
    causal = matrix.select(list(assets.feature_order)).to_pandas()

    # "Never observed SO FAR in this stay", against the trained form's "never
    # observed anywhere in the stay". Rows are already sorted by (stay_id,
    # charttime), so a running total within the stay is the causal answer.
    seen_so_far = matrix.select([
        (pl.col(f"{p}_observed").cum_sum().over("stay_id") > 0).alias(p)
        for p in assets.frozen_params])
    for p in assets.frozen_params:
        causal[f"{p}_structurally_missing_in_stay"] = np.where(
            seen_so_far[p].to_numpy(), 0.0, 1.0).astype(np.float32)

    # "Hours since ventilation started", against the trained form's total
    # episode length. The stay's first reading stands in for the start.
    elapsed = matrix.select(
        ((pl.col("charttime") - pl.col("charttime").min().over("stay_id"))
         .dt.total_seconds() // 3600).alias("h"))
    causal["vent_hours"] = elapsed["h"].to_numpy().astype(np.float32)

    # Decomposed, because the two redefinitions are not the same kind of change.
    # The eleven flags only narrow their observation window; `vent_hours` changes
    # MEANING, from total episode length to time elapsed. A combined figure would
    # hide which one is doing the damage, and they have different fixes.
    smis = [f"{p}_structurally_missing_in_stay" for p in assets.frozen_params]
    variants = {
        "structurally_missing x11": smis,
        "vent_hours": ["vent_hours"],
        "both": smis + ["vent_hours"],
    }
    scores, shifts = {}, {}
    for label, columns in variants.items():
        frame_v = matrix.select(list(assets.feature_order)).to_pandas()
        for column in columns:
            frame_v[column] = causal[column].to_numpy()
        scores[label] = scorer.score(_as_model_frame(pl.from_pandas(frame_v), assets))
        shifts[label] = np.abs(scores[label] - calibrated)
        log(f"  {label:<26} mean {shifts[label].mean():.4f}  "
            f"p99 {np.percentile(shifts[label], 99):.4f}  "
            f"max {shifts[label].max():.4f}")

    # ------------------------------------------------------------------ D
    console.rule("[cyan]D -- does the shift move anyone across a cut")
    machine = B.BandMachine.from_json(
        json.loads(C.BAND_TABLE_JSON.read_text())["machine"])
    trained_bands = _replay_bands(machine, matrix, calibrated)
    band_rates = {}
    for label in variants:
        got = _replay_bands(machine, matrix, scores[label])
        disagree = sum(a != b for a, b in zip(trained_bands, got))
        band_rates[label] = disagree / len(trained_bands)
        log(f"  {label:<26} {disagree:>6,} of {len(trained_bands):,} readings "
            f"({100*band_rates[label]:5.2f}%)")
    rate = band_rates["both"]

    # ------------------------------------------------------------------ E
    console.rule("[cyan]E -- core/records.py against s17's own numbers")
    record_parity = _record_parity(frame, matrix, keys, records, scorer, assets)

    # ------------------------------------------------------------------ F
    console.rule("[cyan]F -- why vent_hours dominates: it is not a feature at t")
    leak = _vent_hours_leak_check(records)

    # ------------------------------------------------------------------ G
    console.rule("[cyan]G -- the whole of push(), against real sources")
    push = push_parity.run(assets)

    report = {
        "stays": len(stay_ids),
        "rows": matrix.height,
        "scoring_device": scorer.device,
        "frame_parity": {
            "rows_compared": len(have_record),
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "tolerance": TOLERANCE,
            "exact": bool(delta.max() <= TOLERANCE),
        },
        "timeseries_parity": {
            "rows_compared": compared,
            "columns_disagreeing_unexpectedly": sorted(unexpected),
            "structurally_missing_disagreement": {
                c: n for c, n in bad.items() if c in expected},
        },
        "causal_cost": {
            "note": ("What it costs to serve this model without future "
                     "information. The trained form of these twelve features is "
                     "not obtainable at time t, so this is a floor on serving "
                     "error, not a defect to be fixed by better engineering."),
            "redefined": {
                "*_structurally_missing_in_stay (11)":
                    "never observed in the stay -> not observed so far",
                "vent_hours":
                    "total episode length -> hours since ventilation started",
            },
            "by_variant": {
                label: {
                    "score_shift_mean": float(shifts[label].mean()),
                    "score_shift_p99": float(np.percentile(shifts[label], 99)),
                    "score_shift_max": float(shifts[label].max()),
                    "band_disagreement_rate": band_rates[label],
                }
                for label in variants
            },
        },
        "record_parity": record_parity,
        "vent_hours_leak": leak,
        "push_parity": push,
    }
    C.RPT_TOOL_CAUSAL_PARITY.write_text(json.dumps(report, indent=2))
    log(f"[green]wrote {C.RPT_TOOL_CAUSAL_PARITY}[/green]")

    # A harness that cannot fail is a harness that is not checked. Sections C, D
    # and F measure rather than assert -- the causal cost is a property of the
    # model, not a defect -- so only A, B, E and G decide the exit code.
    failed = {
        "A timeseries": bool(unexpected),
        "B frame parity": bool(delta.max() > TOLERANCE),
        "E record parity": not record_parity["ok"],
        "G push parity": not push["ok"],
    }
    console.rule("[bold cyan]Verdict")
    for name, broke in failed.items():
        log(f"  {name:<18} {'[red]FAIL[/red]' if broke else '[green]pass[/green]'}")
    raise SystemExit(1 if any(failed.values()) else 0)


def _record_parity(frame, matrix: pl.DataFrame, keys, records: dict,
                   scorer, assets: ServingAssets) -> dict:
    """Does the scalar restatement in core/records.py agree with s17?

    s17 computes the shares with column algebra over the whole cohort; the
    service computes them one row at a time. Same definition, different shape,
    so the only honest thing to do is diff them against real output.
    """
    rows = [i for i, (s, t) in enumerate(keys) if (s, _iso(t)) in records][:SHARE_ROWS]
    contribs, bias = scorer.contributions(frame.iloc[rows])

    worst = {"documentation_share": 0.0, "imputed_share": 0.0,
             "attribution_total": 0.0, "attribution_age_min": 0.0}
    top_feature_agree = 0
    reconstructed = 0
    dated_defaults = [0, 0]   # (population_reference WITH an age, total telemetry)
    for n, i in enumerate(rows):
        want = records[(keys[i][0], _iso(keys[i][1]))]
        got = R.attribution(frame.iloc[[i]], contribs[n], float(bias[n]), assets)
        for field in ("documentation_share", "imputed_share", "attribution_total"):
            worst[field] = max(worst[field], abs(got[field] - want[field]))
        if want["attribution_age_min"] is not None and got["attribution_age_min"] is not None:
            worst["attribution_age_min"] = max(
                worst["attribution_age_min"],
                abs(got["attribution_age_min"] - want["attribution_age_min"]))
        top_feature_agree += (got["contributors"][0]["feature"]
                              == want["contributors"][0]["feature"])
        reconstructed += R.reconstructs(got, want["risk"]["calibrated"])

        # The data dictionary says age is null whenever source is
        # population_reference. s17 reports the raw age, so a parameter charted
        # beyond the 240-minute LOCF cutoff has both. Count it rather than
        # assume it cannot happen.
        for t in R.telemetry_from_frame(frame.iloc[[i]], assets):
            if t.source == "population_reference" and t.age_min is not None:
                dated_defaults[0] += 1
            dated_defaults[1] += 1

    log(f"compared {len(rows):,} records")
    for field, delta in worst.items():
        log(f"  {field:<22} max |delta| {delta:.2e}")
    log(f"  top contributor agrees on {top_feature_agree:,}/{len(rows):,}")
    log(f"  record reconstructs its own score: {reconstructed:,}/{len(rows):,}")
    log(f"  cohort defaults carrying an age: {dated_defaults[0]:,} of "
        f"{dated_defaults[1]:,} telemetry entries "
        f"({100*dated_defaults[0]/max(dated_defaults[1],1):.2f}%) -- "
        f"the dictionary says this should be none")
    # Both sides round to four decimal places before comparison, so one ulp of
    # that rounding is the floor -- anything under 5e-4 is agreement, not drift.
    ok = (max(worst.values()) < 5e-4 and top_feature_agree == len(rows)
          and reconstructed == len(rows))
    log("[green]core/records.py reproduces s17[/green]" if ok else
        "[red]core/records.py disagrees with s17[/red]")
    return {"rows": len(rows), "max_abs_delta": worst,
            "cohort_defaults_carrying_an_age": dated_defaults[0],
            "telemetry_entries": dated_defaults[1],
            "top_contributor_agreement": top_feature_agree / len(rows),
            "reconstruction_rate": reconstructed / len(rows), "ok": bool(ok)}


def _vent_hours_leak_check(records: dict) -> dict:
    """Is `vent_hours` a fact about the reading, or about how the stay ended?

    s01_cohort_strict.py:50 computes it as `date_diff('hour', vent_start,
    vent_end)` and s02 broadcasts it per admission, so every reading in a stay
    carries the TOTAL length of the episode -- including the first, before any
    of that time has elapsed. features.json documents `full` as "109 features,
    all at or before t". If this is constant within a stay and generally exceeds
    the span actually observed by then, it is not.

    Measured over the whole matrix rather than the sample, because the claim is
    about the training data and not about this harness.
    """
    df = pl.read_parquet(C.MODEL_MATRIX_PQ,
                         columns=["stay_id", "charttime", "vent_hours"])
    per_stay = df.group_by("stay_id").agg([
        pl.col("vent_hours").n_unique().alias("distinct"),
        pl.col("vent_hours").first().alias("total_hours"),
        pl.len().alias("rows"),
        ((pl.col("charttime").max() - pl.col("charttime").min())
         .dt.total_seconds() / 3600).alias("observed_span_h"),
    ])
    constant = int(per_stay.filter(pl.col("distinct") == 1).height)
    longer = per_stay.filter(pl.col("rows") > 5)
    covers = float((longer["total_hours"] >= longer["observed_span_h"]).mean())

    # How much of the decision rests on it, from the pipeline's own attributions.
    shares, top_rank = [], 0
    for r in records.values():
        for i, c in enumerate(r["contributors"]):
            if c["feature"] == "vent_hours":
                shares.append(abs(c["contribution"]) / r["attribution_total"])
                top_rank += (i == 0)
                break
    present = len(shares)
    shares = np.array(shares) if shares else np.zeros(1)

    log(f"constant within the stay: {constant:,} of {per_stay.height:,} stays "
        f"({100*constant/per_stay.height:.1f}%)")
    log(f"total >= span already observed: {100*covers:.1f}% of stays with >5 readings")
    log(f"in the stored top-8 on {present:,} of {len(records):,} sampled readings "
        f"({100*present/max(len(records),1):.2f}%); mean share of |attribution| "
        f"{shares.mean():.3f}, max {shares.max():.3f}")
    if constant == per_stay.height:
        log("[red]vent_hours is a whole-stay aggregate: every reading knows how "
            "long the episode will last[/red]")
    return {
        "definition": "s01_cohort_strict.py:50 -- date_diff('hour', vent_start, vent_end)",
        "stays": per_stay.height,
        "constant_within_stay": constant,
        "share_total_ge_observed_span": covers,
        # The denominator, persisted. The top-8 figures below are shares of the
        # sample this run drew, not of the record file -- quoting them without
        # it is how a rate gets restated against the wrong base.
        "records_sampled": len(records),
        "in_top8_readings": present,
        "attribution_share_when_present_mean": float(shares.mean()),
        "attribution_share_when_present_max": float(shares.max()),
        "ranked_first_in_sampled_readings": top_rank,
        "verdict": ("Future information. features.json documents every feature as "
                    "at or before t; this one is the length of the completed "
                    "episode, known only once ventilation has ended."),
    }


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def _as_model_frame(df: pl.DataFrame, assets: ServingAssets):
    """Polars frame -> the exact pandas dtypes the booster was fitted against."""
    pdf = df.select(list(assets.feature_order)).to_pandas()
    for column, levels in assets.categorical.items():
        pdf[column] = pd.Categorical(pdf[column], categories=list(levels))
    for column in pdf.columns:
        if column not in assets.categorical:
            pdf[column] = pdf[column].astype(np.float32)
    return pdf


def _replay_bands(machine, matrix: pl.DataFrame, scores) -> list[str]:
    """One stepper per stay, in time order -- the only correct way to band."""
    out = []
    steppers: dict[int, B.BandStepper] = {}
    origins: dict[int, datetime] = {}
    for i, (stay_id, t) in enumerate(zip(matrix["stay_id"], matrix["charttime"])):
        stay_id = int(stay_id)
        if stay_id not in steppers:
            steppers[stay_id] = B.BandStepper(machine)
            origins[stay_id] = t
        minutes = (t - origins[stay_id]).total_seconds() / 60.0
        out.append(steppers[stay_id].push(float(scores[i]), minutes).displayed)
    return out


if __name__ == "__main__":
    main()
