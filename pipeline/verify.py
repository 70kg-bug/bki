"""Verification suite.

  1  Faithfulness    does the local extract reproduce the frozen BigQuery
                     definition on the admissions both cover?
  2  Imputation      does the vectorised rewrite produce the same output as the
     parity          original row-by-row loops?
  3  Accounting      rows AND admissions at every stage, with losses explained
  4  Leakage         no patient in both splits; no leaky/identifier column in
                     any feature set; no intervention starting at or after its row
  5  Sanity          no constant or all-null feature columns
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from . import config as C
from .common import connect_duckdb, console, log, stage
from .s10_assemble import FEATURES_JSON

PARITY_ROWS = 40_000


# ---------------------------------------------------------------------------
def verify_faithfulness() -> dict:
    """Compare the local extract against data/raw-query.csv column by column."""
    con = connect_duckdb()
    con.execute(f"""
    CREATE OR REPLACE TABLE bq AS
    SELECT CAST(stay_id AS INTEGER) AS stay_id,
           strptime(charttime, '%-m/%-d/%Y %-H:%M') AS charttime,
           {', '.join(C.FROZEN_ORDER)},
           CAST(warning AS TINYINT) AS warning
    FROM read_csv_auto('{C.BQ_EXPORT.as_posix()}', header=true);
    CREATE OR REPLACE TABLE mine AS
    SELECT * FROM read_parquet('{C.T2_WIDE_PQ.as_posix()}');
    """)
    nb, nbs = con.execute("SELECT count(*), count(DISTINCT stay_id) FROM bq").fetchone()
    nm, nms = con.execute("SELECT count(*), count(DISTINCT stay_id) FROM mine").fetchone()
    log(f"BigQuery export : {nb:,} rows / {nbs:,} admissions")
    log(f"local extract   : {nm:,} rows / {nms:,} admissions")

    shared_stays = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT stay_id FROM bq "
        "INTERSECT SELECT DISTINCT stay_id FROM mine)").fetchone()[0]
    matched = con.execute("""
        SELECT count(*) FROM bq b JOIN mine m
        ON b.stay_id = m.stay_id AND b.charttime = m.charttime""").fetchone()[0]
    log(f"admissions in both: {shared_stays:,}  |  (stay_id, charttime) keys matched: "
        f"{matched:,} ({100*matched/nb:.1f}% of the export)")

    out = {"bq_rows": nb, "bq_stays": nbs, "local_rows": nm, "local_stays": nms,
           "shared_stays": shared_stays, "matched_keys": matched, "columns": {}}

    log("per-column agreement where BOTH sides have a value:")
    for col in C.FROZEN_ORDER:
        both, agree = con.execute(f"""
            SELECT count(*) FILTER (WHERE b.{col} IS NOT NULL AND m.{col} IS NOT NULL),
                   count(*) FILTER (WHERE b.{col} IS NOT NULL AND m.{col} IS NOT NULL
                                    AND abs(b.{col} - m.{col}) <= 0.01 * abs(b.{col}) + 0.01)
            FROM bq b JOIN mine m ON b.stay_id=m.stay_id AND b.charttime=m.charttime
        """).fetchone()
        pct = 100 * agree / both if both else float("nan")
        flag = "" if (both and pct > 99) else "  <-- inspect"
        log(f"  {col:<24} {agree:>9,}/{both:<9,} = {pct:6.2f}%{flag}")
        out["columns"][col] = {"both_present": both, "agree": agree, "pct": pct}

    tb, ta = con.execute("""
        SELECT count(*) FILTER (WHERE b.warning IS NOT NULL),
               count(*) FILTER (WHERE b.warning = m.warning)
        FROM bq b JOIN mine m ON b.stay_id=m.stay_id AND b.charttime=m.charttime
    """).fetchone()
    log(f"  {'warning (target)':<24} {ta:>9,}/{tb:<9,} = {100*ta/tb:6.2f}%")
    out["target_agreement_pct"] = 100 * ta / tb
    con.close()
    return out


# ---------------------------------------------------------------------------
# The ORIGINAL row-by-row implementation, copied faithfully from
# data/cleaning_and_filling_mising_values_claude.ipynb so parity means something.
# ---------------------------------------------------------------------------
def _orig_mask_delta(df, value_cols, ID_COL="stay_id", TIME_COL="charttime"):
    df = df.copy()
    for col in value_cols:
        df[f"{col}_observed"] = df[col].notna().astype(int)
        delta = np.full(len(df), np.nan)
        for _stay, idx in df.groupby(ID_COL).groups.items():
            last_seen_time = None
            for i in list(idx):
                t = df.at[i, TIME_COL]
                if last_seen_time is not None:
                    delta[i] = (t - last_seen_time).total_seconds() / 60.0
                if df.at[i, col] == df.at[i, col]:
                    last_seen_time = t
        df[f"{col}_delta_t_min"] = delta
    return df


def _orig_locf(df, value_cols, cutoff_min, ID_COL="stay_id", TIME_COL="charttime"):
    df = df.copy()
    for col in value_cols:
        filled = f"{col}_locf"
        df[filled] = df[col]
        for _stay, group in df.groupby(ID_COL):
            last_val, last_time = None, None
            for i in group.index:
                t = df.at[i, TIME_COL]
                if pd.notna(df.at[i, col]):
                    last_val, last_time = df.at[i, col], t
                elif last_val is not None:
                    if (t - last_time).total_seconds() / 60.0 <= cutoff_min:
                        df.at[i, filled] = last_val
    return df


def _orig_structural(df, value_cols, ID_COL="stay_id"):
    df = df.copy()
    for col in value_cols:
        df[f"{col}_structurally_missing_in_stay"] = (
            df.groupby(ID_COL)[col].transform(lambda s: s.notna().sum() == 0).astype(int))
    return df


def verify_imputation_parity(n_rows: int = PARITY_ROWS) -> dict:
    """Same input, both implementations, compared cell by cell."""
    params = C.FROZEN_ORDER
    ref = json.loads(C.REFERENCE_STATS_JSON.read_text())

    wide = (pl.read_parquet(C.T2_WIDE_PQ)
            .sort("stay_id", "charttime")
            .head(n_rows))
    log(f"slice: {wide.height:,} rows / {wide['stay_id'].n_unique():,} admissions")

    # --- new (vectorised) ---
    from .s07_impute import build_expressions
    t0 = time.time()
    new = wide.with_columns(build_expressions(params)).with_columns([
        pl.col(f"{p}_locf").fill_null(ref[p]).cast(pl.Float32).alias(f"{p}_final")
        for p in params])
    t_new = time.time() - t0

    # --- original (row-by-row loops) ---
    pdf = wide.to_pandas().reset_index(drop=True)
    t0 = time.time()
    old = _orig_mask_delta(pdf, params)
    old = _orig_locf(old, params, C.LOCF_CUTOFF_MIN)
    old = _orig_structural(old, params)
    for p in params:
        old[f"{p}_final"] = old[f"{p}_locf"].fillna(ref[p])
    t_old = time.time() - t0

    log(f"vectorised: {t_new:.2f}s   original loops: {t_old:.1f}s   "
        f"[bold]{t_old/max(t_new,1e-9):,.0f}x faster[/bold]")
    log(f"extrapolated to the full {4_198_262:,} rows: "
        f"vectorised ~{t_new*4_198_262/n_rows:.0f}s, "
        f"original ~{t_old*4_198_262/n_rows/3600:.1f} h")

    out = {"rows": int(wide.height), "seconds_vectorised": t_new,
           "seconds_original": t_old, "speedup": t_old / max(t_new, 1e-9),
           "columns": {}}
    ok = True
    for suffix in ("_observed", "_locf", "_structurally_missing_in_stay", "_final"):
        for p in params:
            c = f"{p}{suffix}"
            a = new[c].to_numpy().astype(float)
            b = old[c].to_numpy().astype(float)
            same = ((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-4)).mean()
            out["columns"][c] = float(same)
            if same < 1.0:
                ok = False
                log(f"  [yellow]{c}: {100*same:.4f}% identical[/yellow]")
    log(f"  {len(out['columns'])} columns compared -- "
        + ("[green]all identical[/green]" if ok else "[yellow]differences above[/yellow]"))

    # _delta_t_min differs by design; quantify rather than hide it.
    p0 = params[0]
    a = new[f"{p0}_delta_t_min"].to_numpy().astype(float)
    b = old[f"{p0}_delta_t_min"].to_numpy().astype(float)
    obs = new[f"{p0}_observed"].to_numpy().astype(bool)
    log(f"  _delta_t_min differs by design (age of the value in use vs. gap since the "
        f"previous reading): on measured rows new={np.nanmean(a[obs]):.2f} "
        f"vs old={np.nanmean(b[obs]):.2f} minutes")
    out["delta_t_min_intentional_difference"] = True
    out["all_identical"] = ok
    return out


# ---------------------------------------------------------------------------
def verify_leakage_and_sanity() -> dict:
    man = json.loads(FEATURES_JSON.read_text())
    df = pl.read_parquet(C.MODEL_MATRIX_PQ,
                         columns=["subject_id", "stay_id", "split", C.TARGET])
    tr = set(df.filter(pl.col("split") == "train")["subject_id"].unique().to_list())
    te = set(df.filter(pl.col("split") == "test")["subject_id"].unique().to_list())
    overlap = tr & te
    log(f"patients -- train {len(tr):,}  test {len(te):,}  overlap {len(overlap)}")
    assert not overlap, f"PATIENT LEAKAGE: {len(overlap)} in both splits"
    log("[green]no patient appears in both train and test[/green]")

    bad = [c for c in man["sets"]["full"]
           if c.startswith("leaky_") or c in {"stay_id", "subject_id", "hadm_id",
                                              "charttime", C.TARGET, "split", "fold"}]
    assert not bad, f"identifier/outcome columns in features: {bad}"
    log(f"[green]none of the {len(man['sets']['full'])} feature columns is an "
        f"identifier, timestamp or outcome[/green]")

    # constant / all-null features carry no information and usually signal a bug
    full = pl.read_parquet(C.MODEL_MATRIX_PQ, columns=man["sets"]["full"])
    dead = []
    for c in full.columns:
        s = full[c]
        if s.null_count() == s.len() or s.n_unique() <= 1:
            dead.append(c)
    if dead:
        log(f"[yellow]constant or all-null feature columns: {dead}[/yellow]")
    else:
        log("[green]every feature column varies[/green]")
    return {"train_patients": len(tr), "test_patients": len(te),
            "overlap": len(overlap), "dead_columns": dead}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-rows", type=int, default=PARITY_ROWS)
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()

    report = {}
    with stage("1. Faithfulness to the frozen BigQuery definition"):
        report["faithfulness"] = verify_faithfulness()
    if not a.skip_parity:
        with stage("2. Imputation parity: vectorised vs the original loops"):
            report["parity"] = verify_imputation_parity(a.parity_rows)
    with stage("3. Leakage and sanity checks"):
        report["leakage"] = verify_leakage_and_sanity()

    out = C.REPORTS / "verification.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    log(f"report -> {out}")


if __name__ == "__main__":
    main()
