"""Replay real stays through `StayFeatures.push()` and diff all 109 columns.

This exists because `serving_parity` sections A and B did not: between them they
exercised 55 columns and none of the assembly, which is how a wrong
`ventilator_mode` survived every green check.

The inputs come from the real sources rather than by inverting the model matrix,
which would only re-derive what it is meant to check:

    readings      raw parameter columns, which s07 keeps in the matrix
    infusions     icu/inputevents, the same intervals s08 joins
    mode events   the chartevents cache, itemid 223849
    ICD codes     hosp/diagnoses_icd, which drives all twenty Charlson columns
    context       t1_static plus cohort.vent_start

Exactly twelve columns may differ: `{p}_structurally_missing_in_stay` (11) and
`vent_hours` read the future. Anything else is a defect.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from .. import config as C
from ..common import connect_duckdb, console, log
from ..core.features import (Infusion, PatientContext, Reading, ServingAssets,
                             StayFeatures)

STAYS = 120
TOLERANCE = 1e-3

# s08_table3_interventions.py:45.
ITEM_VENT_MODE = 223849

# Divergence that is designed in, not a defect. Everything else must match.
CAUSAL = {"vent_hours"}


def _drug_groups() -> dict[int, str]:
    """itemid -> group, resolved from d_items the way s08 does."""
    from ..stages.s08_table3_interventions import DRUG_GROUPS  # noqa: PLC0415
    con = connect_duckdb()
    con.execute(f"""CREATE OR REPLACE TABLE d_items AS
                    SELECT * FROM read_csv_auto('{C.D_ITEMS.as_posix()}', header=true)""")
    mapping = {}
    for group, pattern in DRUG_GROUPS.items():
        rows = con.execute(f"""
            SELECT itemid FROM d_items
            WHERE linksto = 'inputevents' AND regexp_matches(lower(label), '{pattern}')
        """).fetchall()
        mapping.update({r[0]: group for r in rows})
    con.close()
    return mapping


def _infusions(stay_ids, groups: dict[int, str]) -> dict[int, list[Infusion]]:
    ids = ", ".join(str(s) for s in stay_ids)
    con = connect_duckdb()
    rows = con.execute(f"""
        SELECT stay_id, itemid, starttime, endtime, COALESCE(rate, 0) AS rate
        FROM read_csv_auto('{C.INPUTEVENTS.as_posix()}', header=true)
        WHERE stay_id IN ({ids}) AND starttime IS NOT NULL
          AND statusdescription IS DISTINCT FROM 'Rewritten'
    """).fetchall()
    con.close()
    out: dict[int, list[Infusion]] = {s: [] for s in stay_ids}
    for stay_id, itemid, start, end, rate in rows:
        group = groups.get(itemid)
        if group:
            out[stay_id].append(Infusion(group, start, end, float(rate)))
    return out


def _mode_events(stay_ids) -> dict[int, list[tuple[datetime, str]]]:
    frame = (pl.read_parquet(C.TS_LONG_PQ, columns=["stay_id", "charttime", "itemid", "value"])
               .filter((pl.col("itemid") == ITEM_VENT_MODE)
                       & pl.col("stay_id").is_in(list(stay_ids))
                       & pl.col("value").is_not_null())
               .sort("stay_id", "charttime"))
    out: dict[int, list[tuple[datetime, str]]] = {s: [] for s in stay_ids}
    for row in frame.iter_rows(named=True):
        out[row["stay_id"]].append((row["charttime"], row["value"]))
    return out


def _icd_codes(hadm_ids) -> dict[int, tuple]:
    ids = ", ".join(str(h) for h in hadm_ids)
    con = connect_duckdb()
    rows = con.execute(f"""
        SELECT hadm_id, icd_code, icd_version
        FROM read_csv_auto('{C.DIAGNOSES_ICD.as_posix()}', header=true)
        WHERE hadm_id IN ({ids})
    """).fetchall()
    con.close()
    out: dict[int, list] = {h: [] for h in hadm_ids}
    for hadm_id, code, version in rows:
        out[hadm_id].append((str(code), int(version)))
    return {h: tuple(v) for h, v in out.items()}


def _context(static: dict, vent_start: datetime, icd: tuple) -> PatientContext:
    return PatientContext(
        sex=static["gender"], age_at_icu=static["age_at_icu"],
        admission_type=static["admission_type"],
        admission_location=static["admission_location"],
        insurance=static["insurance"], language=static["language"],
        marital_status=static["marital_status"], race=static["race"],
        first_careunit=static["first_careunit"],
        ventilation_start=vent_start,
        height_cm=static["height_cm"], weight_kg=static["weight_kg"],
        hours_admit_to_icu=static["hours_admit_to_icu"],
        ed_minutes=static["ed_minutes"],
        prior_icu_stays=int(static["prior_icu_stays"] or 0),
        icd_codes=icd)


def run(assets: ServingAssets, stays: int = STAYS) -> dict:
    """Diff every column push() produces against the model matrix."""
    cohort = pl.read_parquet(C.COHORT_PQ, columns=["stay_id", "hadm_id", "vent_start"])
    stay_ids = cohort["stay_id"].head(stays).to_list()
    cohort = cohort.filter(pl.col("stay_id").is_in(stay_ids))
    hadm = dict(zip(cohort["stay_id"], cohort["hadm_id"]))
    vent_start = dict(zip(cohort["stay_id"], cohort["vent_start"]))

    log("loading the real sources for these stays")
    groups = _drug_groups()
    infusions = _infusions(stay_ids, groups)
    modes = _mode_events(stay_ids)
    icd = _icd_codes(set(hadm.values()))
    log(f"  {sum(len(v) for v in infusions.values()):,} infusion intervals, "
        f"{sum(len(v) for v in modes.values()):,} mode events, "
        f"{sum(len(v) for v in icd.values()):,} diagnosis codes")

    columns = list(assets.feature_order)
    raw = list(assets.frozen_params)
    wanted = list(dict.fromkeys(["stay_id", "charttime"] + raw + columns))
    matrix = (pl.read_parquet(C.MODEL_MATRIX_PQ, columns=wanted)
                .filter(pl.col("stay_id").is_in(stay_ids))
                .sort("stay_id", "charttime"))

    disagree = {c: 0 for c in columns}
    compared = 0

    for stay_id in stay_ids:
        rows = matrix.filter(pl.col("stay_id") == stay_id)
        if rows.is_empty():
            continue
        static = rows.row(0, named=True)
        state = StayFeatures(
            _context(static, vent_start[stay_id], icd[hadm[stay_id]]), assets)

        pending = list(modes[stay_id])
        for row in rows.iter_rows(named=True):
            now = row["charttime"]
            # Both delivery paths, as the stream would: a mode charted with
            # the reading rides on it, one charted between readings arrives
            # separately and keeps its own timestamp.
            mode = None
            while pending and pending[0][0] <= now:
                at, value = pending.pop(0)
                if at == now:
                    mode = value
                else:
                    state.observe_mode(value, at)
            values = {p: row[p] for p in raw if row[p] is not None}
            frame = state.push(Reading(observed_at=now, values=values,
                                       ventilator_mode=mode,
                                       infusions=tuple(infusions[stay_id])))
            compared += 1
            produced = frame.iloc[0]
            for column in columns:
                want, have = row[column], produced[column]
                if column in assets.categorical:
                    have = None if (have is None or (isinstance(have, float))) else str(have)
                    if (want or None) != (have or None):
                        disagree[column] += 1
                    continue
                want_missing = want is None
                have_missing = have is None or have != have          # NaN
                if want_missing and have_missing:
                    continue
                if want_missing != have_missing or abs(float(want) - float(have)) > TOLERANCE:
                    disagree[column] += 1

    bad = {c: v for c, v in disagree.items() if v}
    expected = {c for c in bad
                if c.endswith("_structurally_missing_in_stay")
                or c in CAUSAL}
    unexpected = {c: v for c, v in bad.items() if c not in expected}

    log(f"replayed {compared:,} readings x {len(columns)} columns through push()")
    for column, count in sorted(bad.items(), key=lambda kv: -kv[1])[:10]:
        tag = ("[yellow]by design[/yellow]" if column in expected
               else "[red]UNEXPECTED[/red]")
        log(f"  {column:<52} {count:>6,} ({100*count/compared:5.2f}%)  {tag}")
    if unexpected:
        log(f"[red]{len(unexpected)} columns disagree that should not[/red]")
    else:
        log("[green]all 109 columns reproduce, except the twelve that cannot[/green]")

    return {"stays": len(stay_ids), "readings": compared,
            "columns": len(columns),
            "unexpected": {c: v for c, v in sorted(unexpected.items())},
            "by_design": {c: v for c, v in sorted(bad.items()) if c in expected},
            "ok": not unexpected}


def main() -> None:
    console.rule("[bold cyan]Full push() parity")
    assets = ServingAssets.load(C.MODELS / "serving_assets.json")
    result = run(assets)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
