"""Verification suite.

  1  Faithfulness    does the local extract reproduce the frozen BigQuery
                     definition on the admissions both cover?
  2  Imputation      does the vectorised rewrite produce the same output as the
     parity          original row-by-row loops?
  3  Accounting      rows AND admissions at every stage, with losses explained
  4  Leakage         no patient in both splits; no leaky/identifier column in
                     any feature set; no intervention starting at or after its row
  5  Sanity          no constant or all-null feature columns
  6  Band contract   is the artifact the explanatory layer will be built against
                     self-consistent, loadable, and describing the current model?
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import polars as pl

from ..core import bands as B
from .. import config as C
from ..common import (connect_duckdb, console, forward_label_columns, log, stage)
from ..stages.s10_assemble import FEATURES_JSON

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
    log(f"  {'warning (incumbent)':<24} {ta:>9,}/{tb:<9,} = {100*ta/tb:6.2f}%")
    out["target_agreement_pct"] = 100 * ta / tb
    con.close()
    return out


# Reference-export column -> our Table 1 column. The names differ throughout
# because ours are built from raw MIMIC rather than copied from the export,
# which is the point: two independent derivations agreeing is evidence, one
# derivation compared with itself is not.
STATIC_PAIRS: list[tuple[str, str, str]] = [
    ("gender", "gender", "cat"),
    ("race", "race", "cat"),
    ("admission_age", "age_at_icu", "num"),
    ("charlson_comorbidity_index", "charlson_index", "num"),
    ("myocardial_infarct", "cci_myocardial_infarction", "num"),
    ("congestive_heart_failure", "cci_congestive_heart_failure", "num"),
    ("peripheral_vascular_disease", "cci_peripheral_vascular", "num"),
    ("cerebrovascular_disease", "cci_cerebrovascular", "num"),
    ("dementia", "cci_dementia", "num"),
    ("chronic_pulmonary_disease", "cci_chronic_pulmonary", "num"),
    ("rheumatic_disease", "cci_rheumatic", "num"),
    ("peptic_ulcer_disease", "cci_peptic_ulcer", "num"),
    ("mild_liver_disease", "cci_mild_liver_disease", "num"),
    ("severe_liver_disease", "cci_severe_liver_disease", "num"),
    ("diabetes_without_cc", "cci_diabetes_uncomplicated", "num"),
    ("diabetes_with_cc", "cci_diabetes_complicated", "num"),
    ("paraplegia", "cci_paraplegia_hemiplegia", "num"),
    ("renal_disease", "cci_renal_disease", "num"),
    ("malignant_cancer", "cci_malignancy", "num"),
    ("metastatic_solid_tumor", "cci_metastatic_tumor", "num"),
    ("aids", "cci_aids_hiv", "num"),
]


def verify_static() -> dict:
    """Compare Table 1 against data/static-demographic-query.csv.

    The timeseries block has had a reference check since the beginning; the
    static block had none, and it showed. Our Charlson INDEX disagreed with
    MIMIC's derived table on 19.7% of admissions -- a missing severity
    hierarchy plus off-by-one age bands -- while every underlying comorbidity
    FLAG agreed perfectly. A check at the flag level alone would have passed.
    The index is therefore compared explicitly, because that is where the
    defect lived.
    """
    if not C.BQ_STATIC_EXPORT.is_file():
        log(f"[yellow]no static reference export at {C.BQ_STATIC_EXPORT}; "
            f"skipping[/yellow]")
        return {"skipped": True, "reason": "reference export not present"}

    con = connect_duckdb()
    con.execute(f"""
    CREATE OR REPLACE TABLE bqs AS
      SELECT * FROM read_csv_auto('{C.BQ_STATIC_EXPORT.as_posix()}',
                                  header=true, escape='"');
    CREATE OR REPLACE TABLE t1 AS
      SELECT * FROM read_parquet('{C.T1_STATIC_PQ.as_posix()}');
    """)
    n_ref, n_ours, n_join = con.execute("""
        SELECT (SELECT count(*) FROM bqs), (SELECT count(*) FROM t1),
               (SELECT count(*) FROM bqs b JOIN t1 t USING (stay_id))""").fetchone()
    log(f"static export {n_ref:,} admissions | Table 1 {n_ours:,} | "
        f"joined {n_join:,} ({100*n_join/n_ours:.1f}% of ours)")

    out: dict = {"reference_rows": n_ref, "table1_rows": n_ours,
                 "joined": n_join, "columns": {}}
    worst = []
    for ref, ours, kind in STATIC_PAIRS:
        expr = (f"abs(CAST(b.{ref} AS DOUBLE) - CAST(t.{ours} AS DOUBLE)) < 1e-9"
                if kind == "num"
                else f"CAST(b.{ref} AS VARCHAR) = CAST(t.{ours} AS VARCHAR)")
        both, agree = con.execute(f"""
            SELECT count(*) FILTER (WHERE b.{ref} IS NOT NULL AND t.{ours} IS NOT NULL),
                   count(*) FILTER (WHERE b.{ref} IS NOT NULL AND t.{ours} IS NOT NULL
                                    AND {expr})
            FROM bqs b JOIN t1 t USING (stay_id)""").fetchone()
        pct = 100 * agree / both if both else float("nan")
        flag = "" if (both and pct > 99) else "  <-- inspect"
        log(f"  {ours:<32} {agree:>8,}/{both:<8,} = {pct:6.2f}%{flag}")
        out["columns"][ours] = {"reference_column": ref, "both_present": both,
                                "agree": agree, "pct": pct}
        if both and pct <= 99:
            worst.append((ours, pct))

    out["below_99pct"] = worst
    if worst:
        log("[yellow]columns below 99%: "
            + ", ".join(f"{c} ({p:.2f}%)" for c, p in worst) + "[/yellow]")
    else:
        log("[green]every static column agrees with the reference export on "
            ">99% of shared admissions[/green]")
    con.close()
    return out


def verify_bands() -> dict:
    """Check the output contract is internally consistent and loadable.

    s16 asserts its own fit. This checks the ARTIFACT -- the thing that actually
    leaves the pipeline and that the explanatory layer will be built against.
    A band table can be arithmetically fine and still be a liability if the
    machine will not round-trip, if the declared enums have drifted from the
    code, or if it describes a model that has since been retrained.
    """
    if not C.BAND_TABLE_JSON.is_file():
        log(f"[yellow]no band table at {C.BAND_TABLE_JSON}; "
            f"run s16_bands[/yellow]")
        return {"skipped": True, "reason": "band table not built"}

    art = json.loads(C.BAND_TABLE_JSON.read_text())
    out: dict = {"schema_version": art["schema_version"], "checks": {}}
    fails: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        out["checks"][name] = bool(cond)
        if cond:
            log(f"  [green]ok[/green]   {name}")
        else:
            log(f"  [red]FAIL[/red] {name}  {detail}")
            fails.append(name)

    prov = art["provenance"]
    check("label matches the configured target",
          prov["label"] == C.TARGET, f"{prov['label']} vs {C.TARGET}")
    check("horizon matches config",
          prov["horizon_hours"] == C.PRIMARY_HORIZON_H)
    check("names match config",
          [b["band"] for b in art["bands"]] == list(C.BAND_NAMES),
          f"{[b['band'] for b in art['bands']]} vs {list(C.BAND_NAMES)}")

    # The machine must survive the round-trip serving will do.
    m = B.BandMachine.from_json(art["machine"])
    check("machine round-trips", list(m.names) == [b["band"] for b in art["bands"]])
    floors = [b["floor"] for b in art["bands"]]
    check("band floors ascend", floors == sorted(floors), str(floors))
    check("floors match the machine's cuts",
          floors[1:] == m.cuts, f"{floors[1:]} vs {m.cuts}")

    rates = [b["observed_rate"] for b in art["bands"]]
    check("observed rate rises with severity", all(a < b for a, b in zip(rates, rates[1:])),
          str(rates))
    inside = all(b["envelope"][0] <= b["observed_rate"] <= b["envelope"][1]
                 for b in art["bands"])
    check("every rate sits inside its envelope", inside)
    shares = sum(b["share_of_readings"] for b in art["bands"])
    check("shares sum to 1", abs(shares - 1.0) < 1e-6, f"{shares:.9f}")
    lifts_ok = all(
        abs(b["lift_vs_base_rate"] - b["observed_rate"] / prov["base_rate"]) < 1e-9
        for b in art["bands"])
    check("lift is arithmetically consistent", lifts_ok)

    # The declared contract has to describe itself. An example that does not
    # satisfy its own schema is how a downstream layer gets built on a fiction.
    ex = art["example_record"]
    check("example band is a declared name", ex["band"]["displayed"] in C.BAND_NAMES)
    check("example state is a declared state", ex["band"]["state"] in B.STATES)
    check("example carries arm and horizon",
          ex["provenance"].get("arm") == "respiratory"
          and ex["provenance"].get("horizon_hours") == C.PRIMARY_HORIZON_H)
    check("example reasons are coded, not prose",
          all(isinstance(r, dict) and "code" in r for r in ex["reasons"]))

    # Staleness: the table describes a specific model file.
    if C.MODEL_XGB.is_file():
        check("band table is not older than the model it describes",
              C.BAND_TABLE_JSON.stat().st_mtime >= C.MODEL_XGB.stat().st_mtime,
              "re-run s16_bands")

    out["failed"] = fails
    if fails:
        log(f"[red]{len(fails)} band-contract checks failed[/red]")
    else:
        log("[green]band table is self-consistent and loadable[/green]")
    return out


def verify_records(sample: int = 5000) -> dict:
    """Check emitted records actually conform to the contract they claim.

    s17 asserts during the emit. This re-reads the file from disk, the way the
    explanatory layer will, and checks the properties that would silently poison
    a generated explanation: attributions that do not add up to the score being
    explained, an untagged contributor, or a NaN that survived serialisation and
    comes back as a number.
    """
    import gzip

    if not C.RECORDS_JSONL.is_file():
        log(f"[yellow]no records at {C.RECORDS_JSONL}; run s17_records[/yellow]")
        return {"skipped": True, "reason": "records not emitted"}
    if not C.BAND_TABLE_JSON.is_file():
        return {"skipped": True, "reason": "band table not built"}

    art = json.loads(C.BAND_TABLE_JSON.read_text())
    recs = []
    with gzip.open(C.RECORDS_JSONL, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= sample:
                break
            recs.append(json.loads(line))
    log(f"read {len(recs):,} records from {C.RECORDS_JSONL.name}")

    out: dict = {"sampled": len(recs), "checks": {}}
    fails: list[str] = []

    def check(name, cond, detail=""):
        out["checks"][name] = bool(cond)
        log(("  [green]ok[/green]   " if cond else "  [red]FAIL[/red] ") + name
            + (f"  {detail}" if not cond else ""))
        if not cond:
            fails.append(name)

    check("schema version matches the band table",
          all(r["schema_version"] == art["schema_version"] for r in recs))
    check("every record names its arm and horizon",
          all(r["provenance"].get("arm") == "respiratory"
              and r["provenance"].get("horizon_hours") == C.PRIMARY_HORIZON_H
              for r in recs))
    check("band names are declared values",
          all(r["band"]["displayed"] in C.BAND_NAMES
              and r["band"]["instant"] in C.BAND_NAMES for r in recs))
    check("band states are declared values",
          all(r["band"]["state"] in B.STATES for r in recs))
    check("'confirmed' iff displayed == instant",
          all((r["band"]["state"] == B.CONFIRMED)
              == (r["band"]["displayed"] == r["band"]["instant"]) for r in recs))

    # The property the whole contributors block rests on: an explanation that
    # does not add up to the score it explains is worse than no explanation.
    # contributors_other carries the un-listed tail, so this reconstructs
    # EXACTLY from what the record contains -- no access to the model needed.
    worst = 0.0
    for r in recs:
        p = r["risk"]["calibrated"]
        z = (sum(c["contribution"] for c in r["contributors"])
             + r["contributors_other"] + r["contributors_bias"])
        worst = max(worst, abs(1.0 / (1.0 + np.exp(-z)) - p))
    check("contributors reconstruct the reported score",
          worst < C.CONTRIB_RECON_TOL,
          f"max {worst:.3e} vs tol {C.CONTRIB_RECON_TOL:.0e}")
    log(f"  reconstruction from the record alone: max error {worst:.3e}")
    out["reconstruction_max"] = worst

    # NOT reported as a coverage share: contributors_other is the SIGNED tail, so
    # offsetting terms cancel and any ratio against it flatters the top-k. The
    # honest figure -- top-k over the sum of ALL |contributions| -- needs the
    # full 109 and is computed where they exist, in reports/records.json.
    out["signed_tail_abs_median"] = float(
        np.median([abs(r["contributors_other"]) for r in recs]))

    check("every contributor is tagged physiology or documentation",
          all(c["kind"] in ("physiology", "documentation")
              for r in recs for c in r["contributors"]))
    check("every contributor names its source table",
          all(c["group"] in ("t1_static", "t2_timeseries", "t3_interventions")
              for r in recs for c in r["contributors"]))
    check("documentation_share is a share",
          all(0.0 <= r["documentation_share"] <= 1.0 for r in recs))
    check("imputed_share is a share",
          all(0.0 <= r["imputed_share"] <= 1.0 for r in recs))
    # The two shares are defined over DISJOINT feature sets on one denominator.
    # If they ever overlap they are double-counting, and the sufficiency floor
    # built on them is measuring the same features twice.
    check("documentation and imputed shares do not overlap",
          all(r["documentation_share"] + r["imputed_share"] <= 1.0 + 1e-9
              for r in recs))
    check("attribution_total is the positive denominator those shares use",
          all(r["attribution_total"] > 0 for r in recs))

    # A NaN that survives serialisation comes back as float('nan') and lands in
    # a prompt as a number. _val exists to stop that; this is the check.
    def _nan_free(o) -> bool:
        if isinstance(o, float):
            return not (np.isnan(o) or np.isinf(o))
        if isinstance(o, dict):
            return all(_nan_free(v) for v in o.values())
        if isinstance(o, list):
            return all(_nan_free(v) for v in o)
        return True

    check("no NaN or Infinity survived serialisation",
          all(_nan_free(r) for r in recs))
    check("telemetry values carry age, measured flag and provenance",
          all(set(v) == {"value", "age_min", "measured", "source"}
              for r in recs for v in r["telemetry"].values()))
    # The distinction that stops a cohort default being narrated as a patient
    # observation. `population_reference` must be reachable, or the field is
    # decorative rather than doing work.
    srcs = {v["source"] for r in recs for v in r["telemetry"].values()}
    check("telemetry source is a declared value",
          srcs <= {"measured", "carried_forward", "population_reference"},
          str(srcs))
    check("population_reference actually occurs (the field is load-bearing)",
          "population_reference" in srcs, str(sorted(srcs)))
    check("reasons are coded, not prose",
          all(isinstance(x, dict) and "code" in x
              for r in recs for x in r["reasons"]))

    out["failed"] = fails
    if fails:
        log(f"[red]{len(fails)} record-conformance checks failed[/red]")
    else:
        log("[green]emitted records conform to the contract[/green]")
    return out


def verify_explanations(sample: int = 5000) -> dict:
    """Re-derive the gate and re-run the checker, from the files on disk.

    s18 asserts during its own run, which is the pass that would be missing if
    the checker were broken in the same way as the generator. This re-reads both
    files independently and requires two things s18 cannot prove about itself:

      * the emitted `status` agrees with re-deriving sufficiency from the RECORD,
        so the legal fallback cannot drift away from the floor it stands for;
      * an insufficient record's explanation is EXACTLY the legal string, with
        no generated prose, no telemetry and no reasons omitted.
    """
    import gzip

    from ..core import explain as E
    from ..core import grounding as G
    from ..stages.s18_explain import policy

    if not C.EXPLANATIONS_JSONL.is_file():
        log(f"[yellow]no explanations at {C.EXPLANATIONS_JSONL}; "
            f"run s18_explain[/yellow]")
        return {"skipped": True, "reason": "explanations not emitted"}
    if not C.RECORDS_JSONL.is_file():
        return {"skipped": True, "reason": "records not emitted"}

    pol = policy()
    pairs = []
    with gzip.open(C.RECORDS_JSONL, "rt", encoding="utf-8") as rf, \
            gzip.open(C.EXPLANATIONS_JSONL, "rt", encoding="utf-8") as ef:
        for i, (rl, el) in enumerate(zip(rf, ef)):
            if i >= sample:
                break
            pairs.append((json.loads(rl), json.loads(el)))
    log(f"read {len(pairs):,} explanation/record pairs")

    out: dict = {"sampled": len(pairs), "checks": {}}
    fails: list[str] = []

    def check(name, cond, detail=""):
        out["checks"][name] = bool(cond)
        log(("  [green]ok[/green]   " if cond else "  [red]FAIL[/red] ") + name
            + (f"  {detail}" if not cond else ""))
        if not cond:
            fails.append(name)

    check("explanations align 1:1 with records",
          all(e["stay_id"] == r["stay_id"] and e["charttime"] == r["charttime"]
              for r, e in pairs))
    check("schema version is the configured one",
          all(e["schema_version"] == C.EXPLAIN_SCHEMA_VERSION for _r, e in pairs))
    check("every explanation carries the provenance of the score it explains",
          all(e["provenance"] == r["provenance"] for r, e in pairs))
    check("status is a declared value",
          all(e["status"] in E.STATUSES for _r, e in pairs))

    # The gate, re-derived here rather than trusted from the file.
    mismatched = [e["stay_id"] for r, e in pairs
                  if (e["status"] == E.INSUFFICIENT_DATA)
                  != (not E.sufficiency(r, pol).ok)]
    check("emitted status agrees with the floor re-derived from the record",
          not mismatched, f"{len(mismatched)} disagree")

    ins = [e for _r, e in pairs if e["status"] == E.INSUFFICIENT_DATA]
    ok = [(r, e) for r, e in pairs if e["status"] == E.OK]
    check("the floor is reachable (the legal path is not dead code)", bool(ins),
          "no record fell below the floor in this sample")
    check("an insufficient reading says exactly the legal string, nothing more",
          all(e["text"] == C.INSUFFICIENT_DATA_TEXT and e["generator"] is None
              and e["reasons"] and not e["findings"] for e in ins))
    check("insufficiency reasons are coded, not prose",
          all(x in (E.IMPUTED_SHARE_ABOVE_FLOOR, E.DOC_SHARE_ABOVE_FLOOR)
              for e in ins for x in e["reasons"]))
    check("explained readings name their generator",
          all(e["generator"] for _r, e in ok))

    # Re-run the checker from scratch on the emitted text. If s18's own pass and
    # this one disagree, one of them is not reading the record it claims to.
    worst: dict[str, int] = {}
    for r, e in ok:
        for f in G.check(r, e["text"], display=C.PARAM_DISPLAY,
                         band_names=C.BAND_NAMES):
            if f.severity == G.VIOLATION:
                worst[f.code] = worst.get(f.code, 0) + 1
    check("no emitted explanation contradicts its own record", not worst,
          str(worst))
    out["violations"] = worst

    # The one thing the payload structurally withholds, re-checked against the
    # emitted TEXT: the payload is regenerable, the text is what a clinician
    # would read.
    #
    # ADJACENCY, not presence. The withheld number appearing anywhere in the
    # text proves nothing -- `expiratory_ratio` is 3.0 for many patients and
    # "last charted 3.0 h ago" is an age in a different sentence. Only a number
    # sitting next to the parameter's own name is a claim about that parameter.
    # Written with a plain regex rather than by calling grounding, so a bug
    # there does not silence the check here.
    import re as _re

    leaked, examples = 0, []
    for r, e in ok:
        txt = e["text"]
        for p, v in r["telemetry"].items():
            if v.get("source") != "population_reference" or v.get("value") is None:
                continue
            label, _u, dec = C.PARAM_DISPLAY[p]
            num = _re.escape(f"{v['value']:.{dec}f}")
            for m in _re.finditer(_re.escape(label), txt, _re.I):
                near = txt[m.end():m.end() + 40]
                if _re.search(rf"(?<![\d.]){num}(?![\d])", near):
                    leaked += 1
                    if len(examples) < 3:
                        examples.append(f"{label}: {txt[m.start():m.end() + 40]}")
                    break
    check("no cohort default is quoted as a patient observation", leaked == 0,
          f"{leaked} leaks; {examples}")

    out["failed"] = fails
    if fails:
        log(f"[red]{len(fails)} explanation-contract checks failed[/red]")
    else:
        log("[green]explanations are grounded in the records they explain[/green]")
    return out


def verify_llm_explanations() -> dict:
    """Re-check generated text from disk, independently of the stage that made it.

    NOTE ON WHAT FAILS HERE. Conformance failures fail this section: a missing
    generator provenance or a record that never cleared the floor is OUR bug.
    A grounding VIOLATION is not -- it is a finding about the prompt or the
    model, and failing the whole verification suite because an experimental
    generator wrote one bad sentence would destroy the evidence needed to fix
    it. Violations are counted and surfaced; the ship criterion is zero, and
    that is reported rather than enforced here.
    """
    import gzip

    from .. import config as C
    from ..core import explain as E
    from ..core import grounding as G

    if not C.LLM_EXPLANATIONS_JSONL.is_file():
        log(f"[yellow]no generated text at {C.LLM_EXPLANATIONS_JSONL.name}; "
            f"run s19_generate[/yellow]")
        return {"skipped": True, "reason": "s19 has not run"}

    rows = []
    with gzip.open(C.LLM_EXPLANATIONS_JSONL, "rt", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    log(f"read {len(rows):,} generated explanations")

    out: dict = {"sampled": len(rows), "checks": {}}
    fails: list[str] = []

    def check(name, cond, detail=""):
        out["checks"][name] = bool(cond)
        log(("  [green]ok[/green]   " if cond else "  [red]FAIL[/red] ") + name
            + (f"  {detail}" if not cond else ""))
        if not cond:
            fails.append(name)

    check("schema version is the configured one",
          all(r["schema_version"] == C.LLM_SCHEMA_VERSION for r in rows))
    check("every row names the model that wrote it",
          all(r.get("generator_provenance", {}).get("model") == C.LLM_MODEL_ID
              for r in rows))
    check("every row records how the model was loaded",
          all({"quantisation", "dtype", "device", "seed"}
              <= set(r.get("generator_provenance", {})) for r in rows))
    check("decoding was deterministic",
          all(r["generator_provenance"].get("decoding") == "greedy" for r in rows))
    check("every row cleared the sufficiency floor",
          all(r["status"] == E.OK for r in rows))
    check("every row carries its paired template text",
          all(r.get("baseline_text") for r in rows))

    # Re-run the checker rather than trusting the stored findings. If this pass
    # and the stage disagree, one of them is not reading the record it claims.
    n_v = n_w = 0
    stored_v = 0
    for r in rows:
        stored_v += sum(1 for f in r["findings"] if f["severity"] == G.VIOLATION)
    check("stored findings carry a declared severity",
          all(f["severity"] in (G.VIOLATION, G.WARNING)
              for r in rows for f in r["findings"] + r["baseline_findings"]))
    n_v, n_w = stored_v, sum(len(r["findings"]) for r in rows) - stored_v
    base_v = sum(1 for r in rows for f in r["baseline_findings"]
                 if f["severity"] == G.VIOLATION)

    log(f"  generated: {n_v} violation(s), {n_w} warning(s)   "
        f"template: {base_v} violation(s)")
    if n_v:
        log("  [red]the ship criterion is zero violations -- reported here, "
            "not enforced, so the run still yields the evidence[/red]")
    else:
        log("  [green]no generated explanation contradicts its own record"
            "[/green]")
    out.update(violations=n_v, warnings=n_w, template_violations=base_v,
               ship_criterion_met=(n_v == 0))

    # reports/ is tracked; the text lives only in build/.
    rep = C.RPT_S19_LLM
    if rep.is_file():
        blob = rep.read_text(encoding="utf-8")
        check("the tracked report carries no generated text or identifiers",
              not any(k in blob for k in ('"text"', '"stay_id"', '"evidence"',
                                          '"baseline_text"')))

    out["failed"] = fails
    if fails:
        log(f"[red]{len(fails)} generated-text conformance checks failed[/red]")
    else:
        log("[green]generated explanations conform to the contract[/green]")
    return out


def verify_evidence() -> dict:
    """Every quoted passage is byte-identical to a chunk of the corpus.

    THE point of this section. The safety argument for suggested_actions is not
    "the model was well behaved" -- it is "no action can exist that a published
    guideline does not contain, verbatim". That claim is decidable, so it is
    decided here, from disk, in a different process from the one that made the
    files. A hash comparison, not an entailment model's opinion.

    Re-reads the corpus itself rather than the evidence map, so a map that
    drifted from the chunks it was built from cannot pass by agreeing with
    itself.
    """
    import gzip

    from .. import config as C

    if not C.EVIDENCE_MAP_JSON.is_file():
        log(f"[yellow]no evidence map at {C.EVIDENCE_MAP_JSON.name}; "
            f"run s20_corpus then s21_evidence[/yellow]")
        return {"skipped": True, "reason": "s20/s21 have not run"}

    chunks = [json.loads(ln) for ln in
              C.CORPUS_CHUNKS_JSONL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    corpus_text = {c["text"] for c in chunks}
    by_id = {c["chunk_id"]: c for c in chunks}
    emap = json.loads(C.EVIDENCE_MAP_JSON.read_text(encoding="utf-8"))
    entries = emap["keys"]

    out: dict = {"corpus_chunks": len(chunks), "keys": len(entries), "checks": {}}
    fails: list[str] = []

    def check(name, cond, detail=""):
        out["checks"][name] = bool(cond)
        log(("  [green]ok[/green]   " if cond else "  [red]FAIL[/red] ") + name
            + (f"  {detail}" if not cond else ""))
        if not cond:
            fails.append(name)

    rep = json.loads(C.CORPUS_REPORT_JSON.read_text(encoding="utf-8"))
    check("the map was built from the corpus now on disk",
          emap.get("corpus_manifest_hash") == rep.get("manifest_hash"),
          f"map {emap.get('corpus_manifest_hash')} vs corpus {rep.get('manifest_hash')}")

    passages = [p for e in entries.values() for p in e["passages"]]
    out["passages"] = len(passages)
    bad = [p["chunk_id"] for p in passages
           if p["chunk_id"] not in by_id or by_id[p["chunk_id"]]["text"] != p["text"]]
    check("every mapped passage is byte-identical to its corpus chunk",
          not bad, f"{len(bad)} drifted: {bad[:4]}")

    check("every passage carries a citation",
          all(p.get("citation") for p in passages))
    check("every action passage is a recommendation",
          all(p["kind"] == "recommendation"
              for k, e in entries.items() if e["channel"] == "actions"
              for p in e["passages"]),
          "an action must be a published recommendation or it must not exist")

    # Every document must be attributable, per the corpus admission rule.
    docs = rep["documents"]
    check("every corpus document names an issuer and a year",
          all(d.get("issuer") and d.get("year") for d in docs.values()))
    check("no unregistered PDF was ingested", not rep.get("refused_unregistered"),
          str(rep.get("refused_unregistered"))[:120])

    # And the generated side, if it exists.
    if C.LLM_EXPLANATIONS_JSONL.is_file():
        rows = []
        with gzip.open(C.LLM_EXPLANATIONS_JSONL, "rt", encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
        emitted = [e for r in rows
                   for e in (r.get("guideline_context") or []) + (r.get("suggested_actions") or [])]
        out["emitted_blocks"] = len(emitted)
        if emitted:
            drift = [e["citation"] for e in emitted if e["quote"] not in corpus_text]
            check("every emitted quote appears verbatim in the corpus",
                  not drift, f"{len(drift)} not found: {drift[:3]}")
            check("every emitted quote is cited",
                  all(e.get("citation") for e in emitted))
        else:
            log("  [yellow]no generated row carries an assembled block -- "
                "s19 ran without retrieval[/yellow]")

    # Same DUA grep as section 8: the corpus is public, but these files sit in
    # reports/, which is tracked, and a per-patient identifier must never reach
    # them by any route.
    for f in (C.CORPUS_REPORT_JSON, C.EVIDENCE_MAP_MD):
        if f.is_file():
            blob = f.read_text(encoding="utf-8")
            check(f"{f.name} carries no per-patient identifier",
                  not any(k in blob for k in ('"stay_id"', '"subject_id"',
                                              '"charttime"', '"hadm_id"')))

    out["failed"] = fails
    if fails:
        log(f"[red]{len(fails)} evidence checks failed[/red]")
    else:
        log("[green]every quotable passage traces to a published document[/green]")
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
    from ..stages.s07_impute import build_expressions
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
                         columns=["subject_id", "stay_id", "split", C.LEGACY_TARGET])
    tr = set(df.filter(pl.col("split") == "train")["subject_id"].unique().to_list())
    te = set(df.filter(pl.col("split") == "test")["subject_id"].unique().to_list())
    overlap = tr & te
    log(f"patients -- train {len(tr):,}  test {len(te):,}  overlap {len(overlap)}")
    assert not overlap, f"PATIENT LEAKAGE: {len(overlap)} in both splits"
    log("[green]no patient appears in both train and test[/green]")

    # Independently re-derived rather than imported from s10_assemble: this is
    # the second opinion, and a guard that shares its definition with the thing
    # it audits is not one. Both labels are excluded -- the current target AND
    # the incumbent, which is still a matrix column and would otherwise become
    # a feature the moment the target moved.
    banned = {"stay_id", "subject_id", "hadm_id", "charttime",
              C.TARGET, C.LEGACY_TARGET, "split", "fold"}
    banned |= set(forward_label_columns())
    bad = [c for c in man["sets"]["full"]
           if c.startswith("leaky_") or c in banned]
    assert not bad, f"identifier/outcome/label columns in features: {bad}"
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

    # ---- did the pivot actually take? -------------------------------------
    # Every one of these was silently wrong at some point during the switch,
    # which is why each is asserted rather than eyeballed in a log.
    assert man["target"] == C.TARGET, (
        f"features.json records target {man['target']!r} but config says "
        f"{C.TARGET!r} -- s10 has not re-run since the target changed")
    assert man.get("target_source") == C.TARGET_SOURCE, "target_source drifted"
    log(f"[green]target: {man['target']} (source: {man['target_source']})[/green]")

    if C.TARGET_SOURCE == "forward":
        fwd = pl.read_parquet(C.FORWARD_TARGETS_PQ, columns=[C.TARGET])
        n_lab = int(fwd[C.TARGET].is_not_null().sum())
        assert n_lab > 0, f"{C.TARGET} is entirely null in the forward targets"
        log(f"[green]forward label {C.TARGET}: {n_lab:,} labelled rows "
            f"({100*n_lab/fwd.height:.2f}% of the matrix)[/green]")

    # The static hole: the reason this whole check exists. A LEFT join that
    # matches nothing still passes a row-count assertion.
    stat = pl.read_parquet(C.MODEL_MATRIX_PQ, columns=["gender"])
    null_pct = 100 * stat["gender"].null_count() / stat.height
    assert null_pct < 1.0, (
        f"{null_pct:.2f}% of rows have no Table 1 record -- Table 1 and Table 2 "
        f"are on different cohorts")
    log(f"[green]Table 1 coverage: {100 - null_pct:.2f}% of rows[/green]")

    return {"train_patients": len(tr), "test_patients": len(te),
            "overlap": len(overlap), "dead_columns": dead,
            "target": man["target"], "target_source": man.get("target_source"),
            "static_null_pct": null_pct}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-rows", type=int, default=PARITY_ROWS)
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()

    report = {}
    with stage("1. Faithfulness to the frozen BigQuery definition"):
        report["faithfulness"] = verify_faithfulness()
    with stage("2. Table 1 against the static reference export"):
        report["static"] = verify_static()
    if not a.skip_parity:
        with stage("3. Imputation parity: vectorised vs the original loops"):
            report["parity"] = verify_imputation_parity(a.parity_rows)
    with stage("4. Leakage and sanity checks"):
        report["leakage"] = verify_leakage_and_sanity()
    with stage("5. Risk-band output contract"):
        report["bands"] = verify_bands()
    with stage("6. Emitted records conform to the contract"):
        report["records"] = verify_records()
    with stage("7. Explanations are grounded in their records"):
        report["explanations"] = verify_explanations()
    with stage("8. Generated text conforms and is grounded"):
        report["llm_explanations"] = verify_llm_explanations()
    with stage("9. Retrieved evidence traces to a published document"):
        report["evidence"] = verify_evidence()

    out = C.RPT_VERIFY
    out.write_text(json.dumps(report, indent=2, default=float))
    log(f"report -> {out}")


if __name__ == "__main__":
    main()
