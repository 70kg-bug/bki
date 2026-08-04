"""Stage 14 -- forward-looking target D (composite deterioration), at row grain.

`explore_targets.py` measured D's *prevalence* on hourly bins. That was enough to
rank candidate targets; it is not a trainable label. This builds D per
(stay_id, charttime) -- the grain the model matrix actually uses -- so a model can
be fitted on it and compared against `warning` on identical rows.

D is positive when, within (t, t+H], ANY of:

    FiO2 escalation    max(measured fio2) - fio2_locf >= 20   clinician action
    PEEP escalation    max(measured peep) - peep_locf >= 3    clinician action
    Desaturation       min(measured spo2) < 88                PATIENT STATE
    New vasopressor    vasopressor_running goes 0 -> 1        clinician action

Baselines are the LOCF value -- the setting currently in force, what a clinician
sees. Forward values must be MEASURED: an event has to be observed to count, so
an imputed value can never manufacture one.

CENSORING. A forward label is not always observable. A row is:
    positive  if the event is seen inside the window
    negative  only if the window is fully covered by the stay
    NULL      if the window is truncated AND nothing was seen -- genuinely unknown

Marking those NULL rows negative would teach the model that "the stay ended" means
"no deterioration", which is wrong and worst exactly where deterioration matters.
Dropping every truncated row instead would cost 6.95% of rows but 18.1% of
ADMISSIONS -- every stay shorter than H disappears -- and distinct admissions is
the quantity this whole pipeline optimises. Keeping truncated positives discards
only the genuinely unobservable.

The forward window starts at 1 SECOND FOLLOWING, so a row can never be labelled by
an event at or before its own timestamp. That is the reverse-causation guard, built
into the frame rather than asserted afterwards.
"""
from __future__ import annotations

import json

from . import config as C
from .common import (account_parquet, cached_stage, connect_duckdb, console,
                     heartbeat, log)

SUMMARY_JSON = C.REPORTS / "forward_targets.json"


def _window(h: int) -> str:
    return (f"(PARTITION BY stay_id ORDER BY charttime "
            f"RANGE BETWEEN INTERVAL 1 SECOND FOLLOWING "
            f"AND INTERVAL {h} HOUR FOLLOWING)")


def _components(h: int) -> str:
    """Component flags for one horizon. COALESCE to false: a missing baseline or
    an unmeasured forward value means the component cannot fire, not NULL.

    The `_s` (strict) variants additionally require the baseline to be FRESH --
    measured within STRICT_BASELINE_AGE_MIN. An escalation judged against a
    four-hour-old setting may be gradual drift rather than an acute change. Strict
    is the robustness check; it costs eligibility, and trading eligibility for
    freshness re-admits a little of the documentation confound, so both are built
    and both are reported.
    """
    fresh_f = f"fio2_delta_t_min <= {C.STRICT_BASELINE_AGE_MIN}"
    fresh_p = f"peep_delta_t_min <= {C.STRICT_BASELINE_AGE_MIN}"
    return f"""
        COALESCE(fio2_a{h} - fio2_locf >= {C.D_FIO2_RISE}, false) AS d_fio2_{h}h,
        COALESCE(peep_a{h} - peep_locf >= {C.D_PEEP_RISE}, false) AS d_peep_{h}h,
        COALESCE(spo2_a{h} < {C.D_SPO2_BELOW}, false)             AS d_spo2_{h}h,
        COALESCE(vaso_a{h} = 1 AND vaso_now = 0, false)           AS d_vaso_{h}h,
        COALESCE(fio2_a{h} - fio2_locf >= {C.D_FIO2_RISE} AND {fresh_f}, false)
                                                                  AS s_fio2_{h}h,
        COALESCE(peep_a{h} - peep_locf >= {C.D_PEEP_RISE} AND {fresh_p}, false)
                                                                  AS s_peep_{h}h"""


def main(force: bool = False) -> None:
    sources = [C.T2_IMPUTED_PQ, C.T3_INTERV_PQ]
    with cached_stage("s14_forward_targets", sources=sources,
                      output=C.FORWARD_TARGETS_PQ, force=force) as ran:
        if not ran:
            return
        _build()
    account_parquet("forward targets", C.FORWARD_TARGETS_PQ, subject_col=None)
    _report()


def _build() -> None:
    con = connect_duckdb()

    fwd_cols = ",\n".join(
        f"""       min(spo2) OVER w{h} AS spo2_a{h},
       max(fio2) OVER w{h} AS fio2_a{h},
       max(peep) OVER w{h} AS peep_a{h},
       max(vaso_now) OVER w{h} AS vaso_a{h}""" for h in C.HORIZONS_H)
    windows = ",\n".join(f"       w{h} AS {_window(h)}" for h in C.HORIZONS_H)
    comps = ",".join(_components(h) for h in C.HORIZONS_H)
    flag_cols = ",\n".join(
        "             " + ", ".join(f"{p}_{h}h" for p in
                                    ("d_fio2", "d_peep", "d_spo2", "d_vaso",
                                     "s_fio2", "s_peep"))
        for h in C.HORIZONS_H)

    # A row is positive if any component fired; negative only when the whole
    # window fits inside the stay; otherwise unknown and left NULL.
    labels = ",\n".join(f"""
        CASE WHEN d_fio2_{h}h OR d_peep_{h}h OR d_spo2_{h}h OR d_vaso_{h}h THEN 1
             WHEN charttime + INTERVAL {h} HOUR <= t_end THEN 0
             ELSE NULL END::TINYINT AS y_composite_{h}h,
        CASE WHEN s_fio2_{h}h OR s_peep_{h}h OR d_spo2_{h}h OR d_vaso_{h}h THEN 1
             WHEN charttime + INTERVAL {h} HOUR <= t_end THEN 0
             ELSE NULL END::TINYINT AS y_strict_{h}h,
        (charttime + INTERVAL {h} HOUR <= t_end) AS complete_{h}h"""
        for h in C.HORIZONS_H)

    sql = f"""
    COPY (
      WITH base AS (
        SELECT t.stay_id, t.charttime,
               t.spo2, t.fio2, t.peep,
               t.fio2_locf, t.peep_locf,
               t.fio2_delta_t_min, t.peep_delta_t_min,
               COALESCE(i.vasopressor_running, 0) AS vaso_now
        FROM read_parquet('{C.T2_IMPUTED_PQ.as_posix()}') t
        LEFT JOIN read_parquet('{C.T3_INTERV_PQ.as_posix()}') i
               ON i.stay_id = t.stay_id AND i.charttime = t.charttime
      ),
      fwd AS (
        SELECT stay_id, charttime, fio2_locf, peep_locf, vaso_now,
               fio2_delta_t_min, peep_delta_t_min,
               max(charttime) OVER (PARTITION BY stay_id) AS t_end,
{fwd_cols}
        FROM base
        WINDOW
{windows}
      ),
      flagged AS (
        SELECT stay_id, charttime, t_end, {comps}
        FROM fwd
      )
      SELECT stay_id, charttime,{labels},
{flag_cols}
      FROM flagged
    ) TO '{C.FORWARD_TARGETS_PQ.as_posix()}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    with heartbeat("forward-window aggregation", watch=C.FORWARD_TARGETS_PQ):
        con.execute(sql)
    con.close()


def _report() -> None:
    con = connect_duckdb()
    con.execute(f"CREATE OR REPLACE VIEW f AS SELECT * FROM "
                f"read_parquet('{C.FORWARD_TARGETS_PQ.as_posix()}')")
    total, stays = con.execute("SELECT count(*), count(DISTINCT stay_id) FROM f").fetchone()
    out: dict = {"rows": total, "admissions": stays,
                 "definition": {"fio2_rise": C.D_FIO2_RISE, "peep_rise": C.D_PEEP_RISE,
                                "spo2_below": C.D_SPO2_BELOW,
                                "baseline": "LOCF value in force",
                                "forward_values": "measured only"},
                 "horizons": {}}

    out["variants"] = {}
    for variant, col in (("composite", "y_composite"), ("strict", "y_strict")):
        console.rule(f"[bold cyan]Target D ({variant}) -- prevalence and coverage "
                     f"by horizon")
        log(f"  {'horizon':>8} {'labelled':>12} {'dropped':>10} {'positive':>11} "
            f"{'rate':>8} {'admissions >=1':>15}")
        out["variants"][variant] = {}
        prev_rate = -1.0
        for h in C.HORIZONS_H:
            n_lab, n_pos, n_stay = con.execute(f"""
                SELECT count(*) FILTER (WHERE {col}_{h}h IS NOT NULL),
                       sum({col}_{h}h),
                       count(DISTINCT CASE WHEN {col}_{h}h = 1 THEN stay_id END)
                FROM f""").fetchone()
            rate = n_pos / n_lab
            dropped = total - n_lab
            log(f"  {h:>6} h {n_lab:>12,} {dropped:>10,} {n_pos:>11,} "
                f"{100*rate:>7.2f}% {n_stay:>14,} ({100*n_stay/stays:.1f}%)")
            out["variants"][variant][f"{h}h"] = {
                "rows_labelled": n_lab, "rows_dropped_unobservable": dropped,
                "pct_dropped": 100 * dropped / total,
                "positive": int(n_pos), "positive_rate": rate,
                "admissions_any": n_stay, "admissions_any_pct": 100 * n_stay / stays}
            # A longer look-ahead cannot contain fewer events.
            assert rate >= prev_rate, (
                f"{variant}: prevalence fell from {prev_rate:.4f} to {rate:.4f} "
                f"at {h} h -- a longer window cannot see fewer events")
            prev_rate = rate
    out["horizons"] = out["variants"]["composite"]

    # Strict is a subset of composite by construction; if it is not, the SQL is wrong.
    for h in C.HORIZONS_H:
        assert (out["variants"]["strict"][f"{h}h"]["positive"]
                <= out["variants"]["composite"][f"{h}h"]["positive"]), (
            f"strict fired more often than composite at {h} h -- impossible")

    h = C.PRIMARY_HORIZON_H
    console.rule(f"[bold cyan]Components at {h} h -- which ones carry D")
    comp_names = {"d_spo2": "desaturation SpO2<88  (patient state)",
                  "d_fio2": "FiO2 escalation       (clinician action)",
                  "d_peep": "PEEP escalation       (clinician action)",
                  "d_vaso": "new vasopressor       (clinician action)"}
    n_lab = out["horizons"][f"{h}h"]["rows_labelled"]
    n_pos = out["horizons"][f"{h}h"]["positive"]
    comps = {}
    for c, label in comp_names.items():
        alone, fires = con.execute(f"""
            SELECT count(*) FILTER (WHERE {c}_{h}h AND NOT (
                     {" OR ".join(f"{o}_{h}h" for o in comp_names if o != c)})),
                   count(*) FILTER (WHERE {c}_{h}h)
            FROM f WHERE y_composite_{h}h IS NOT NULL""").fetchone()
        comps[c] = {"fires": fires, "share_of_positives": fires / max(n_pos, 1),
                    "sole_trigger": alone, "marginal_share": alone / max(n_pos, 1)}
        log(f"  {label}  fires {fires:>9,} ({100*fires/max(n_pos,1):>5.1f}% of positives)  "
            f"sole trigger {alone:>9,} ({100*alone/max(n_pos,1):>5.1f}%)")

    # What is left if the clinician-action components are removed? This bounds the
    # reverse-causation exposure instead of hand-waving about it.
    pure, = con.execute(f"""
        SELECT count(*) FILTER (WHERE d_spo2_{h}h)
        FROM f WHERE y_composite_{h}h IS NOT NULL""").fetchone()
    log(f"  [dim]patient-state component alone would give "
        f"{pure:,} positives ({100*pure/n_lab:.2f}% of labelled rows)[/dim]")
    out["components_6h"] = comps
    out["patient_state_only_rate"] = pure / n_lab

    # ------------------------------------------------------------------
    # Why this does not reproduce the published 7.98%
    #
    # explore_targets.py used hourly bins and took the CURRENT HOUR'S measured
    # minimum as the escalation baseline. FiO2 is charted on ~22% of rows, so in
    # roughly four bins out of five the FiO2 and PEEP components silently could
    # not fire at all -- the 7.98% was an undercount driven by charting sparsity,
    # not a property of the definition. Using the value in force is both more
    # clinically faithful and ~4x more eligible. That is the entire delta, and it
    # is reported rather than tuned away.
    # ------------------------------------------------------------------
    elig = con.execute(f"""
        SELECT round(100.0*count(*) FILTER (WHERE fio2 IS NOT NULL)/count(*),2),
               round(100.0*count(*) FILTER (WHERE fio2_locf IS NOT NULL)/count(*),2),
               round(100.0*count(*) FILTER (WHERE peep IS NOT NULL)/count(*),2),
               round(100.0*count(*) FILTER (WHERE peep_locf IS NOT NULL)/count(*),2)
        FROM read_parquet('{C.T2_IMPUTED_PQ.as_posix()}')""").fetchone()
    published = 0.0798
    got = out["horizons"][f"{C.PRIMARY_HORIZON_H}h"]["positive_rate"]
    strict = out["variants"]["strict"][f"{C.PRIMARY_HORIZON_H}h"]["positive_rate"]
    console.rule("[bold cyan]Reconciliation with the published 7.98% estimate")
    log(f"  published (hourly bins, same-hour measured baseline) {100*published:>6.2f}%")
    log(f"  row grain, baseline = value in force                 {100*got:>6.2f}%  "
        f"({100*(got-published):+.2f} pp)")
    log(f"  row grain, baseline must be <={C.STRICT_BASELINE_AGE_MIN:.0f} min old       "
        f"        {100*strict:>6.2f}%  ({100*(strict-published):+.2f} pp)")
    log(f"  [dim]cause: baseline eligibility -- fio2 measured {elig[0]}% of rows vs "
        f"fio2_locf {elig[1]}%; peep {elig[2]}% vs {elig[3]}%[/dim]")
    out["reconciliation"] = {
        "published_hourly_estimate": published,
        "row_grain_composite": got, "row_grain_strict": strict,
        "baseline_eligibility_pct": {"fio2_measured": elig[0], "fio2_locf": elig[1],
                                     "peep_measured": elig[2], "peep_locf": elig[3]},
        "explanation": "explore_targets required a same-hour measured baseline, so "
                       "the FiO2/PEEP components could not fire on ~78% of bins. The "
                       "published 7.98% was an undercount caused by charting "
                       "sparsity, not a different definition."}
    # Sanity only: a build this far off would mean the definition really did drift.
    assert 0.02 < got < 0.35, f"row-grain prevalence {got:.4f} is not credible"

    SUMMARY_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    log(f"report -> {SUMMARY_JSON}")
    con.close()


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
