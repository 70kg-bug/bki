"""Measure candidate prediction targets, so the label can be chosen on evidence.

`warning` is a caregiver documentation flag: a model given only *which* readings
were charted reaches 84% of full performance on it. This module computes the
realistic alternatives from data already on disk and reports how learnable each
one looks -- prevalence, assessable cohort, and (for time-varying targets) how
much of a stay is positive.

Nothing here changes the pipeline. It produces reports/target_candidates.json.
"""
from __future__ import annotations

import json

from . import config as C
from .common import connect_duckdb, console, log, stage

OUT = C.REPORTS / "target_candidates.json"

# NHSN ventilator-associated condition thresholds
VAC_PEEP_RISE = 3.0     # cmH2O above the baseline daily minimum
VAC_FIO2_RISE = 20.0    # percentage points above the baseline daily minimum
VAC_MIN_VENT_DAYS = 4   # 2 stable days + 2 worsened days

HORIZON_HOURS = 6
DESAT_THRESHOLD = 88.0


def main() -> None:
    con = connect_duckdb()
    con.execute(f"""
    CREATE OR REPLACE VIEW wide AS SELECT * FROM read_parquet('{C.T2_WIDE_PQ.as_posix()}');
    CREATE OR REPLACE VIEW t4   AS SELECT * FROM read_parquet('{C.T4_OUTCOMES_PQ.as_posix()}');
    CREATE OR REPLACE VIEW t3   AS SELECT * FROM read_parquet('{C.T3_INTERV_PQ.as_posix()}');
    """)
    total_rows, total_stays = con.execute(
        "SELECT count(*), count(DISTINCT stay_id) FROM wide").fetchone()
    out: dict = {"cohort": {"rows": total_rows, "admissions": total_stays}}
    log(f"cohort: {total_rows:,} rows / {total_stays:,} admissions")

    # ---------------------------------------------------------------- baseline
    with stage("Current target: `warning`"):
        n, rate = con.execute(
            f"SELECT sum({C.LEGACY_TARGET}), avg({C.LEGACY_TARGET}) FROM wide").fetchone()
        stays_any = con.execute(
            f"SELECT count(DISTINCT stay_id) FROM wide "
            f"WHERE {C.LEGACY_TARGET}=1").fetchone()[0]
        log(f"  row-level positives {n:,} ({100*rate:.2f}%)  |  "
            f"admissions with >=1 positive {stays_any:,} "
            f"({100*stays_any/total_stays:.1f}%)")
        out["warning"] = {"row_rate": rate, "rows_positive": n,
                          "admissions_any": stays_any,
                          "kind": "per-timestamp",
                          "note": "caregiver documentation flag; 84% reachable "
                                  "from charting pattern alone"}

    # ------------------------------------------------- A. NHSN VAC (per stay)
    with stage("A. NHSN ventilator-associated condition (VAC)"):
        con.execute(f"""
        CREATE OR REPLACE TABLE daily AS
        SELECT stay_id, CAST(charttime AS DATE) AS day,
               min(peep) AS min_peep, min(fio2) AS min_fio2
        FROM wide
        WHERE peep IS NOT NULL OR fio2 IS NOT NULL
        GROUP BY 1, 2;

        CREATE OR REPLACE TABLE daily_seq AS
        SELECT *, row_number() OVER (PARTITION BY stay_id ORDER BY day) AS d,
               count(*)      OVER (PARTITION BY stay_id)                AS n_days
        FROM daily;

        -- baseline = the lower of the two preceding days (NHSN: 2 days of
        -- stable-or-decreasing support), worsening must persist 2 days.
        CREATE OR REPLACE TABLE vac AS
        SELECT stay_id, day,
               min_peep, min_fio2,
               least(lag(min_peep, 1) OVER w, lag(min_peep, 2) OVER w) AS base_peep,
               least(lag(min_fio2, 1) OVER w, lag(min_fio2, 2) OVER w) AS base_fio2,
               lead(min_peep, 1) OVER w AS next_peep,
               lead(min_fio2, 1) OVER w AS next_fio2,
               n_days
        FROM daily_seq
        WINDOW w AS (PARTITION BY stay_id ORDER BY day);

        CREATE OR REPLACE TABLE vac_flag AS
        SELECT stay_id,
               max(CASE WHEN n_days >= {VAC_MIN_VENT_DAYS}
                         AND ((min_peep - base_peep >= {VAC_PEEP_RISE}
                               AND next_peep - base_peep >= {VAC_PEEP_RISE})
                           OR (min_fio2 - base_fio2 >= {VAC_FIO2_RISE}
                               AND next_fio2 - base_fio2 >= {VAC_FIO2_RISE}))
                        THEN 1 ELSE 0 END) AS vac
        FROM vac GROUP BY stay_id;
        """)
        assessable, vac_n = con.execute(f"""
            SELECT count(*) FILTER (WHERE s.n_days >= {VAC_MIN_VENT_DAYS}),
                   sum(v.vac)
            FROM vac_flag v
            JOIN (SELECT stay_id, max(n_days) AS n_days FROM daily_seq GROUP BY 1) s
              USING (stay_id)""").fetchone()
        log(f"  assessable admissions (>= {VAC_MIN_VENT_DAYS} ventilated days): "
            f"{assessable:,} of {total_stays:,}")
        log(f"  VAC positive: {vac_n:,} ({100*vac_n/max(assessable,1):.1f}% of assessable, "
            f"{100*vac_n/total_stays:.1f}% of cohort)")
        out["nhsn_vac"] = {"assessable_admissions": assessable, "positive": int(vac_n),
                           "rate_of_assessable": vac_n / max(assessable, 1),
                           "kind": "per-admission (day-anchored)",
                           "note": "published standard; protocol PDF already in "
                                   "bki/rag document/nhsn_vae_vap.pdf"}

    # ------------------------- B/C. time-varying targets on an hourly grid ----
    with stage(f"B/C. Time-varying targets, {HORIZON_HOURS} h look-ahead"):
        con.execute(f"""
        CREATE OR REPLACE TABLE hourly AS
        SELECT w.stay_id,
               date_trunc('hour', w.charttime) AS hr,
               min(w.fio2) AS fio2_min, max(w.fio2) AS fio2_max,
               min(w.peep) AS peep_min, max(w.peep) AS peep_max,
               min(w.spo2) AS spo2_min,
               max(COALESCE(t.vasopressor_running, 0)) AS vaso
        FROM wide w
        LEFT JOIN t3 t ON t.stay_id = w.stay_id AND t.charttime = w.charttime
        GROUP BY 1, 2;
        """)
        n_bins = con.execute("SELECT count(*) FROM hourly").fetchone()[0]
        log(f"  hourly bins: {n_bins:,}")

        # forward window (t, t+H] via a bounded range self-join
        con.execute(f"""
        CREATE OR REPLACE TABLE fwd AS
        SELECT a.stay_id, a.hr,
               a.fio2_min AS fio2_now, a.peep_min AS peep_now, a.vaso AS vaso_now,
               max(b.fio2_max) AS fio2_ahead,
               max(b.peep_max) AS peep_ahead,
               min(b.spo2_min) AS spo2_ahead,
               max(b.vaso)     AS vaso_ahead
        FROM hourly a
        LEFT JOIN hourly b
               ON b.stay_id = a.stay_id
              AND b.hr >  a.hr
              AND b.hr <= a.hr + INTERVAL {HORIZON_HOURS} HOUR
        GROUP BY 1, 2, 3, 4, 5;
        """)

        # --- B. ventilator support escalation
        esc = con.execute(f"""
            SELECT count(*),
                   count(*) FILTER (WHERE
                        (fio2_ahead - fio2_now >= {VAC_FIO2_RISE})
                     OR (peep_ahead - peep_now >= {VAC_PEEP_RISE})),
                   count(DISTINCT stay_id) FILTER (WHERE
                        (fio2_ahead - fio2_now >= {VAC_FIO2_RISE})
                     OR (peep_ahead - peep_now >= {VAC_PEEP_RISE}))
            FROM fwd""").fetchone()
        log(f"  B. support escalation (FiO2 +{VAC_FIO2_RISE:g}pts or "
            f"PEEP +{VAC_PEEP_RISE:g}) within {HORIZON_HOURS} h: "
            f"{esc[1]:,}/{esc[0]:,} bins ({100*esc[1]/esc[0]:.2f}%), "
            f"{esc[2]:,} admissions ({100*esc[2]/total_stays:.1f}%)")
        out["support_escalation"] = {"bins": esc[0], "positive": esc[1],
                                     "bin_rate": esc[1] / esc[0],
                                     "admissions_any": esc[2],
                                     "kind": "per-timestamp, forward-looking",
                                     "horizon_hours": HORIZON_HOURS}

        # --- C. desaturation
        des = con.execute(f"""
            SELECT count(*),
                   count(*) FILTER (WHERE spo2_ahead < {DESAT_THRESHOLD}),
                   count(DISTINCT stay_id) FILTER (WHERE spo2_ahead < {DESAT_THRESHOLD})
            FROM fwd""").fetchone()
        log(f"  C. desaturation (SpO2 < {DESAT_THRESHOLD:g}%) within "
            f"{HORIZON_HOURS} h: {des[1]:,}/{des[0]:,} bins "
            f"({100*des[1]/des[0]:.2f}%), {des[2]:,} admissions "
            f"({100*des[2]/total_stays:.1f}%)")
        out["desaturation"] = {"bins": des[0], "positive": des[1],
                               "bin_rate": des[1] / des[0],
                               "admissions_any": des[2],
                               "kind": "per-timestamp, forward-looking",
                               "horizon_hours": HORIZON_HOURS}

        # --- D. composite deterioration
        comp = con.execute(f"""
            SELECT count(*),
                   count(*) FILTER (WHERE
                        (fio2_ahead - fio2_now >= {VAC_FIO2_RISE})
                     OR (peep_ahead - peep_now >= {VAC_PEEP_RISE})
                     OR (spo2_ahead < {DESAT_THRESHOLD})
                     OR (vaso_ahead = 1 AND vaso_now = 0)),
                   count(DISTINCT stay_id) FILTER (WHERE
                        (fio2_ahead - fio2_now >= {VAC_FIO2_RISE})
                     OR (peep_ahead - peep_now >= {VAC_PEEP_RISE})
                     OR (spo2_ahead < {DESAT_THRESHOLD})
                     OR (vaso_ahead = 1 AND vaso_now = 0))
            FROM fwd""").fetchone()
        log(f"  D. composite deterioration (any of the above, or a new "
            f"vasopressor) within {HORIZON_HOURS} h: {comp[1]:,}/{comp[0]:,} bins "
            f"({100*comp[1]/comp[0]:.2f}%), {comp[2]:,} admissions "
            f"({100*comp[2]/total_stays:.1f}%)")
        out["composite_deterioration"] = {"bins": comp[0], "positive": comp[1],
                                          "bin_rate": comp[1] / comp[0],
                                          "admissions_any": comp[2],
                                          "kind": "per-timestamp, forward-looking",
                                          "horizon_hours": HORIZON_HOURS}

    # -------------------------------------------------- E. stay-level outcomes
    with stage("E. Stay-level outcomes (already built, Table 4)"):
        for col in ("in_hospital_mortality", "icu_mortality", "mortality_28d",
                    "mortality_90d", "prolonged_ventilation", "extubation_failure",
                    "icu_readmission"):
            n, rate = con.execute(f"SELECT sum({col}), avg({col}) FROM t4").fetchone()
            log(f"  {col:<24} {n:>7,}  ({100*rate:5.2f}% of admissions)")
            out[col] = {"positive": int(n), "admission_rate": rate,
                        "kind": "per-admission"}

    OUT.write_text(json.dumps(out, indent=2, default=float))
    log(f"report -> {OUT}")
    con.close()


if __name__ == "__main__":
    main()
