"""Stage 4 -- final cohort: strict vs. wider detection.

Procedure documentation is imperfect: some genuinely ventilated admissions have
no 225792 record while still carrying ventilator settings. This stage quantifies
that gap and produces the cohort actually used downstream.

Everything added here is still an *invasively ventilated* patient -- only the
detection is more sensitive. Signals used are invasive-specific on purpose:
an endotracheal/tracheostomy tube, or a charted ventilator mode. PEEP alone is
reported but NOT used to admit a stay, because it also appears with BiPAP.
"""
from __future__ import annotations

from . import config as C
from .common import account, account_parquet, cached_stage, connect_duckdb, log
from .s01_cohort_strict import COHORT_STRICT_PQ

ITEM_VENT_MODE = 223849
ITEM_O2_DEVICE = 226732
ITEM_PEEP = 220339
INVASIVE_DEVICES = ("Endotracheal tube", "Tracheostomy tube")


def main(force: bool = False) -> None:
    sources = [COHORT_STRICT_PQ, C.TS_LONG_PQ]
    with cached_stage("s04_cohort_final", sources=sources,
                      output=C.COHORT_PQ, force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()
        con.execute(f"""
        CREATE OR REPLACE TABLE strict AS
          SELECT * FROM read_parquet('{COHORT_STRICT_PQ.as_posix()}');
        CREATE OR REPLACE VIEW cache AS
          SELECT * FROM read_parquet('{C.TS_LONG_PQ.as_posix()}');
        """)
        n_strict = con.execute("SELECT count(*) FROM strict").fetchone()[0]
        account("cohort A (strict): invasive ventilation procedure", rows=n_strict,
                stays=n_strict)

        devices = ", ".join(f"'{d}'" for d in INVASIVE_DEVICES)
        con.execute(f"""
        CREATE OR REPLACE TABLE evidence AS
        SELECT stay_id, subject_id, hadm_id,
               min(charttime) AS ev_start,
               max(charttime) AS ev_end,
               max(CASE WHEN itemid = {ITEM_VENT_MODE} THEN 1 ELSE 0 END) AS has_vent_mode,
               max(CASE WHEN itemid = {ITEM_O2_DEVICE} AND value IN ({devices})
                        THEN 1 ELSE 0 END)                                 AS has_airway,
               max(CASE WHEN itemid = {ITEM_PEEP} THEN 1 ELSE 0 END)       AS has_peep
        FROM cache
        WHERE itemid IN ({ITEM_VENT_MODE}, {ITEM_O2_DEVICE}, {ITEM_PEEP})
        GROUP BY stay_id, subject_id, hadm_id;
        """)

        for label, pred in [
            ("ventilator mode charted", "has_vent_mode = 1"),
            ("endotracheal / tracheostomy tube", "has_airway = 1"),
            ("PEEP charted (reported only, NOT admitting)", "has_peep = 1"),
            ("invasive evidence (mode OR airway)", "has_vent_mode = 1 OR has_airway = 1"),
        ]:
            tot, extra = con.execute(f"""
                SELECT count(*), count(*) FILTER (WHERE stay_id NOT IN (SELECT stay_id FROM strict))
                FROM evidence WHERE {pred}""").fetchone()
            log(f"  {label:<44} {tot:>7,} admissions   (+{extra:,} not in strict)")

        # Union: strict, plus invasive-evidence stays the procedure table missed.
        con.execute("""
        CREATE OR REPLACE TABLE cohort AS
        SELECT subject_id, hadm_id, stay_id, first_careunit, last_careunit,
               intime, outtime, los, vent_start, vent_end, n_vent_events,
               vent_hours, cohort_source
        FROM strict
        UNION ALL
        SELECT e.subject_id, e.hadm_id, e.stay_id,
               NULL, NULL, NULL, NULL, NULL,
               e.ev_start AS vent_start, e.ev_end AS vent_end, 0 AS n_vent_events,
               date_diff('hour', e.ev_start, e.ev_end) AS vent_hours,
               'evidence' AS cohort_source
        FROM evidence e
        WHERE (e.has_vent_mode = 1 OR e.has_airway = 1)
          AND e.stay_id NOT IN (SELECT stay_id FROM strict)
          AND e.ev_end > e.ev_start;
        """)

        # Backfill icustays attributes for the evidence-only admissions.
        con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT c.subject_id, c.hadm_id, c.stay_id,
               COALESCE(c.first_careunit, i.first_careunit) AS first_careunit,
               COALESCE(c.last_careunit,  i.last_careunit)  AS last_careunit,
               COALESCE(c.intime,  i.intime)                AS intime,
               COALESCE(c.outtime, i.outtime)               AS outtime,
               COALESCE(c.los,     i.los)                   AS los,
               c.vent_start, c.vent_end, c.n_vent_events, c.vent_hours, c.cohort_source
        FROM cohort c
        LEFT JOIN read_csv_auto('{C.ICUSTAYS.as_posix()}', header=true) i USING (stay_id);
        """)

        n_wide, subj = con.execute(
            "SELECT count(*), count(DISTINCT subject_id) FROM cohort").fetchone()
        gained = n_wide - n_strict
        account("cohort B (adopted): strict + invasive evidence", rows=n_wide,
                stays=n_wide, subjects=subj,
                note=f"+{gained:,} vs strict (+{100*gained/n_strict:.1f}%)")

        con.execute(f"""COPY cohort TO '{C.COHORT_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)""")
        con.close()

    account_parquet("cohort.parquet", C.COHORT_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
