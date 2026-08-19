"""Stage 1 -- the strict ventilated cohort.

Admissions with a recorded invasive-ventilation procedure (itemid 225792),
together with the interval they were actually ventilated. Cheap: reads only
icustays and procedureevents.

The wider detection pass lives in s04_cohort_final, which needs the cache.
"""
from __future__ import annotations

from .. import config as C
from ..common import account, cached_stage, connect_duckdb, log

COHORT_STRICT_PQ = C.BUILD / "cohort_strict.parquet"


def main(force: bool = False) -> None:
    with cached_stage("s01_cohort_strict", sources=[C.ICUSTAYS, C.PROCEDUREEVENTS],
                      output=COHORT_STRICT_PQ, force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()

        con.execute(f"""
        CREATE OR REPLACE TABLE icustays AS
          SELECT * FROM read_csv_auto('{C.ICUSTAYS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE proc AS
          SELECT * FROM read_csv_auto('{C.PROCEDUREEVENTS.as_posix()}', header=true);
        """)

        total_stays, total_subj = con.execute(
            "SELECT count(*), count(DISTINCT subject_id) FROM icustays").fetchone()
        account("icustays (all ICU admissions)", rows=total_stays, stays=total_stays,
                subjects=total_subj)

        con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        WITH vent AS (
            SELECT stay_id,
                   min(starttime)              AS vent_start,
                   max(COALESCE(endtime, starttime)) AS vent_end,
                   count(*)                    AS n_vent_events
            FROM proc
            WHERE itemid = {C.ITEM_INVASIVE_VENT} AND stay_id IS NOT NULL
            GROUP BY stay_id
        )
        SELECT i.subject_id, i.hadm_id, i.stay_id,
               i.first_careunit, i.last_careunit, i.intime, i.outtime, i.los,
               v.vent_start, v.vent_end, v.n_vent_events,
               -- !! FLAGGED FOR REMOVAL -- this is FUTURE INFORMATION.
               -- It is the length of the COMPLETED episode, and s02 broadcasts
               -- it per admission, so every reading carries how long the stay
               -- will last -- including the first. features.json documents
               -- `full` as "all at or before t"; this is the exception, and the
               -- leakage guards miss it because they look for leaky_ prefixes
               -- and forward-label columns, not whole-stay aggregates.
               -- Measured: constant in 39,319/39,319 stays; serving the causal
               -- form instead moves the displayed band on 7.95% of readings.
               -- See FINDINGS.md 7.1 and reports/tool_causal_parity.json.
               -- Left in place because training is finalised. At the next
               -- retrain use elapsed hours at t, which is what serving uses.
               date_diff('hour', v.vent_start, v.vent_end) AS vent_hours,
               'strict' AS cohort_source
        FROM icustays i JOIN vent v USING (stay_id)
        WHERE v.vent_end > v.vent_start;
        """)

        n, subj = con.execute(
            "SELECT count(*), count(DISTINCT subject_id) FROM cohort").fetchone()
        account("cohort A: invasive ventilation (225792)", rows=n, stays=n, subjects=subj)

        niv = con.execute(
            f"SELECT count(DISTINCT stay_id) FROM proc WHERE itemid={C.ITEM_NONINVASIVE_VENT}"
        ).fetchone()[0]
        log(f"(non-invasive ventilation 225794 covers {niv:,} admissions -- "
            f"excluded per the 'strictly ventilated' decision)")

        stats = con.execute("""
            SELECT median(vent_hours), quantile_cont(vent_hours, 0.25),
                   quantile_cont(vent_hours, 0.75), max(vent_hours),
                   sum(CASE WHEN vent_hours < 1 THEN 1 ELSE 0 END)
            FROM cohort
        """).fetchone()
        log(f"ventilated hours: median {stats[0]:.1f}  IQR {stats[1]:.1f}-{stats[2]:.1f}  "
            f"max {stats[3]:.0f}   (<1h: {stats[4]:,} admissions)")

        con.execute(f"""COPY cohort TO '{COHORT_STRICT_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)""")
        con.close()


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
