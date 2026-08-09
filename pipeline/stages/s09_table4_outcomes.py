"""Stage 9 -- Table 4: outcomes, one row per admission.

Built as specified. The training target stays `warning` (stage 5); these are for
stratified evaluation -- e.g. does the model hold up equally on patients who
died and patients who survived -- and for future work.

Everything here is known only after the fact, so nothing in this table may ever
enter the feature matrix. Stage 10 enforces that.
"""
from __future__ import annotations

from .. import config as C
from ..common import account_parquet, cached_stage, connect_duckdb, log

REINTUBATION_WINDOW_H = 48
PROLONGED_VENT_H = 24 * 7


def main(force: bool = False) -> None:
    sources = [C.COHORT_PQ, C.ADMISSIONS, C.PATIENTS, C.PROCEDUREEVENTS, C.ICUSTAYS]
    with cached_stage("s09_table4_outcomes", sources=sources,
                      output=C.T4_OUTCOMES_PQ, force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()
        con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
          SELECT * FROM read_parquet('{C.COHORT_PQ.as_posix()}');
        CREATE OR REPLACE TABLE adm AS
          SELECT * FROM read_csv_auto('{C.ADMISSIONS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE pat AS
          SELECT * FROM read_csv_auto('{C.PATIENTS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE icu AS
          SELECT * FROM read_csv_auto('{C.ICUSTAYS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE proc AS
          SELECT * FROM read_csv_auto('{C.PROCEDUREEVENTS.as_posix()}', header=true);
        """)

        # Reintubation: a later invasive-ventilation episode starting within
        # REINTUBATION_WINDOW_H of the previous one ending.
        con.execute(f"""
        CREATE OR REPLACE TABLE vent_episodes AS
        SELECT stay_id, starttime, endtime,
               lead(starttime) OVER (PARTITION BY stay_id ORDER BY starttime) AS next_start
        FROM proc WHERE itemid = {C.ITEM_INVASIVE_VENT} AND stay_id IS NOT NULL;

        CREATE OR REPLACE TABLE reintub AS
        SELECT stay_id,
               max(CASE WHEN next_start IS NOT NULL
                         AND date_diff('hour', endtime, next_start)
                             BETWEEN 0 AND {REINTUBATION_WINDOW_H}
                        THEN 1 ELSE 0 END) AS extubation_failure
        FROM vent_episodes GROUP BY stay_id;
        """)

        # ICU readmission: any later ICU stay for the same hospital admission.
        con.execute("""
        CREATE OR REPLACE TABLE readmit AS
        SELECT a.stay_id,
               CASE WHEN count(b.stay_id) > 0 THEN 1 ELSE 0 END AS icu_readmission
        FROM icu a LEFT JOIN icu b
          ON a.hadm_id = b.hadm_id AND b.intime > a.outtime
        GROUP BY a.stay_id;
        """)

        con.execute(f"""
        CREATE OR REPLACE TABLE t4 AS
        SELECT c.stay_id, c.subject_id, c.hadm_id,
               CAST(COALESCE(a.hospital_expire_flag, 0) AS TINYINT) AS in_hospital_mortality,
               CAST(CASE WHEN a.deathtime IS NOT NULL
                          AND a.deathtime BETWEEN c.intime AND COALESCE(c.outtime, a.dischtime)
                         THEN 1 ELSE 0 END AS TINYINT)              AS icu_mortality,
               CAST(CASE WHEN p.dod IS NOT NULL
                          AND date_diff('day', c.intime, p.dod) <= 28 THEN 1 ELSE 0 END
                    AS TINYINT)                                     AS mortality_28d,
               CAST(CASE WHEN p.dod IS NOT NULL
                          AND date_diff('day', c.intime, p.dod) <= 90 THEN 1 ELSE 0 END
                    AS TINYINT)                                     AS mortality_90d,
               CAST(CASE WHEN c.vent_hours > {PROLONGED_VENT_H} THEN 1 ELSE 0 END AS TINYINT)
                                                                    AS prolonged_ventilation,
               CAST(COALESCE(r.extubation_failure, 0) AS TINYINT)   AS extubation_failure,
               CAST(COALESCE(rd.icu_readmission, 0) AS TINYINT)     AS icu_readmission,
               c.vent_hours, c.los AS icu_los_days
        FROM cohort c
        LEFT JOIN adm     a  ON a.hadm_id    = c.hadm_id
        LEFT JOIN pat     p  ON p.subject_id = c.subject_id
        LEFT JOIN reintub r  ON r.stay_id    = c.stay_id
        LEFT JOIN readmit rd ON rd.stay_id   = c.stay_id;
        """)

        con.execute(f"""COPY t4 TO '{C.T4_OUTCOMES_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)""")

        log("outcome prevalence across the ventilated cohort:")
        for col in ("in_hospital_mortality", "icu_mortality", "mortality_28d",
                    "mortality_90d", "prolonged_ventilation", "extubation_failure",
                    "icu_readmission"):
            n, pct = con.execute(
                f"SELECT sum({col}), 100.0*avg({col}) FROM t4").fetchone()
            log(f"  {col:<24} {n:>7,}  ({pct:5.2f}%)")
        con.close()

    account_parquet("Table 4 (outcomes)", C.T4_OUTCOMES_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
