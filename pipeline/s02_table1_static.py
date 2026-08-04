"""Stage 2 -- Table 1: static patient background, one row per admission.

patients + admissions + icustays + Charlson comorbidities.

Columns known only at or after discharge are built but tagged `leaky_` so the
assembler can exclude them from the feature matrix. Discharge location, total
length of stay and death times are outcomes, not inputs.
"""
from __future__ import annotations

from . import config as C
from .charlson import CHARLSON, age_points_sql, category_sql, score_sql
from .common import account_parquet, cached_stage, connect_duckdb, log
from .s01_cohort_strict import COHORT_STRICT_PQ


def main(force: bool = False) -> None:
    sources = [C.PATIENTS, C.ADMISSIONS, C.DIAGNOSES_ICD, COHORT_STRICT_PQ]
    with cached_stage("s02_table1_static", sources=sources,
                      output=C.T1_STATIC_PQ, force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()

        con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
          SELECT * FROM read_parquet('{COHORT_STRICT_PQ.as_posix()}');
        CREATE OR REPLACE TABLE patients AS
          SELECT * FROM read_csv_auto('{C.PATIENTS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE admissions AS
          SELECT * FROM read_csv_auto('{C.ADMISSIONS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE diagnoses AS
          SELECT hadm_id, icd_version, upper(trim(icd_code)) AS code
          FROM read_csv_auto('{C.DIAGNOSES_ICD.as_posix()}', header=true);
        """)

        v9, v10 = con.execute(
            "SELECT sum(icd_version=9), sum(icd_version=10) FROM diagnoses").fetchone()
        log(f"diagnoses: ICD-9 {v9:,} ({100*v9/(v9+v10):.1f}%) | "
            f"ICD-10 {v10:,} ({100*v10/(v9+v10):.1f}%) -- both mapped")

        # --- Charlson: one pass, per-category flags aggregated to the admission
        flags = ",\n               ".join(
            f"max(CASE WHEN {category_sql(n)} THEN 1 ELSE 0 END) AS cci_{n}"
            for n in CHARLSON)
        con.execute(f"""
        CREATE OR REPLACE TABLE comorb AS
        SELECT hadm_id,
               {flags}
        FROM diagnoses
        GROUP BY hadm_id;
        """)
        log(f"Charlson: {len(CHARLSON)} categories resolved over "
            f"{con.execute('SELECT count(*) FROM comorb').fetchone()[0]:,} admissions")

        cci_cols = ", ".join(f"COALESCE(cm.cci_{n}, 0) AS cci_{n}" for n in CHARLSON)

        con.execute(f"""
        CREATE OR REPLACE TABLE t1 AS
        SELECT
            c.stay_id, c.subject_id, c.hadm_id,

            -- ---------- features available at the bedside ----------
            p.gender,
            p.anchor_age + (year(c.intime) - p.anchor_year)          AS age_at_icu,
            a.admission_type, a.admission_location,
            a.insurance, a.language, a.marital_status, a.race,
            c.first_careunit,
            date_diff('hour', a.admittime, c.intime)                 AS hours_admit_to_icu,
            date_diff('minute', a.edregtime, a.edouttime)            AS ed_minutes,
            row_number() OVER (PARTITION BY c.subject_id ORDER BY c.intime) - 1
                                                                     AS prior_icu_stays,
            {cci_cols},

            -- ---------- outcome / post-hoc: NOT model inputs ----------
            c.intime  AS leaky_icu_intime,
            c.outtime AS leaky_icu_outtime,
            c.los     AS leaky_icu_los_days,
            c.vent_start, c.vent_end, c.vent_hours,
            a.admittime AS leaky_admittime, a.dischtime AS leaky_dischtime,
            a.deathtime AS leaky_deathtime,
            a.discharge_location AS leaky_discharge_location,
            a.hospital_expire_flag AS leaky_hospital_expire_flag,
            p.dod AS leaky_dod
        FROM cohort c
        LEFT JOIN patients   p USING (subject_id)
        LEFT JOIN admissions a USING (hadm_id)
        LEFT JOIN comorb    cm USING (hadm_id);
        """)

        # Charlson index = weighted categories + age adjustment
        con.execute(f"""
        CREATE OR REPLACE TABLE t1 AS
        SELECT *,
               ({score_sql('cci_')})              AS charlson_comorbidity_score,
               ({age_points_sql('age_at_icu')})   AS charlson_age_points,
               ({score_sql('cci_')}) + ({age_points_sql('age_at_icu')})
                                                  AS charlson_index
        FROM t1;
        """)

        con.execute(f"""COPY t1 TO '{C.T1_STATIC_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD)""")
        account_parquet("Table 1 (static)", C.T1_STATIC_PQ)

        s = con.execute("""
            SELECT median(age_at_icu), median(charlson_index),
                   avg(cci_chronic_pulmonary), avg(cci_congestive_heart_failure),
                   avg(cci_renal_disease), avg(cci_malignancy), avg(cci_diabetes_complicated),
                   sum(age_at_icu IS NULL)
            FROM t1""").fetchone()
        log(f"age median {s[0]:.0f} | Charlson index median {s[1]:.0f}")
        log(f"prevalence -- COPD {100*s[2]:.1f}%  CHF {100*s[3]:.1f}%  renal {100*s[4]:.1f}%  "
            f"malignancy {100*s[5]:.1f}%  complicated diabetes {100*s[6]:.1f}%")
        if s[7]:
            log(f"[yellow]{s[7]:,} admissions with no age[/yellow]")

        ncols = con.execute("SELECT count(*) FROM (DESCRIBE t1)").fetchone()[0]
        nleak = con.execute(
            "SELECT count(*) FROM (DESCRIBE t1) WHERE column_name LIKE 'leaky_%'").fetchone()[0]
        log(f"{ncols} columns ({nleak} tagged leaky_ and excluded from features)")
        con.close()


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
