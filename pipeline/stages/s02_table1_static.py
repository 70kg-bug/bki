"""Stage 2 -- Table 1: static patient background, one row per admission.

patients + admissions + icustays + Charlson comorbidities + body metrics.

Columns known only at or after discharge are built but tagged `leaky_` so the
assembler can exclude them from the feature matrix. Discharge location, total
length of stay and death times are outcomes, not inputs.

RUNS AFTER s04, ON THE FINAL COHORT. It used to run second, against the strict
cohort, while every later stage used the wider final cohort -- so 7,423
admissions reached the model matrix with all 33 static features NULL. That is
not merely missing data: `gender IS NULL` was a bit-exact indicator of which
cohort arm a stay came from, and the two arms have different label
prevalence, so the model was handed a free stratum flag. Building Table 1 on
the final cohort fills the block from the same source and deletes the shortcut.
"""
from __future__ import annotations

from .. import config as C
from ..core.charlson import CHARLSON, age_points_sql, category_sql, score_sql
from ..common import account_parquet, cached_stage, connect_duckdb, log

# Plausibility bounds for the body metrics. The charted values carry obvious
# junk -- the BigQuery export of the same field runs from 1 kg to 784 kg.
# Out-of-range values become NULL rather than being clipped: a clipped 784 kg
# is still wrong, just less obviously so.
HEIGHT_CM_RANGE = (100.0, 250.0)
WEIGHT_KG_RANGE = (20.0, 400.0)

IN_TO_CM = 2.54


def _strict_cohort():
    """The pre-fix cohort. Retained so the gate harness can reproduce the hole."""
    from .s01_cohort_strict import COHORT_STRICT_PQ
    return COHORT_STRICT_PQ


def main(force: bool = False) -> None:
    cohort_pq = C.COHORT_PQ if C.S02_FINAL_COHORT else _strict_cohort()
    sources = [C.PATIENTS, C.ADMISSIONS, C.DIAGNOSES_ICD, cohort_pq]
    if C.STATIC_BODY_METRICS:
        sources.append(C.TS_LONG_PQ)
    with cached_stage("s02_table1_static", sources=sources,
                      output=C.T1_STATIC_PQ, force=force,
                      extra=C.FP_STATIC) as ran:
        if not ran:
            return
        con = connect_duckdb()
        log(f"cohort: {cohort_pq.name} "
            f"({'final -- includes the s04 invasive-evidence arm' if C.S02_FINAL_COHORT else 'STRICT -- legacy, leaves a hole in s10'})")

        con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
          SELECT * FROM read_parquet('{cohort_pq.as_posix()}');
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

        # --- Body metrics from the chartevents cache ----------------------
        # Read out of ts_long rather than hosp/omr. Both measure the same
        # thing; only one is measured on ICU patients. The EARLIEST reading in
        # the stay is taken, not the nearest or the mean: a later weight is
        # partly a treatment effect (fluid resuscitation routinely adds kilos),
        # and a feature must not be downstream of the thing it predicts.
        if C.STATIC_BODY_METRICS:
            lo_h, hi_h = HEIGHT_CM_RANGE
            lo_w, hi_w = WEIGHT_KG_RANGE
            scale = " ".join(
                f"WHEN itemid = {i} THEN valuenum * {f}"
                for i, f in {**C.BODY_WEIGHT_ITEMIDS, **C.BODY_HEIGHT_ITEMIDS}.items())
            w_ids = ", ".join(str(i) for i in C.BODY_WEIGHT_ITEMIDS)
            h_ids = ", ".join(str(i) for i in C.BODY_HEIGHT_ITEMIDS)
            con.execute(f"""
            CREATE OR REPLACE TABLE body AS
            WITH raw AS (
                SELECT stay_id, charttime, itemid,
                       CASE {scale} END AS v,
                       CASE WHEN itemid IN ({w_ids}) THEN 'weight' ELSE 'height' END AS metric
                FROM read_parquet('{C.TS_LONG_PQ.as_posix()}')
                WHERE itemid IN ({w_ids}, {h_ids}) AND valuenum IS NOT NULL
            ),
            ranked AS (
                SELECT stay_id, metric, v,
                       row_number() OVER (PARTITION BY stay_id, metric
                                          ORDER BY charttime) AS rn
                FROM raw
                WHERE (metric = 'weight' AND v BETWEEN {lo_w} AND {hi_w})
                   OR (metric = 'height' AND v BETWEEN {lo_h} AND {hi_h})
            )
            SELECT stay_id,
                   max(CASE WHEN metric = 'height' THEN v END) AS height_cm,
                   max(CASE WHEN metric = 'weight' THEN v END) AS weight_kg
            FROM ranked WHERE rn = 1 GROUP BY stay_id;
            """)
            n_c = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
            n_b, n_h, n_w = con.execute(
                "SELECT count(*), count(height_cm), count(weight_kg) FROM body").fetchone()
            log(f"body metrics: height {n_h:,} ({100*n_h/n_c:.1f}% of cohort), "
                f"weight {n_w:,} ({100*n_w/n_c:.1f}%) over {n_b:,} admissions")
        else:
            con.execute("CREATE OR REPLACE TABLE body AS "
                        "SELECT stay_id, NULL::DOUBLE AS height_cm, "
                        "NULL::DOUBLE AS weight_kg FROM cohort LIMIT 0")

        # ARDSNet predicted body weight -- a function of height and sex only,
        # never of actual weight. It is the denominator that makes tidal
        # volume comparable between a 150 cm and a 190 cm patient, which raw
        # millilitres are not.
        if C.STATIC_BODY_METRICS:
            height_in = f"(b.height_cm / {IN_TO_CM})"
            body_cols = f"""
            b.height_cm, b.weight_kg,
            CASE WHEN b.height_cm IS NULL OR p.gender IS NULL THEN NULL
                 WHEN p.gender = 'M' THEN 50.0  + 2.3 * ({height_in} - 60.0)
                 ELSE                      45.5 + 2.3 * ({height_in} - 60.0)
            END                                                      AS pbw_kg,"""
        else:
            body_cols = ""

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
            {body_cols}

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
        LEFT JOIN comorb    cm USING (hadm_id)
        LEFT JOIN body       b USING (stay_id);
        """)

        # Charlson index = weighted categories + age adjustment
        score = score_sql("cci_", hierarchy=C.CHARLSON_HIERARCHY)
        age_pts = age_points_sql("age_at_icu")
        con.execute(f"""
        CREATE OR REPLACE TABLE t1 AS
        SELECT *,
               ({score})                          AS charlson_comorbidity_score,
               ({age_pts})                        AS charlson_age_points,
               ({score}) + ({age_pts})            AS charlson_index
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
