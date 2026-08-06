"""Stage 8 -- Table 3: interventions (drug infusions + ventilator mode).

Drug itemids are resolved from d_items by label at runtime rather than
hardcoded, so a wrong constant cannot silently produce an all-zero column --
whatever matched is logged.

REVERSE-CAUSATION GUARD, and it matters more than it looks:
treatments are given *because* a patient is deteriorating. An infusion started
at the same moment as the target would predict it beautifully while teaching the
model nothing about physiology. Only intervals that began STRICTLY BEFORE a row's
timestamp are visible to that row. Stage 10 asserts this holds.
"""
from __future__ import annotations

import polars as pl

from . import config as C
from .common import account_parquet, cached_stage, connect_duckdb, log

# group -> regex over the d_items label (case-insensitive)
#
# Corticosteroids are deliberately absent: MIMIC-IV's icu/inputevents has only
# 133 "Medications" items and none are steroids -- IV steroids are recorded in
# hosp/prescriptions and hosp/emar instead. Including the group would ship an
# all-zero column, which is noise dressed up as a feature. Picking it up would
# mean a second ETL path over prescriptions (3.5 GB), which the plan scopes as
# a follow-on phase.
DRUG_GROUPS: dict[str, str] = {
    "vasopressor": r"norepinephrine|epinephrine|phenylephrine|vasopressin|dopamine|dobutamine",
    "sedative": r"propofol|midazolam|dexmedetomidine|lorazepam|ketamine",
    "opioid": r"fentanyl|morphine|hydromorphone|remifentanil",
    "paralytic": r"cisatracurium|rocuronium|vecuronium|succinylcholine",
}
ITEM_VENT_MODE = 223849


def main(force: bool = False) -> None:
    sources = [C.INPUTEVENTS, C.D_ITEMS, C.T2_WIDE_PQ, C.TS_LONG_PQ]
    with cached_stage("s08_table3_interventions", sources=sources,
                      output=C.T3_INTERV_PQ, force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()
        con.execute(f"""
        CREATE OR REPLACE TABLE d_items AS
          SELECT * FROM read_csv_auto('{C.D_ITEMS.as_posix()}', header=true);
        CREATE OR REPLACE TABLE wide AS
          SELECT stay_id, charttime FROM read_parquet('{C.T2_WIDE_PQ.as_posix()}');
        """)

        # --- resolve drug itemids from labels --------------------------------
        rows = []
        empty: list[str] = []
        for grp, pattern in DRUG_GROUPS.items():
            got = con.execute(f"""
                SELECT itemid, label FROM d_items
                WHERE linksto = 'inputevents' AND regexp_matches(lower(label), '{pattern}')
            """).fetchall()
            log(f"  {grp:<16} {len(got):>3} itemids: "
                + ", ".join(sorted({r[1] for r in got}))[:110])
            if not got:
                empty.append(grp)
            rows += [(grp, r[0]) for r in got]
        if empty:
            # An unmatched group would become an all-zero column -- noise that
            # looks like a feature. Fail loudly rather than ship it.
            raise RuntimeError(
                f"drug groups matched no itemids: {empty}. Either fix the pattern "
                f"or remove the group -- an all-zero column must not reach the model.")
        if not rows:
            raise RuntimeError("no drug itemids resolved -- check DRUG_GROUPS patterns")
        con.execute("CREATE OR REPLACE TABLE drug_map (grp VARCHAR, itemid BIGINT)")
        con.executemany("INSERT INTO drug_map VALUES (?, ?)", rows)

        ids = ", ".join(str(i) for _, i in rows)
        con.execute(f"""
        CREATE OR REPLACE TABLE infusions AS
        SELECT i.stay_id, m.grp, i.starttime, i.endtime,
               CAST(COALESCE(i.rate, 0) AS DOUBLE)   AS rate,
               CAST(COALESCE(i.amount, 0) AS DOUBLE) AS amount
        FROM read_csv_auto('{C.INPUTEVENTS.as_posix()}', header=true) i
        JOIN drug_map m USING (itemid)
        WHERE i.stay_id IS NOT NULL AND i.starttime IS NOT NULL
          AND i.statusdescription IS DISTINCT FROM 'Rewritten'
          AND i.stay_id IN (SELECT DISTINCT stay_id FROM wide);
        """)
        n_inf = con.execute("SELECT count(*) FROM infusions").fetchone()[0]
        log(f"infusion intervals inside the cohort: {n_inf:,}")

        groups = list(DRUG_GROUPS)
        active = ",\n               ".join(
            f"CAST(max(CASE WHEN f.grp='{g}' THEN 1 ELSE 0 END) AS TINYINT) AS {g}_running,\n"
            f"               CAST(COALESCE(sum(CASE WHEN f.grp='{g}' THEN f.rate END), 0) "
            f"AS FLOAT) AS {g}_rate,\n"
            f"               CAST(COALESCE(max(CASE WHEN f.grp='{g}' THEN "
            f"date_diff('minute', f.starttime, w.charttime) END), -1) AS FLOAT) "
            f"AS {g}_minutes_since_start"
            for g in groups)

        # STRICTLY BEFORE: f.starttime < w.charttime
        con.execute(f"""
        CREATE OR REPLACE TABLE t3_drugs AS
        SELECT w.stay_id, w.charttime,
               {active}
        FROM wide w
        LEFT JOIN infusions f
               ON f.stay_id = w.stay_id
              AND f.starttime <  w.charttime
              AND COALESCE(f.endtime, f.starttime) > w.charttime
        GROUP BY w.stay_id, w.charttime;
        """)
        log("drug state resolved per timestamp (strictly-before intervals only)")

        # --- ventilator mode: last mode charted strictly before the row -------
        con.execute(f"""
        CREATE OR REPLACE TABLE modes AS
        SELECT stay_id, charttime, value AS ventilator_mode
        FROM read_parquet('{C.TS_LONG_PQ.as_posix()}')
        WHERE itemid = {ITEM_VENT_MODE} AND value IS NOT NULL;

        CREATE OR REPLACE TABLE t3 AS
        SELECT d.*,
               m.ventilator_mode,
               CAST(COALESCE(date_diff('minute', m.charttime, d.charttime), -1) AS FLOAT)
                   AS ventilator_mode_age_min
        FROM t3_drugs d
        ASOF LEFT JOIN modes m
          ON d.stay_id = m.stay_id AND d.charttime > m.charttime;
        """)

        con.execute(f"""COPY (SELECT * FROM t3 ORDER BY stay_id, charttime)
                        TO '{C.T3_INTERV_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""")

        log("intervention prevalence (share of rows with the drug running):")
        for g in groups:
            pct = con.execute(f"SELECT 100.0*avg({g}_running) FROM t3").fetchone()[0]
            log(f"  {g:<16} {pct:5.2f}%")
        nm, top = con.execute("""
            SELECT count(*) FILTER (WHERE ventilator_mode IS NOT NULL),
                   list(DISTINCT ventilator_mode)[1:6] FROM t3""").fetchone()
        log(f"  ventilator mode known on {nm:,} rows; e.g. {top}")
        con.close()

    account_parquet("Table 3 (interventions)", C.T3_INTERV_PQ, subject_col=None)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
