"""Stage 5 -- pivot the frozen eleven to one row per (stay_id, charttime).

Reads the cache, not the 42 GB source, so re-running with different plausibility
ranges or a different cohort costs seconds.

Adds the range checking and FiO2 unit normalisation that the original pipeline
had no equivalent of at all.
"""
from __future__ import annotations

from .. import config as C
from ..common import account, account_parquet, cached_stage, connect_duckdb, log

FIO2_ITEM = C.FROZEN_PARAMS["fio2"]


def _one_source(name: str, itemid: int) -> str:
    """Range-checked (and for FiO2, unit-normalised) median for one itemid."""
    lo, hi = C.PLAUSIBLE_RANGES[name]
    if itemid == FIO2_ITEM:
        # MIMIC charts oxygen concentration both as a fraction and a percentage.
        v = (f"CASE WHEN valuenum > 0 AND valuenum <= {C.FIO2_FRACTION_MAX} "
             f"THEN valuenum * 100 ELSE valuenum END")
    else:
        v = "valuenum"
    return (f"median(CASE WHEN itemid = {itemid} "
            f"AND ({v}) BETWEEN {lo} AND {hi} THEN ({v}) END)")


def _value_expr(name: str) -> str:
    """One frozen column, taking its sources in preference order."""
    ids = C.FROZEN_PARAM_SOURCES[name]
    parts = [_one_source(name, i) for i in ids]
    body = parts[0] if len(parts) == 1 else "COALESCE(" + ", ".join(parts) + ")"
    return f"{body} AS {name}"


def main(force: bool = False) -> None:
    sources = [C.TS_LONG_PQ, C.COHORT_PQ]
    with cached_stage("s05_pivot", sources=sources, output=C.T2_WIDE_PQ,
                      force=force) as ran:
        if not ran:
            return
        con = connect_duckdb()
        con.execute(f"""
        CREATE OR REPLACE VIEW cache  AS SELECT * FROM read_parquet('{C.TS_LONG_PQ.as_posix()}');
        CREATE OR REPLACE TABLE cohort AS SELECT * FROM read_parquet('{C.COHORT_PQ.as_posix()}');
        """)

        frozen_ids = ", ".join(str(i) for i in C.FROZEN_SOURCE_ITEMIDS)
        multi = {n: ids for n, ids in C.FROZEN_PARAM_SOURCES.items() if len(ids) > 1}
        for n, ids in multi.items():
            log(f"[bold]{n}[/bold] sourced from {ids} in preference order "
                f"(matches the frozen BigQuery definition)")

        # --- diagnostics before filtering, so losses are explicit -------------
        tot = con.execute(
            f"SELECT count(*) FROM cache WHERE itemid IN ({frozen_ids})").fetchone()[0]
        in_cohort = con.execute(f"""
            SELECT count(*) FROM cache c JOIN cohort h USING (stay_id)
            WHERE c.itemid IN ({frozen_ids})""").fetchone()[0]
        in_window = con.execute(f"""
            SELECT count(*) FROM cache c JOIN cohort h USING (stay_id)
            WHERE c.itemid IN ({frozen_ids})
              AND c.charttime BETWEEN h.vent_start AND h.vent_end""").fetchone()[0]
        log(f"frozen-eleven events: {tot:,} in cache | {in_cohort:,} in cohort "
            f"({100*in_cohort/tot:.1f}%) | {in_window:,} inside the ventilation window "
            f"({100*in_window/tot:.1f}%)")

        # --- FiO2 unit check: confirm rather than assume ----------------------
        frac, pct, zero = con.execute(f"""
            SELECT count(*) FILTER (WHERE valuenum > 0 AND valuenum <= 1.0),
                   count(*) FILTER (WHERE valuenum > 1.0),
                   count(*) FILTER (WHERE valuenum <= 0)
            FROM cache WHERE itemid = {FIO2_ITEM} AND valuenum IS NOT NULL""").fetchone()
        log(f"FiO2 units -- fraction-scale (<=1.0): {frac:,} | percentage-scale: {pct:,} "
            f"| non-positive: {zero:,}"
            + ("  [yellow]-> fractions rescaled x100[/yellow]" if frac else
               "  -> all percentage-scale, no rescale needed"))

        # --- out-of-range accounting -----------------------------------------
        log("values nulled by plausibility range:")
        for name in C.FROZEN_ORDER:
            itemid = C.FROZEN_PARAMS[name]
            lo, hi = C.PLAUSIBLE_RANGES[name]
            v = (f"CASE WHEN valuenum > 0 AND valuenum <= {C.FIO2_FRACTION_MAX} "
                 f"THEN valuenum*100 ELSE valuenum END") if itemid == FIO2_ITEM else "valuenum"
            n, bad = con.execute(f"""
                SELECT count(*) FILTER (WHERE valuenum IS NOT NULL),
                       count(*) FILTER (WHERE valuenum IS NOT NULL
                                        AND NOT (({v}) BETWEEN {lo} AND {hi}))
                FROM cache WHERE itemid = {itemid}""").fetchone()
            if n:
                log(f"  {name:<24} {bad:>8,} / {n:>10,}  ({100*bad/n:5.2f}%)  "
                    f"range [{lo:g}, {hi:g}]")

        # --- pivot ------------------------------------------------------------
        value_cols = ",\n               ".join(_value_expr(n) for n in C.FROZEN_ORDER)
        con.execute(f"""
        CREATE OR REPLACE TABLE wide AS
        SELECT c.stay_id,
               any_value(c.subject_id) AS subject_id,
               c.charttime,
               {value_cols},
               CAST(max(c.warning) AS TINYINT) AS {C.LEGACY_TARGET}
        FROM cache c
        JOIN cohort h USING (stay_id)
        WHERE c.itemid IN ({frozen_ids})
          AND c.charttime BETWEEN h.vent_start AND h.vent_end
        GROUP BY c.stay_id, c.charttime;
        """)

        # A row with no surviving value carries no information -- drop, and say so.
        any_val = " OR ".join(f"{n} IS NOT NULL" for n in C.FROZEN_ORDER)
        before, before_stays = con.execute(
            "SELECT count(*), count(DISTINCT stay_id) FROM wide").fetchone()
        con.execute(f"CREATE OR REPLACE TABLE wide AS SELECT * FROM wide WHERE {any_val}")
        after, after_stays = con.execute(
            "SELECT count(*), count(DISTINCT stay_id) FROM wide").fetchone()
        log(f"dropped {before-after:,} all-null rows "
            f"({before_stays-after_stays:,} admissions lost)")

        con.execute(f"""COPY (SELECT * FROM wide ORDER BY stay_id, charttime)
                        TO '{C.T2_WIDE_PQ.as_posix()}'
                        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""")

        # Fill rates -- the sparsity the imputation stage then has to handle.
        log("column fill rates after pivot:")
        for name in C.FROZEN_ORDER:
            pct_filled = con.execute(
                f"SELECT 100.0*count({name})/count(*) FROM wide").fetchone()[0]
            log(f"  {name:<24} {pct_filled:5.1f}%")
        pos, tot_rows = con.execute(
            f"SELECT sum({C.LEGACY_TARGET}), count(*) FROM wide").fetchone()
        log(f"`{C.LEGACY_TARGET}` (incumbent label, kept for verification): "
            f"{pos:,} positive of {tot_rows:,} "
            f"({100*pos/tot_rows:.2f}%)")
        con.close()

    account_parquet("Table 2 (wide time-series)", C.T2_WIDE_PQ)


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
