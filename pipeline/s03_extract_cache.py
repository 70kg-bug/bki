"""Stage 3 -- THE ONLY 42 GB SCAN.

Streams chartevents.csv once and writes a compact long-format Parquet cache.

Deliberately filtered by *itemid only*, not by cohort: keeping the cache
cohort-independent means changing the cohort definition later re-runs a
seconds-long stage instead of another full scan.
"""
from __future__ import annotations

from . import config as C
from .common import (account, account_parquet, cached_stage, connect_duckdb,
                     heartbeat, log)

# Types match what DuckDB's sniffer reports, so the reader never has to re-plan;
# the narrowing casts happen in the projection below instead.
CHARTEVENTS_COLUMNS = {
    "subject_id": "BIGINT",
    "hadm_id": "BIGINT",
    "stay_id": "BIGINT",
    "caregiver_id": "BIGINT",
    "charttime": "TIMESTAMP",
    "storetime": "TIMESTAMP",
    "itemid": "BIGINT",
    "value": "VARCHAR",
    "valuenum": "DOUBLE",
    "valueuom": "VARCHAR",
    "warning": "BIGINT",
}


def main(force: bool = False) -> None:
    with cached_stage("s03_extract_cache", sources=[C.CHARTEVENTS],
                      output=C.TS_LONG_PQ, force=force) as ran:
        if not ran:
            return

        con = connect_duckdb()
        cols_sql = ", ".join(f"'{k}': '{v}'" for k, v in CHARTEVENTS_COLUMNS.items())
        items_sql = ", ".join(str(i) for i in C.CACHE_ITEMIDS)

        log(f"scanning {C.CHARTEVENTS.stat().st_size/1e9:.1f} GB for "
            f"{len(C.CACHE_ITEMIDS)} itemids ({len(C.FROZEN_PARAMS)} frozen "
            f"+ {len(C.EXTRA_CACHED_ITEMS)} stored-not-trained)")
        log("this is the only pass over the source file; everything downstream reads the cache")

        sql = f"""
        COPY (
            SELECT CAST(stay_id AS INTEGER)    AS stay_id,
                   CAST(subject_id AS INTEGER) AS subject_id,
                   CAST(hadm_id AS INTEGER)    AS hadm_id,
                   charttime,
                   CAST(itemid AS INTEGER)     AS itemid,
                   CAST(valuenum AS FLOAT)     AS valuenum,
                   value,
                   CAST(COALESCE(warning, 0) AS TINYINT) AS warning
            FROM read_csv('{C.CHARTEVENTS.as_posix()}',
                          header = true,
                          columns = {{{cols_sql}}},
                          quote = '"',
                          escape = '"',
                          parallel = true)
            WHERE itemid IN ({items_sql})
              AND stay_id IS NOT NULL
        ) TO '{C.TS_LONG_PQ.as_posix()}'
          (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000);
        """

        with heartbeat("chartevents scan", watch=C.TS_LONG_PQ, every=20.0):
            con.execute(sql)
        con.close()

    # Reporting lives OUTSIDE the cached_stage body on purpose: a bug in the
    # summary must never be able to delete a cache that took a full scan to build.
    _report()


def _report() -> None:
    con = connect_duckdb()
    account_parquet("cache (ts_long.parquet)", C.TS_LONG_PQ)
    rows = con.execute(f"""
        SELECT itemid, count(*) AS n, count(DISTINCT stay_id) AS stays,
               sum(warning) AS warns
        FROM read_parquet('{C.TS_LONG_PQ.as_posix()}')
        GROUP BY itemid ORDER BY n DESC
    """).fetchall()
    log("per-item cache contents ([bold]*[/bold] = one of the frozen eleven):")
    for itemid, n, stays, warns in rows:
        name = C.ITEMID_TO_NAME.get(itemid, "?")
        frozen = "*" if itemid in C.FROZEN_PARAMS.values() else " "
        log(f"  {frozen} {itemid}  {name:<24} {n:>12,}  stays={stays:>7,}  "
            f"warning=1: {warns:>10,}")
    missing = set(C.CACHE_ITEMIDS) - {r[0] for r in rows}
    if missing:
        log(f"[yellow]no rows for itemids: {sorted(missing)}[/yellow]")
    con.close()


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
