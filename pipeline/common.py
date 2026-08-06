"""Shared helpers: DuckDB setup, progress reporting, manifests, row/stay accounting."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
from rich.console import Console
from rich.table import Table

from . import config as C

console = Console(highlight=False, soft_wrap=True)

_T0 = time.time()


def _hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def log(msg: str, style: str = "") -> None:
    console.print(f"[dim]{_hms(time.time() - _T0)}[/dim]  {msg}", style=style)


@contextmanager
def stage(name: str):
    """Print a banner, time the block, report success or failure honestly."""
    console.rule(f"[bold cyan]{name}")
    t = time.time()
    try:
        yield
    except BaseException:
        console.print(f"[red]FAILED[/red] {name} [dim]after {_hms(time.time() - t)}[/dim]\n")
        raise
    console.print(f"[green]DONE[/green] {name} [dim]in {_hms(time.time() - t)}[/dim]\n")


@contextmanager
def heartbeat(label: str, watch: Path | None = None, every: float = 15.0):
    """Emit a progress line every `every` seconds while a long op runs.

    DuckDB gives no usable callback for a streaming COPY, so we report elapsed
    time and (when given) the growth of the output file instead of guessing.
    """
    stop = threading.Event()

    def _tick():
        t = time.time()
        while not stop.wait(every):
            extra = ""
            if watch is not None and watch.exists():
                extra = f" | output {watch.stat().st_size / 1e6:,.0f} MB"
            console.print(f"  [dim]... {label}: {_hms(time.time() - t)} elapsed{extra}[/dim]")

    th = threading.Thread(target=_tick, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=1)


def connect_duckdb(read_only_pragmas: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{C.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={C.DUCKDB_THREADS}")
    con.execute(f"SET temp_directory='{C.DUCKDB_TEMP_DIR.as_posix()}'")
    try:
        con.execute(f"SET max_temp_directory_size='{C.DUCKDB_MAX_TEMP}'")
    except duckdb.Error:
        pass  # older builds do not expose this knob
    con.execute("SET preserve_insertion_order=false")  # lets the CSV reader stream
    return con


# --------------------------------------------------------------------------
# Accounting -- rows AND distinct admissions, at every stage.
# Distinct admissions is the number this whole pipeline optimises, so it is
# never reported implicitly.
# --------------------------------------------------------------------------
_ACCOUNTING: list[dict[str, Any]] = []


def account(label: str, *, rows: int, stays: int | None = None,
            subjects: int | None = None, note: str = "") -> None:
    _ACCOUNTING.append(dict(label=label, rows=rows, stays=stays,
                            subjects=subjects, note=note))
    bits = [f"rows={rows:,}"]
    if stays is not None:
        bits.append(f"admissions={stays:,}")
    if subjects is not None:
        bits.append(f"patients={subjects:,}")
    if note:
        bits.append(note)
    log(f"[bold]{label}[/bold]  " + "  ".join(bits))


def parquet_columns(path: Path) -> list[str]:
    con = connect_duckdb()
    try:
        cur = con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0")
        return [d[0] for d in cur.description]
    finally:
        con.close()


def account_parquet(label: str, path: Path, stay_col: str = "stay_id",
                    subject_col: str | None = "subject_id") -> None:
    con = connect_duckdb()
    cur = con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0")
    cols = {d[0] for d in cur.description}
    sel = ["count(*)"]
    sel.append(f"count(DISTINCT {stay_col})" if stay_col in cols else "NULL")
    sel.append(f"count(DISTINCT {subject_col})" if subject_col and subject_col in cols else "NULL")
    rows, stays, subs = con.execute(
        f"SELECT {', '.join(sel)} FROM read_parquet('{path.as_posix()}')").fetchone()
    con.close()
    account(label, rows=rows, stays=stays, subjects=subs,
            note=f"{path.stat().st_size / 1e6:,.0f} MB")


def accounting_table() -> None:
    t = Table(title="Row / admission accounting", header_style="bold")
    t.add_column("Stage"); t.add_column("Rows", justify="right")
    t.add_column("Admissions", justify="right"); t.add_column("Patients", justify="right")
    t.add_column("Note")
    for r in _ACCOUNTING:
        t.add_row(r["label"], f"{r['rows']:,}",
                  f"{r['stays']:,}" if r["stays"] is not None else "-",
                  f"{r['subjects']:,}" if r["subjects"] is not None else "-",
                  r["note"])
    console.print(t)


# --------------------------------------------------------------------------
# Manifests -- so a re-run recomputes only what actually moved.
# --------------------------------------------------------------------------
def _file_fingerprint(p: Path) -> str:
    """Cheap identity: size + mtime. Hashing 42 GB per run would defeat the point."""
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def config_hash(*extra: Any) -> str:
    payload = json.dumps({
        "cache_itemids": C.CACHE_ITEMIDS,
        "frozen": C.FROZEN_PARAMS,
        "frozen_sources": C.FROZEN_PARAM_SOURCES,
        "ranges": C.PLAUSIBLE_RANGES,
        "locf_cutoff": C.LOCF_CUTOFF_MIN,
        "extra": [str(e) for e in extra],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_manifest(name: str, *, sources: list[Path], output: Path, **fields: Any) -> None:
    m = {
        "stage": name,
        "config_hash": config_hash(),
        "sources": {str(s): _file_fingerprint(s) for s in sources if s.exists()},
        "output": str(output),
        "output_bytes": output.stat().st_size if output.exists() else None,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    (C.MANIFEST_DIR / f"{name}.json").write_text(json.dumps(m, indent=2, default=str))


def is_current(name: str, *, sources: list[Path], output: Path) -> bool:
    """True when the stage's output exists and its inputs/config are unchanged."""
    mf = C.MANIFEST_DIR / f"{name}.json"
    if not (mf.exists() and output.exists()):
        return False
    try:
        m = json.loads(mf.read_text())
    except json.JSONDecodeError:
        return False
    if m.get("config_hash") != config_hash():
        return False
    want = {str(s): _file_fingerprint(s) for s in sources if s.exists()}
    return m.get("sources") == want


@contextmanager
def cached_stage(name: str, *, sources: list[Path], output: Path, force: bool = False):
    """Skip the body when the output is already current. Yields True if it ran."""
    if not force and is_current(name, sources=sources, output=output):
        log(f"[yellow]SKIP[/yellow] {name} -- output current "
            f"({output.name}, {output.stat().st_size / 1e6:,.0f} MB)")
        yield False
        return
    try:
        with stage(name):
            yield True
    except BaseException:
        # A half-written Parquet must never be mistaken for a finished one.
        if output.exists():
            output.unlink(missing_ok=True)
            log(f"[yellow]removed partial output {output.name}[/yellow]")
        raise
    write_manifest(name, sources=sources, output=output)


def scan(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)
