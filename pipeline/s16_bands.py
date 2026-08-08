"""Stage 16 -- risk bands and hysteresis: the pipeline's OUTPUT CONTRACT.

The four labels LOW / MEDIUM / HIGH / CRITICAL are not a display choice. They are
the interface the LLM + RAG explanatory layer consumes downstream, so this stage
produces a versioned artifact carrying the bands, the machine that assigns them,
and -- critically -- the measured event rate behind each name. A model handed the
bare string "HIGH" will invent what HIGH means and sound confident doing it; the
observed rate is what makes the resulting sentence checkable.

WHAT IS SOLVED HERE
-------------------
1. Three cuts, each set so that PROMOTIONS INTO that band cost no more than its
   budget -- C.BAND_PROMOTION_BUDGETS, currently 0.70 / 0.45 / 0.20 per
   ventilated patient-day. These are NOT C.ALERT_BUDGETS (1 / 2 / 4): s13
   anchored on OCCUPANCY, hours where the score sat above a threshold, so its
   published cuts differ and its numbers are not comparable to these. Carrying
   s13's figures across is exactly how this stage once produced a band table
   with 97% of readings in HIGH or CRITICAL. A patient held at HIGH for six
   hours is one interruption, not six, and interruptions are what alarm fatigue
   is about.
2. Hysteresis parameters, MEASURED rather than chosen: every combination in the
   sweep grid is evaluated and the winner MAXIMISES sensitivity at the top band,
   subject to median lost warning time under C.BAND_MAX_LOST_LEAD_MIN and the
   flip rate under its ceiling. Selecting on flip rate alone is confounded --
   each configuration re-solves its cuts to hold the budget, so hysteresis meets
   it at a lower cut, and lower cuts are crossed more often.
3. A rate envelope per band, derived from the calibration fold with a
   patient-level bootstrap and asserted on test. A retrain that moves HIGH's real
   rate outside its envelope fails the build rather than quietly changing what the
   explanatory layer is told HIGH means.

WHAT THIS STAGE CANNOT SOLVE -- READ BEFORE SHIPPING A DWELL
------------------------------------------------------------
The intended deployment reads a bedside ventilator at ~1 Hz and promotes after a
few seconds of sustained elevation. MIMIC-IV charts ventilator settings on an
irregular grid whose median interval is reported in the artifact and is measured
in TENS OF MINUTES. Two consequences, both recorded in the artifact rather than
buried here:

  * The dwell parameters fitted here are properties of the MIMIC grid. A
    10-second persistence requirement is far below the resolution of the only
    data available, so this stage can neither validate nor refute it. The
    deployment profile is DECLARED in the artifact, not calibrated.
  * The CUTS are properties of the calibrated score distribution and transfer
    better -- but only to the extent the input distribution does. At 1 Hz every
    ventilator parameter is measured every second, so `_delta_t_min` collapses to
    ~0 and `_observed` to 1 across the 33 documentation features that carry
    18.1% of this model's skill. That is a train/serve distribution shift this
    pipeline has not measured and this stage does not fix.

PROTOCOL
--------
Cuts and hysteresis are fitted on the CALIBRATION fold and reported on TEST. The
test fold already set s13's thresholds; tuning a second thing on it would spend
the same held-out data twice. The calibration fold is honest here -- the model
never saw those patients either.

Nothing is refitted. s13 persists no per-row predictions, so this stage reloads
the persisted model and calibrator and re-scores. That is exact, not approximate,
and the reproduction check turns it into a standing integrity test on models/.
"""
from __future__ import annotations

import json
import time
from itertools import product

import numpy as np

from . import bands as B
from . import config as C
from . import scoring
from .common import cached_stage, log, stage
from .s11_train import load, metrics
from .s13_calibrate import hour_key

REPORT_JSON = C.REPORTS / "bands.json"
CALIBRATION_JSON = C.REPORTS / "calibration.json"

N_BOOTSTRAP = 400          # patient-level, for the band-rate envelope
REPRO_TOL = 1e-9
SWEEP_STAYS = 1500         # admissions sampled for the parameter sweep
COARSE_POINTS = 25         # top-down scan before bisection refinement


# ---------------------------------------------------------------------------
# Row plumbing
# ---------------------------------------------------------------------------
def stay_groups(stay: np.ndarray, charttime: np.ndarray, mask: np.ndarray):
    """Row indices grouped by admission, each group sorted by charttime.

    Sorted explicitly rather than trusting the matrix order: the machine is a
    stepper, so one out-of-order reading corrupts every latch after it. Indices
    address the FULL arrays, so callers pass full-length p and y.

    minutes[i] is elapsed minutes since that stay's first reading; the machine
    reads only differences, so the origin is arbitrary if it is per-stay.
    """
    idx = np.flatnonzero(mask)
    idx = idx[np.lexsort((charttime[idx], stay[idx]))]
    s = stay[idx]
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    groups = [idx[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]
    secs = charttime.astype("datetime64[s]").astype(np.int64)
    minutes = [(secs[g] - secs[g][0]) / 60.0 for g in groups]
    return groups, minutes


def patient_days(stay: np.ndarray, charttime: np.ndarray, mask: np.ndarray) -> float:
    """Monitored patient-days on the same distinct-(stay, clock-hour) basis as
    s13_calibrate.hour_key, so a budget here and a budget there mean the same
    thing even though one counts promotions and the other occupancy."""
    return len(np.unique(hour_key(stay[mask], charttime[mask]))) / 24.0


def grid_interval_min(minutes) -> dict:
    """How often this cohort is actually charted.

    Reported in the artifact because every dwell below is denominated in it. The
    deployment grid is ~1 Hz; the gap between that and this is the single most
    important caveat on the fitted parameters.
    """
    gaps = np.concatenate([np.diff(mn) for mn in minutes if len(mn) > 1])
    return dict(median_min=float(np.median(gaps)),
                p25_min=float(np.percentile(gaps, 25)),
                p75_min=float(np.percentile(gaps, 75)),
                intervals=int(gaps.size))


def subsample(groups, minutes, n, seed=C.RANDOM_SEED):
    """A seeded subset of admissions for the parameter sweep.

    The sweep compares configurations against each other, which a subsample
    supports; the chosen configuration's cuts are then re-solved on the full
    fold. Logged rather than silent -- a bounded search that reads as exhaustive
    is how a coverage gap ships unnoticed.
    """
    if len(groups) <= n:
        return groups, minutes, len(groups)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(groups), n, replace=False)
    return [groups[i] for i in pick], [minutes[i] for i in pick], n


# ---------------------------------------------------------------------------
# Solving one cut against a promotion budget
# ---------------------------------------------------------------------------
def promotions(boundary: B.Boundary, p, groups, minutes, max_gap_min) -> int:
    """How many times this latch turns on across every admission."""
    n = 0
    for g, mins in zip(groups, minutes):
        prev = False
        for v in B.latch_series(boundary, p[g], mins, max_gap_min):
            if v and not prev:
                n += 1
            prev = v
    return n


def _rate_fn(p, groups, minutes, floor_below, cfg, days):
    def rate_at(t: float) -> float:
        b = B.Boundary(t, floor_below, cfg["promote_dwell_min"],
                       cfg["promote_min_readings"], cfg["demote_margin_frac"],
                       cfg["demote_dwell_min"])
        return promotions(b, p, groups, minutes, cfg["max_gap_min"]) / days
    return rate_at


def solve_cut(p, groups, minutes, *, budget, floor_below, cfg, days, grid) -> dict:
    """Lowest cut that fits the budget, taken from ABOVE the peak.

    The promotion rate is NOT monotone in the cut -- it is hump-shaped. Push the
    cut very low and every stay latches on once and never releases, so promotions
    fall to about one per admission; push it very high and almost nothing crosses,
    so they fall again. The maximum sits in between.

    A plain "lowest threshold that fits" rule -- correct for s13's monotone
    occupancy sweep -- would therefore happily return a near-zero cut that meets
    the budget by keeping every patient permanently at the top band. So the scan
    runs top-down and stops at the first cut that breaks the budget, which lands
    on the most sensitive cut on the high side of the peak. Bisection then refines
    between the last good point and the first bad one.
    """
    rate_at = _rate_fn(p, groups, minutes, floor_below, cfg, days)
    coarse = np.linspace(len(grid) - 1, 0, COARSE_POINTS).astype(int)

    good, bad, peak = None, None, 0.0
    for i in coarse:
        r = rate_at(grid[i])
        peak = max(peak, r)
        if r <= budget:
            good = i
        else:
            bad = i
            break
    # Both degenerate cases below used to return a cut anyway. They must not:
    # an unreachable budget silently yields the grid floor, where every stay
    # latches on once and never releases, which reads as "budget comfortably
    # met" while putting nearly every reading in the top band.
    if good is None:
        raise AssertionError(
            f"budget {budget:.2f}/patient-day is over-subscribed: even the "
            f"highest cut on the grid ({grid[-1]:.4f}) fires "
            f"{rate_at(grid[-1]):.3f}/day. Raise the budget or widen the grid.")
    if bad is None:
        # Not necessarily a bad configuration -- heavy hysteresis suppresses
        # re-promotions so thoroughly that the ceiling never binds anywhere on
        # the grid. But then the budget places no cut, and the scan would fall
        # through to the grid floor: a band holding almost every reading. Such a
        # configuration is un-placeable under this rule, not disqualified on
        # merit, and the sweep records it as rejected rather than guessing.
        raise AssertionError(
            f"budget {budget:.2f}/patient-day never binds: the promotion rate "
            f"peaks at {peak:.3f}/day across the whole grid, so no cut is "
            f"selected by it. Promotion budgets are not alert-hour budgets -- "
            f"if every configuration lands here, set BAND_PROMOTION_BUDGETS "
            f"below {peak:.3f}.")
    lo, hi = bad, good                  # lo < hi by construction (descending scan)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if rate_at(grid[mid]) <= budget:
            hi = mid
        else:
            lo = mid
    return dict(cut=float(grid[hi]), promotions_per_day=rate_at(grid[hi]),
                budget=budget, rate_just_below=rate_at(grid[lo]),
                cut_just_below=float(grid[lo]))


def rate_profile(p, groups, minutes, *, floor_below, cfg, days, grid) -> list[dict]:
    """Promotion rate across the grid, so the hump is visible in the report
    rather than being an assertion nobody can check."""
    out = []
    rate_at = _rate_fn(p, groups, minutes, floor_below, cfg, days)
    for q in (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98):
        t = grid[int(q * (len(grid) - 1))]
        out.append(dict(quantile=q, cut=float(t), promotions_per_day=rate_at(t)))
    return out


# ---------------------------------------------------------------------------
# Evaluating a fitted machine
# ---------------------------------------------------------------------------
def evaluate(machine: B.BandMachine, p, y, groups, minutes, days) -> dict:
    """Everything the selection rule and the artifact need, in one pass.

    `p` and `y` are FULL-LENGTH; `groups` indexes into them. Rows outside the
    fold stay at -1 so a mis-indexed caller shows up as a crash rather than as a
    silent pile of extra LOW readings.
    """
    n_bands = len(machine.names)
    high = n_bands - 2                     # HIGH is second from the top
    top = n_bands - 1

    disp = np.full(len(p), -1, dtype=np.int8)
    inst = np.full(len(p), -1, dtype=np.int8)
    flips = 0
    lost_all, lost_pos = [], []

    for g, mins in zip(groups, minutes):
        views = B.run(machine, p[g], mins)
        d = np.fromiter((v.displayed_index for v in views), np.int8, len(views))
        i = np.fromiter((v.instant_index for v in views), np.int8, len(views))
        disp[g], inst[g] = d, i
        flips += int((d[1:] != d[:-1]).sum())
        # Lost warning time: when the badge reached HIGH-or-above, against when
        # the raw score first did. Displayed can never lead instant -- a latch
        # only turns on after its cut has been crossed -- so this is >= 0.
        hit_i = np.flatnonzero(i >= high)
        if hit_i.size:
            hit_d = np.flatnonzero(d >= high)
            lost = (mins[hit_d[0]] if hit_d.size else mins[-1]) - mins[hit_i[0]]
            lost_all.append(lost)
            if y[g].max() == 1:
                lost_pos.append(lost)

    rows = np.concatenate(groups) if groups else np.array([], dtype=int)
    d_sub, y_sub = disp[rows], y[rows]
    band_rows = []
    for k in range(n_bands):
        mk = d_sub == k
        band_rows.append(dict(
            band=machine.names[k], index=k, rows=int(mk.sum()),
            share=float(mk.mean()) if d_sub.size else 0.0,
            observed_rate=float(y_sub[mk].mean()) if mk.any() else float("nan")))

    # Detection at the displayed band -- what the alarm actually catches. This
    # is the objective the sweep optimises, because at a fixed promotion budget
    # flips are confounded by the cut moving and this is not.
    pos = y_sub == 1
    sens = float((d_sub[pos] >= high).mean()) if pos.any() else float("nan")

    # Ratchet: does the top band ABSORB patients as a stay lengthens? Measured
    # as displayed-vs-instant top occupancy within stay-duration quartiles, so
    # the fact that longer stays are sicker -- and would legitimately spend more
    # time at the top -- is already in the denominator.
    dur = np.array([mn[-1] - mn[0] if len(mn) > 1 else 0.0 for mn in minutes])
    qcut = np.quantile(dur, [0.25, 0.50, 0.75]) if dur.size else np.zeros(3)
    qidx = np.searchsorted(qcut, dur, side="right")
    ratchet = []
    for k in range(4):
        sel = [g for g, qq in zip(groups, qidx) if qq == k]
        if not sel:
            continue
        rr = np.concatenate(sel)
        dtop, itop = float((disp[rr] == top).mean()), float((inst[rr] == top).mean())
        ratchet.append(dict(quartile=k, stays=len(sel), rows=int(rr.size),
                            median_hours=float(np.median(dur[qidx == k]) / 60.0),
                            displayed=dtop, instant=itop,
                            ratio=(dtop / itop) if itop > 0 else None))

    return dict(
        displayed=disp, instant=inst, rows=rows,
        sensitivity_high=sens,
        flips_per_day=flips / days,
        lost_lead_median_min=float(np.median(lost_all)) if lost_all else 0.0,
        lost_lead_p90_min=float(np.percentile(lost_all, 90)) if lost_all else 0.0,
        lost_lead_median_min_positive=float(np.median(lost_pos)) if lost_pos else 0.0,
        stays_reaching_high=len(lost_all),
        bands=band_rows, ratchet=ratchet,
        occupancy_top_displayed=float((d_sub == top).mean()) if d_sub.size else 0.0,
        occupancy_top_instant=float((inst[rows] == top).mean()) if rows.size else 0.0,
        base_rate=float(y_sub.mean()) if y_sub.size else float("nan"))


def bootstrap_band_rates(y, band_idx, subject_id, n_bands, *,
                         n=N_BOOTSTRAP, seed=C.RANDOM_SEED) -> list[dict]:
    """Patient-level bootstrap CI on each band's observed event rate.

    Same block-resampling method as s12_baselines.bootstrap_metric_diff -- rows
    within a patient are correlated, so a row-level bootstrap would give
    misleadingly tight intervals and an envelope too narrow to survive an honest
    retrain. That function returns a difference of two score vectors, so what is
    reused here is the method, not the call.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(subject_id, kind="stable")
    uniq, starts = np.unique(subject_id[order], return_index=True)
    bounds = np.append(starts, len(order))
    blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]

    draws: list[list[float]] = [[] for _ in range(n_bands)]
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        b, yy = band_idx[idx], y[idx]
        for k in range(n_bands):
            mk = b == k
            if mk.any():
                draws[k].append(float(yy[mk].mean()))
    out = []
    for k in range(n_bands):
        d = np.array(draws[k]) if draws[k] else np.array([float("nan")])
        out.append(dict(lo=float(np.percentile(d, 2.5)),
                        hi=float(np.percentile(d, 97.5)),
                        resamples=len(draws[k])))
    return out


# ---------------------------------------------------------------------------
def sweep_configs() -> list[dict]:
    """Every hysteresis combination in the grid.

    promote_min_readings is derived, not swept: 1 when no dwell is requested (so
    the all-zero config is a true no-hysteresis reference), otherwise 2, so one
    reading cannot satisfy a dwell purely by sitting next to a long charting gap.
    """
    return [dict(promote_dwell_min=pd_,
                 promote_min_readings=1 if pd_ == 0.0 else 2,
                 demote_margin_frac=mf, demote_dwell_min=dd,
                 max_gap_min=float(C.LOCF_CUTOFF_MIN))
            for pd_, mf, dd in product(C.BAND_PROMOTE_DWELL_MIN,
                                       C.BAND_DEMOTE_MARGIN_FRAC,
                                       C.BAND_DEMOTE_DWELL_MIN)]


def config_label(cfg: dict) -> str:
    return (f"up{cfg['promote_dwell_min']:.0f}m/{cfg['promote_min_readings']}r "
            f"margin{cfg['demote_margin_frac']:.2f} "
            f"down{cfg['demote_dwell_min']:.0f}m")


def build_machine(cuts, cfg) -> B.BandMachine:
    return B.BandMachine(
        names=C.BAND_NAMES,
        boundaries=B.boundaries_from_cuts(
            cuts, promote_dwell_min=cfg["promote_dwell_min"],
            promote_min_readings=cfg["promote_min_readings"],
            demote_margin_frac=cfg["demote_margin_frac"],
            demote_dwell_min=cfg["demote_dwell_min"]),
        max_gap_min=cfg["max_gap_min"])


def solve_all_cuts(p, groups, minutes, cfg, days, grid) -> tuple[list[float], list[dict]]:
    """Bottom-up: each cut's deadband is a fraction of the gap to the cut below,
    so MEDIUM must be solved before HIGH and HIGH before CRITICAL."""
    cuts, solved, floor = [], [], 0.0
    for budget in C.BAND_PROMOTION_BUDGETS:
        s = solve_cut(p, groups, minutes, budget=budget, floor_below=floor,
                      cfg=cfg, days=days, grid=grid)
        cuts.append(s["cut"])
        solved.append(s)
        floor = s["cut"]
    return cuts, solved


# ---------------------------------------------------------------------------
def main(force: bool = False) -> None:
    with cached_stage("s16_bands",
                      sources=[C.MODEL_MATRIX_PQ, C.FOLDS_PQ, C.MODEL_XGB,
                               C.CALIBRATOR_PKL, C.OPERATING_POINT_JSON],
                      output=C.BAND_TABLE_JSON, force=force,
                      extra=C.FP_BANDS) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    report: dict = {}
    t_start = time.time()

    # ------------------------------------------------------------- score
    with stage("Re-score from the persisted artifacts"):
        X, m = load("full")
        y, sub, stay, ct = m["y"], m["subject_id"], m["stay_id"], m["charttime"]
        is_train = m["split"] == "train"
        cal = is_train & (m["fold"] == C.CALIB_FOLD)
        te = m["split"] == "test"

        # Assertion 1 (feature order) and the device pin both live in
        # scoring.load_scorer -- the device is part of the model's identity, not
        # a runtime detail, and s17 must load it the same way. The reproduction
        # check below is what catches a silent CPU fallback.
        sc = scoring.load_scorer(expect_columns=X.columns)
        log(f"  scorer on [bold]{sc.device}[/bold], Platt slope {sc.slope:.4f}")

        p_all = np.zeros(len(y), dtype=np.float64)
        for nm, mask in (("calibrate", cal), ("test", te)):
            p_all[mask] = sc.score(X[mask])
            log(f"  scored {nm:<10} {int(mask.sum()):>9,} rows  "
                f"{len(np.unique(sub[mask])):>6,} patients")
        del X

    # ------------------------------------------------- reproduction check
    with stage("Reproduction check against reports/calibration.json"):
        pub = json.loads(CALIBRATION_JSON.read_text())["heads"]["H3_xgb_platt"]
        got = metrics(y[te], p_all[te])
        for k in ("average_precision", "roc_auc", "brier"):
            d = abs(got[k] - pub[k])
            # Assertion 2: the artifacts on disk must BE the ones that produced
            # the published numbers, not merely resemble them.
            assert d < REPRO_TOL, (
                f"{k} does not reproduce: {got[k]:.12f} vs published "
                f"{pub[k]:.12f} (delta {d:.2e}). models/ and reports/ disagree.")
            log(f"  {k:<20} {got[k]:.6f}  reproduces to {d:.1e}")
        report["reproduction"] = {k: got[k] for k in
                                  ("average_precision", "roc_auc", "brier")}

    # ------------------------------------------------------------- setup
    g_cal, t_cal = stay_groups(stay, ct, cal)
    g_te, t_te = stay_groups(stay, ct, te)
    d_cal, d_te = patient_days(stay, ct, cal), patient_days(stay, ct, te)
    grid = np.unique(np.quantile(p_all[cal],
                                 np.linspace(0.05, 0.99999, C.BAND_GRID_POINTS)))
    interval = grid_interval_min(t_cal)
    report["grid"] = interval
    log(f"calibration fold: {len(g_cal):,} admissions, {d_cal:,.0f} patient-days; "
        f"test: {len(g_te):,} admissions, {d_te:,.0f} patient-days")
    log(f"charting interval: median {interval['median_min']:.1f} min "
        f"(IQR {interval['p25_min']:.1f}-{interval['p75_min']:.1f}) -- every dwell "
        f"below is denominated in this, NOT in the ~1 Hz deployment grid")

    gs, ts, n_sub = subsample(g_cal, t_cal, SWEEP_STAYS)
    # Measured, not pro-rated: stays vary in length, so scaling d_cal by the
    # admission count would solve the sweep's cuts against a denominator the
    # final solve does not use.
    sub_mask = np.zeros(len(y), dtype=bool)
    sub_mask[np.concatenate(gs)] = True
    d_sub = patient_days(stay, ct, sub_mask)
    log(f"sweep runs on {n_sub:,} of {len(g_cal):,} admissions "
        f"(seeded); the chosen configuration is re-solved on all of them")

    # ------------------------------------------------------------- sweep
    configs = sweep_configs()
    with stage(f"Sweep {len(configs)} hysteresis configurations"):
        rows = []
        for ci, cfg in enumerate(configs, 1):
            t0 = time.time()
            try:
                cuts, solved = solve_all_cuts(p_all, gs, ts, cfg, d_sub, grid)
                machine = build_machine(cuts, cfg)
            except (ValueError, AssertionError) as e:
                # A config whose cuts collapse or whose budget cannot be met is
                # rejected, not silently repaired. If every config lands here the
                # budgets themselves are wrong -- surfaced below.
                log(f"  {ci:>2}/{len(configs)} {config_label(cfg):<36} "
                    f"[yellow]rejected: {e}[/yellow]")
                rows.append(dict(config=cfg, rejected=str(e)))
                continue
            ev = evaluate(machine, p_all, y, gs, ts, d_sub)
            rows.append(dict(config=cfg, cuts=cuts, solved=solved,
                             sensitivity_high=ev["sensitivity_high"],
                             flips_per_day=ev["flips_per_day"],
                             lost_lead_median_min=ev["lost_lead_median_min"],
                             lost_lead_p90_min=ev["lost_lead_p90_min"],
                             lost_lead_median_min_positive=ev[
                                 "lost_lead_median_min_positive"],
                             bands=ev["bands"], seconds=round(time.time() - t0, 1)))
            log(f"  {ci:>2}/{len(configs)} {config_label(cfg):<36} "
                f"cuts {'/'.join(f'{c:.3f}' for c in cuts)}  "
                f"sens {ev['sensitivity_high']:.3f}  "
                f"flips {ev['flips_per_day']:.3f}  "
                f"lost {ev['lost_lead_median_min']:.0f}m")
        report["sweep"] = rows

    # ------------------------------------------------------------ choose
    with stage("Select -- most detection at a fixed promotion budget"):
        usable = [r for r in rows if "rejected" not in r]
        assert usable, ("every hysteresis configuration was rejected; first "
                        f"reason: {rows[0]['rejected']}")
        ref = next(r for r in usable
                   if r["config"]["promote_dwell_min"] == 0.0
                   and r["config"]["demote_margin_frac"] == 0.0
                   and r["config"]["demote_dwell_min"] == 0.0)
        flip_ceiling = ref["flips_per_day"] * (1 + C.BAND_MAX_FLIP_INCREASE)
        ok = [r for r in usable
              if r["lost_lead_median_min"] <= C.BAND_MAX_LOST_LEAD_MIN
              and r["flips_per_day"] <= flip_ceiling]
        assert ok, (f"no configuration keeps median lost lead under "
                    f"{C.BAND_MAX_LOST_LEAD_MIN:.0f} min and flips under "
                    f"{flip_ceiling:.3f}/day")
        # Ties broken toward fewer flips, then less delay -- so a configuration
        # only carries added flicker if it buys detection with it.
        best = max(ok, key=lambda r: (r["sensitivity_high"],
                                      -r["flips_per_day"],
                                      -r["lost_lead_median_min"]))
        # Assertion 3: whatever ships must catch at least as much as no
        # hysteresis does at the same alert cost. Hysteresis that costs
        # detection is pure added delay.
        assert best["sensitivity_high"] >= ref["sensitivity_high"], (
            f"hysteresis loses detection: {best['sensitivity_high']:.4f} vs "
            f"reference {ref['sensitivity_high']:.4f} at the same budget")
        d_sens = best["sensitivity_high"] - ref["sensitivity_high"]
        d_flip = best["flips_per_day"] - ref["flips_per_day"]
        log(f"  chosen    {config_label(best['config'])}")
        log(f"  detection at >= {C.BAND_NAMES[-2]}: "
            f"{ref['sensitivity_high']:.4f} -> {best['sensitivity_high']:.4f} "
            f"({d_sens:+.4f}) at the same promotion budget")
        log(f"  cuts      {'/'.join(f'{c:.4f}' for c in ref['cuts'])} -> "
            f"{'/'.join(f'{c:.4f}' for c in best['cuts'])}")
        log(f"  flips/day {ref['flips_per_day']:.3f} -> "
            f"{best['flips_per_day']:.3f} ({d_flip:+.3f}, "
            f"ceiling {flip_ceiling:.3f})")
        log(f"  median lost warning time {best['lost_lead_median_min']:.0f} min "
            f"(cap {C.BAND_MAX_LOST_LEAD_MIN:.0f}); on label-positive stays "
            f"{best['lost_lead_median_min_positive']:.0f} min")
        report["selection"] = dict(
            chosen=best["config"], reference=ref["config"],
            reference_cuts=ref["cuts"], chosen_cuts=best["cuts"],
            reference_sensitivity=ref["sensitivity_high"],
            chosen_sensitivity=best["sensitivity_high"],
            reference_flips_per_day=ref["flips_per_day"],
            chosen_flips_per_day=best["flips_per_day"],
            flip_ceiling=flip_ceiling,
            rule=(f"max detection at >= {C.BAND_NAMES[-2]} subject to median "
                  f"lost lead <= {C.BAND_MAX_LOST_LEAD_MIN:.0f} min and flips "
                  f"<= {1 + C.BAND_MAX_FLIP_INCREASE:.2f}x the no-hysteresis "
                  f"reference"),
            swept_admissions=n_sub, total_admissions=len(g_cal))

    # ------------------------------------------- re-solve on the full fold
    cfg = best["config"]
    with stage("Re-solve the chosen cuts on the whole calibration fold"):
        cuts, solved = solve_all_cuts(p_all, g_cal, t_cal, cfg, d_cal, grid)
        machine = build_machine(cuts, cfg)
        profiles, floor = [], 0.0
        for bk, s in enumerate(solved):
            log(f"  {C.BAND_NAMES[bk + 1]:<9} cut {s['cut']:.4f}  "
                f"{s['promotions_per_day']:.3f} promotions/patient-day "
                f"(budget {s['budget']:.2f})")
            # Assertion 4: the cut must be the most sensitive one that fits --
            # the next grid point down has to break the budget, or the top-down
            # scan stopped somewhere it should not have.
            if "rate_just_below" in s:
                assert s["rate_just_below"] > s["budget"], (
                    f"{C.BAND_NAMES[bk + 1]} cut is not the most sensitive that "
                    f"fits: {s['cut_just_below']:.4f} gives "
                    f"{s['rate_just_below']:.3f}/day, within budget "
                    f"{s['budget']:.0f}")
            profiles.append(dict(
                band=C.BAND_NAMES[bk + 1],
                profile=rate_profile(p_all, gs, ts, floor_below=floor, cfg=cfg,
                                     days=d_sub, grid=grid)))
            floor = s["cut"]
        report["cuts"] = solved
        report["rate_profiles"] = profiles

    # -------------------------------------------- does hysteresis do its job?
    with stage("Isolation -- the same cuts, with and without hysteresis"):
        # The sweep cannot answer this: every configuration there re-solves its
        # cuts, so a flip comparison mixes the machine's effect with the cut's.
        # Holding the cuts fixed is the only way to see what hysteresis alone
        # does to the display.
        off = dict(cfg, promote_dwell_min=0.0, promote_min_readings=1,
                   demote_margin_frac=0.0, demote_dwell_min=0.0)
        ev_off = evaluate(build_machine(cuts, off), p_all, y, g_cal, t_cal, d_cal)
        ev_on = evaluate(machine, p_all, y, g_cal, t_cal, d_cal)
        red = 100 * (1 - ev_on["flips_per_day"] / ev_off["flips_per_day"])
        log(f"  cuts held at {'/'.join(f'{c:.4f}' for c in cuts)}")
        log(f"  flips/day   {ev_off['flips_per_day']:.3f} (off) -> "
            f"{ev_on['flips_per_day']:.3f} (on)   {red:.1f}% fewer")
        log(f"  detection   {ev_off['sensitivity_high']:.4f} -> "
            f"{ev_on['sensitivity_high']:.4f}")
        # Assertion 3b: this is the claim hysteresis actually makes, in the one
        # comparison where it is not confounded. If it fails here the machine is
        # not doing what it is for.
        assert ev_on["flips_per_day"] < ev_off["flips_per_day"], (
            f"at fixed cuts hysteresis does not reduce flips: "
            f"{ev_on['flips_per_day']:.3f} vs {ev_off['flips_per_day']:.3f}")
        report["isolation"] = dict(
            cuts=cuts, flips_off=ev_off["flips_per_day"],
            flips_on=ev_on["flips_per_day"], flip_reduction_pct=red,
            sensitivity_off=ev_off["sensitivity_high"],
            sensitivity_on=ev_on["sensitivity_high"],
            note=("Cuts held fixed so the comparison isolates the machine. In "
                  "the sweep each configuration re-solves its cuts to hold the "
                  "promotion budget, which confounds any flip comparison."))

    # ------------------------------------------------------------ envelope
    with stage("Rate envelope from the calibration fold"):
        ev_cal = ev_on            # already computed by the isolation check
        cis = bootstrap_band_rates(y[ev_cal["rows"]],
                                   ev_cal["displayed"][ev_cal["rows"]],
                                   sub[ev_cal["rows"]], len(C.BAND_NAMES))
        envelope = []
        for k, (row, ci) in enumerate(zip(ev_cal["bands"], cis)):
            lo = max(0.0, ci["lo"] - C.BAND_RATE_TOLERANCE)
            hi = min(1.0, ci["hi"] + C.BAND_RATE_TOLERANCE)
            envelope.append(dict(band=C.BAND_NAMES[k], lo=lo, hi=hi,
                                 calibration_rate=row["observed_rate"]))
            log(f"  {C.BAND_NAMES[k]:<9} calibration rate "
                f"{row['observed_rate']:.4f}   envelope [{lo:.4f}, {hi:.4f}]")
        report["envelope"] = envelope

    # ---------------------------------------------------------------- test
    with stage("Freeze and report on the test fold"):
        ev_te = evaluate(machine, p_all, y, g_te, t_te, d_te)
        base = float(y[te].mean())
        rates = [r["observed_rate"] for r in ev_te["bands"]]

        # Assertion 5: the names have to mean what they say.
        assert all(a < b for a, b in zip(rates, rates[1:])), \
            f"observed rate is not increasing across bands: {rates}"
        # Assertion 6: a band nobody is ever in is a label the explanatory layer
        # can never ground.
        assert all(r["rows"] > 0 for r in ev_te["bands"]), \
            f"empty band: {[(r['band'], r['rows']) for r in ev_te['bands']]}"
        # Assertion 7: test rates must land inside the calibration envelope.
        for r, e in zip(ev_te["bands"], envelope):
            assert e["lo"] <= r["observed_rate"] <= e["hi"], (
                f"{r['band']} observed rate {r['observed_rate']:.4f} is outside "
                f"its envelope [{e['lo']:.4f}, {e['hi']:.4f}] -- the band no "
                f"longer means what the schema says it means")
        # Assertion 8: the top band must not ABSORB. Inflation on its own is the
        # mechanism, not a fault -- demote-slow holds the badge up on purpose.
        # What would be a fault is that hold growing with stay length, which is
        # a band nobody ever leaves.
        sat = (ev_te["occupancy_top_displayed"]
               / max(ev_te["occupancy_top_instant"], 1e-12))
        rr = [r for r in ev_te["ratchet"] if r["ratio"]]
        assert len(rr) >= 2, "not enough stay-duration quartiles to test ratcheting"
        growth = rr[-1]["ratio"] / rr[0]["ratio"]
        assert growth <= C.BAND_MAX_RATCHET, (
            f"{C.BAND_NAMES[-1]} ratchets: hysteresis inflates its occupancy "
            f"{rr[0]['ratio']:.2f}x in the shortest stays but "
            f"{rr[-1]['ratio']:.2f}x in the longest ({growth:.2f}x growth, "
            f"ceiling {C.BAND_MAX_RATCHET}) -- the band is absorbing")
        for r in ev_te["ratchet"]:
            log(f"  ratchet q{r['quartile'] + 1} "
                f"(median {r['median_hours']:>5.1f} h, {r['stays']:>5,} stays)  "
                f"instant {r['instant']:.4f} -> displayed {r['displayed']:.4f}  "
                f"{r['ratio']:.2f}x")
        # Assertion 9: most monitored time must be unremarkable. Assertions 5-8
        # all passed on a table that put 97% of readings in the top two bands,
        # because rising rates, non-empty bands and a self-derived envelope are
        # all satisfiable by a degenerate fit. This is the one that would have
        # caught it.
        low = ev_te["bands"][0]["share"]
        assert low >= C.BAND_MIN_LOW_SHARE, (
            f"only {100 * low:.1f}% of readings are {C.BAND_NAMES[0]} "
            f"(floor {100 * C.BAND_MIN_LOW_SHARE:.0f}%) -- a monitor whose "
            f"quiet state is rare is not a monitor")

        for r, e in zip(ev_te["bands"], envelope):
            log(f"  {r['band']:<9} {100 * r['share']:>5.1f}% of readings   "
                f"observed {r['observed_rate']:.4f}   "
                f"{r['observed_rate'] / base:>5.2f}x base   "
                f"envelope [{e['lo']:.3f}, {e['hi']:.3f}]")
        log(f"  flips/day {ev_te['flips_per_day']:.3f}   "
            f"{C.BAND_NAMES[-1]} occupancy inflation {sat:.2f}x   "
            f"base rate {base:.4f}")
        report["test"] = dict(
            base_rate=base, sensitivity_high=ev_te["sensitivity_high"],
            flips_per_day=ev_te["flips_per_day"],
            lost_lead_median_min=ev_te["lost_lead_median_min"],
            lost_lead_p90_min=ev_te["lost_lead_p90_min"],
            lost_lead_median_min_positive=ev_te["lost_lead_median_min_positive"],
            stays_reaching_high=ev_te["stays_reaching_high"],
            saturation=sat, ratchet=ev_te["ratchet"],
            ratchet_growth=growth, patient_days=d_te,
            bands=[dict(r, lift=r["observed_rate"] / base)
                   for r in ev_te["bands"]])

    # ------------------------------------------------------------- persist
    with stage("Persist the band table and output contract"):
        art = build_artifact(machine, ev_te, envelope, base, interval)
        C.BAND_TABLE_JSON.write_text(json.dumps(art, indent=2), encoding="utf-8")
        report["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report["seconds"] = round(time.time() - t_start, 1)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=float),
                               encoding="utf-8")
        for p in (C.BAND_TABLE_JSON, REPORT_JSON):
            log(f"  {p.name:<26} {p.stat().st_size / 1e3:,.1f} kB")


# ---------------------------------------------------------------------------
def build_artifact(machine, ev_te, envelope, base_rate, interval) -> dict:
    """The versioned contract: band table, machine, and the record schema.

    Schema and band table ship in ONE file because the schema describes the
    table; splitting them lets them drift. Kept separate from
    operating_point_*.json because s13 rewrites that file, and anything merged
    into it would be clobbered on the next training run.

    The example record is SYNTHETIC. A real row keyed to a real stay_id would be
    a per-patient MIMIC extract, which the PhysioNet DUA does not permit leaving
    this machine.
    """
    telemetry_fields = {
        p: {"value": f"{p}_final", "age_min": f"{p}_delta_t_min",
            "measured": f"{p}_observed"}
        for p in C.FROZEN_PARAMS}

    return {
        "schema_version": C.RISK_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": {
            "model": C.MODEL_XGB.name,
            "calibrator": C.CALIBRATOR_PKL.name,
            "operating_point": C.OPERATING_POINT_JSON.name,
            "label": C.TARGET,
            "arm": "respiratory",
            "horizon_hours": C.PRIMARY_HORIZON_H,
            "base_rate": base_rate,
            "scoring_device": "cuda" if C.USE_GPU else "cpu",
            "device_warning": (
                "These cuts are only valid for scores produced on the device "
                "named above. XGBoost's CPU and CUDA predictors disagree on this "
                "model by 4.85e-03 on the raw score on average and 1.25e-01 at "
                "worst (417k of 765k test rows), which is large next to cuts of "
                "this size. Serving on the other device shifts every band."),
            "note": "Every band is a property of (model, label, cohort). "
                    "Re-derive whenever any of the three moves.",
        },
        "fitted_on": {
            "cohort": "MIMIC-IV 3.1, calibration fold",
            "charting_interval_min": interval,
            "warning": (
                "Dwell parameters are denominated in THIS grid. The deployment "
                "grid is ~1 Hz, three orders of magnitude finer. See "
                "deployment_profile before shipping any dwell."),
        },
        "deployment_profile": {
            "status": "ILLUSTRATIVE -- shape of the problem, not a committed rate",
            "example_sampling_hz": C.BAND_DEPLOY_HZ,
            "example_promote_dwell_min": C.BAND_DEPLOY_PROMOTE_SEC / 60.0,
            "example_promote_min_readings": int(C.BAND_DEPLOY_PROMOTE_SEC
                                                * C.BAND_DEPLOY_HZ),
            "why_not_calibrated": (
                f"MIMIC charts ventilator settings every "
                f"{interval['median_min']:.0f} min at the median. Any persistence "
                f"requirement shorter than that is below the resolution of the "
                f"only data available, so this dataset can neither validate nor "
                f"refute it. Whatever the bedside rate turns out to be, the dwell "
                f"must be re-expressed for it -- Boundary carries both a time and "
                f"a reading count so the same machine covers either grid."),
            "open_risk": (
                "On a grid fine enough that every ventilator parameter is measured "
                "every tick, _delta_t_min collapses to ~0 and _observed to 1 "
                "across the 33 documentation features that carry 18.1% of this "
                "model's skill. That train/serve shift is unmeasured, and the "
                "cuts below are fitted on the charted grid, not the bedside one."),
        },
        "machine": machine.to_json(),
        "bands": [
            {"band": r["band"], "index": r["index"],
             "floor": (0.0 if r["index"] == 0
                       else machine.boundaries[r["index"] - 1].cut),
             "share_of_readings": r["share"],
             "observed_rate": r["observed_rate"],
             "lift_vs_base_rate": r["observed_rate"] / base_rate,
             "envelope": [e["lo"], e["hi"]],
             "budget_promotions_per_patient_day": (
                 None if r["index"] == 0 else C.BAND_PROMOTION_BUDGETS[r["index"] - 1])}
            for r, e in zip(ev_te["bands"], envelope)],
        "apply": (
            "p_calibrated = sigmoid(a * logit(p_raw) + b); feed p_calibrated to "
            "pipeline.bands.BandStepper.push(p, t_minutes) once per reading, one "
            "stepper per stay. band.displayed is what a clinician sees; "
            "band.instant is the raw crossing. Do NOT re-derive the band from a "
            "threshold comparison -- that discards the hysteresis and "
            "reintroduces the flicker this table exists to remove."),
        "record_schema": {
            "schema_version": "string, matches this file",
            "provenance": "object, copied from this file -- an explanation must "
                          "never be attachable to a score from another model",
            "risk": {"calibrated": "float [0,1], a probability",
                     "is_probability": "bool, always true post-calibration"},
            "band": {
                "displayed": f"enum {list(C.BAND_NAMES)} -- post-hysteresis",
                "instant": f"enum {list(C.BAND_NAMES)} -- raw crossing",
                "state": f"enum {list(B.STATES)}; 'confirmed' iff displayed == instant",
                "readings_in_state": "int, consecutive readings at this band",
                "observed_rate": "float, measured rate for this band -- the anchor "
                                 "for any statement about what the band means",
                "base_rate": "float, cohort rate for comparison",
                "lift": "float, observed_rate / base_rate",
                "envelope": "[lo, hi], the range observed_rate may occupy"},
            "telemetry": {
                "_shape": "one entry per frozen parameter: "
                          "{value, age_min, measured, source}",
                "_why_age": "a LOCF-carried value handed over without its age "
                            "becomes 'FiO2 is currently 60%' -- false, and stated "
                            "confidently. age_min is how old the value in use is.",
                "_source": (
                    "'measured' | 'carried_forward' | 'population_reference'. "
                    "THREE states, not two. `value` is the *_final column, which "
                    "s07_impute defines as *_locf filled from a COHORT REFERENCE "
                    "where the patient has no value at all -- 20.4% of PEEP "
                    "readings, 13.8% of FiO2. Those are population statistics, "
                    "not observations of this patient, and must never be "
                    "narrated as measurements. Where _locf exists, _final is "
                    "bit-identical to it."),
                "_source_columns": telemetry_fields},
            "reasons": "list of {code, ...}. Enumerated codes, never prose: it "
                       "keeps the model band and any hard safety rule separately "
                       "attributable and gives retrieval a stable key.",
            "contributors": {
                "_shape": f"top {C.CONTRIB_TOP_K} by |contribution|, each "
                          "{feature, value, contribution, kind, group}",
                "_space": "contribution is in CALIBRATED-LOGIT space: "
                          "sigmoid(sum(contribution) + contributors_bias) equals "
                          "risk.calibrated exactly. Raw TreeSHAP attributes the "
                          "pre-calibration margin; it is rescaled by the Platt "
                          "slope, which is exact because Platt is affine in the "
                          "margin.",
                "_kind": "'physiology' | 'documentation'. THE DISTINCTION IS "
                         "LOAD-BEARING. The 33 documentation features "
                         "(_observed, _delta_t_min, _structurally_missing_in_stay) "
                         "carry 18.1% of this model's skill, so they genuinely "
                         "appear near the top. They are labelled rather than "
                         "hidden, because dropping them would misrepresent the "
                         "model -- but an explanation that narrates "
                         "'spo2_delta_t_min' as a clinical finding is reporting a "
                         "charting artifact as physiology. Never let generated "
                         "text describe a documentation contributor as a change "
                         "in the patient.",
                "_group": "source table from build/features.json: t1_static, "
                          "t2_all, t3_interventions",
            },
            "contributors_other": f"float, the summed contribution of every "
                                  f"feature OUTSIDE the top {C.CONTRIB_TOP_K}. "
                                  f"Present so the record is self-verifying: "
                                  f"sigmoid(sum(contributors) + "
                                  f"contributors_other + contributors_bias) "
                                  f"equals risk.calibrated. Without it a "
                                  f"consumer holding only the top-k cannot tell "
                                  f"a truncated explanation from a wrong one.",
            "contributors_bias": "float, the attribution baseline; include it "
                                 "when summing",
            "documentation_share": "float, this reading's share of total "
                                   "|attribution| coming from documentation "
                                   "features -- the honest caveat on any "
                                   "explanation built from this record",
        },
        "example_record": {
            "_note": "SYNTHETIC. Shape only; no MIMIC row leaves this machine.",
            "schema_version": C.RISK_SCHEMA_VERSION,
            "provenance": {"model": C.MODEL_XGB.name, "label": C.TARGET,
                           "arm": "respiratory",
                           "horizon_hours": C.PRIMARY_HORIZON_H},
            "risk": {"calibrated": 0.2143, "is_probability": True},
            "band": {"displayed": "HIGH", "instant": "HIGH", "state": B.CONFIRMED,
                     "readings_in_state": 3, "observed_rate": 0.31,
                     "base_rate": 0.09, "lift": 3.4, "envelope": [0.20, 0.45]},
            "telemetry": {
                "fio2": {"value": 60.0, "age_min": 12, "measured": True,
                         "source": "measured"},
                "peep": {"value": 12.0, "age_min": 184, "measured": False,
                         "source": "carried_forward"},
                "etco2": {"value": 34.0, "age_min": None, "measured": False,
                          "source": "population_reference"}},
            "reasons": [{"code": "MODEL_BAND", "band": "HIGH"}],
            "contributors": [
                {"feature": "fio2_final", "value": 60.0, "contribution": 0.41,
                 "kind": "physiology", "group": "t2_all"},
                {"feature": "spo2_delta_t_min", "value": 184.0,
                 "contribution": 0.22, "kind": "documentation",
                 "group": "t2_all"},
                {"feature": "tidal_volume_ml_per_kg_pbw", "value": 9.1,
                 "contribution": -0.13, "kind": "physiology",
                 "group": "t2_all"}],
            "contributors_other": -0.19,
            "contributors_bias": -2.41,
            "documentation_share": 0.19,
        },
    }


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="re-fit even when the manifest is current -- needed "
                          "after a code change, which no fingerprint covers")
    main(force=_ap.parse_args().force)
