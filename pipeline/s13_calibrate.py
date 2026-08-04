"""Stage 13 -- rank vs. level: is the [0,1] risk score or the AP ranking better?

They are not alternatives. AP is *computed from* the score. The real distinction
is what part of the score you rely on:

  RANK   AP and ROC-AUC ask "are sicker patients above healthier ones?" Both are
         invariant under ANY monotone rescaling of the output.
  LEVEL  bki/backend/main.py:85 applies `risk_prob > 0.70`. That is a test on
         the level, and it is not invariant to anything.

Every number reported so far measured rank. Nothing measured level, and level is
where the models fail. A constant forecast of the base rate scores Brier
p(1-p) = 0.0499; the shipped XGBoost scores 0.0804 -- worse than a constant,
despite ROC-AUC 0.9512. The cause is visible in the code rather than inferred:
`scale_pos_weight ~= 18` (s11_train.py:96, :113) trains against a reweighted
prior and nothing ever converts the output back.

Eight heads are compared on identical patients, features and folds. Two are the
question as literally posed: an XGBoost regressor emitting [0,1] (H5), and the
bki classifier -> calibrator -> regressor chain (H6, H7).

Squared-error regression on a 0/1 target IS Brier minimisation, so "AP vs risk
score" is really "rank-optimised vs level-optimised". Which wins is measured
here, not assumed.

Also fixes a defect found in s12_baselines.py:140, which passes the test set in
as the early-stopping eval set -- so the tree count was chosen on test. H0
reproduces that protocol so the size of the leak is reported, not asserted.
"""
from __future__ import annotations

import gc
import json
import time
import warnings

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import config as C
from .common import cached_stage, console, log, stage
from .s11_train import FIXED, fit_lightgbm, load, metrics
from .s12_baselines import bootstrap_metric_diff

warnings.filterwarnings("ignore", category=UserWarning)

REPORT_JSON = C.REPORTS / "calibration.json"
BAKEOFF_JSON = C.REPORTS / "bakeoff_results.json"

N_BINS = 15                 # equal-mass reliability bins
SKLEARN_CHAIN_ROWS = 50_000  # H7 only: sklearn GBC is ~9x slower per feature
XGB_FALLBACK = dict(learning_rate=0.05, max_depth=8, min_child_weight=50,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0)


# ---------------------------------------------------------------------------
# Level metrics -- the half of the picture that was never measured
# ---------------------------------------------------------------------------
def reliability(y, p, n_bins: int = N_BINS) -> list[dict]:
    """Observed rate vs predicted rate, in EQUAL-MASS bins.

    Equal-width bins are useless at 5% prevalence: nearly every row lands in the
    first bin and the rest are empty, which flatters any model.
    """
    q = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(dict(bin=b, n=int(m.sum()),
                         mean_predicted=float(p[m].mean()),
                         observed=float(y[m].mean())))
    return rows


def calibration_error(rows: list[dict], n_total: int) -> tuple[float, float]:
    """ECE (mass-weighted mean gap) and MCE (worst single bin)."""
    gap = np.array([abs(r["mean_predicted"] - r["observed"]) for r in rows])
    w = np.array([r["n"] for r in rows], dtype=float) / n_total
    return float((gap * w).sum()), float(gap.max())


def murphy(y, p, rows: list[dict]) -> dict:
    """Brier = reliability - resolution + uncertainty (Murphy 1973).

    Binned, so approximate -- but it is exactly the decomposition the question
    turns on. `reliability` is "the levels are wrong" (recoverable by rescaling);
    `resolution` is "the ranking is informative" (not recoverable by anything).
    """
    n, ybar = len(y), float(y.mean())
    rel = sum(r["n"] / n * (r["mean_predicted"] - r["observed"]) ** 2 for r in rows)
    res = sum(r["n"] / n * (r["observed"] - ybar) ** 2 for r in rows)
    return dict(reliability=float(rel), resolution=float(res),
                uncertainty=float(ybar * (1.0 - ybar)))


def score(y, p) -> tuple[dict, list[dict]]:
    """Rank metrics and level metrics side by side, for one head."""
    br = float(y.mean())
    ref = br * (1.0 - br)          # Brier of "the base rate, for everyone"
    rows = reliability(y, p)
    ece, mce = calibration_error(rows, len(y))
    m = metrics(y, p)
    m.update(brier_skill=float(1.0 - m["brier"] / ref),
             brier_reference_constant=float(ref),
             ece=ece, mce=mce,
             mean_predicted=float(p.mean()),
             **murphy(y, p, rows))
    return m, rows


def ece_of(y, p) -> float:
    """ECE as a bare callable, for the patient-level bootstrap."""
    rows = reliability(y, p)
    return calibration_error(rows, len(y))[0]


# ---------------------------------------------------------------------------
# Calibration maps. Both are monotone, so neither can damage the ranking --
# assertion 2 in main() turns that from a claim into a check.
# ---------------------------------------------------------------------------
def _logit(p, eps: float = 1e-12):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_platt(p_cal, y_cal):
    """Platt scaling: sigmoid(a*logit(p) + b), fitted unregularised.

    Strictly increasing whenever a > 0, so AP and ROC-AUC are preserved exactly.
    eps is 1e-12 rather than the usual 1e-6 so clipping cannot tie apart-but-tiny
    scores together and perturb AP.
    """
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p_cal).reshape(-1, 1), y_cal)
    coef = float(lr.coef_[0][0])
    return (lambda p: lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]), lr, coef


def fit_isotonic(p_cal, y_cal):
    """Non-parametric, monotone but only weakly -- it creates ties, so AP can
    move very slightly. More flexible than Platt; needs more calibration data."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(p_cal, dtype=np.float64), y_cal)
    return (lambda p: iso.predict(np.asarray(p, dtype=np.float64))), iso


# ---------------------------------------------------------------------------
# One XGBoost code path for every XGBoost head
# ---------------------------------------------------------------------------
def fit_xgb(Xtr, ytr, Xes, yes, *, objective="binary:logistic", spw=None,
            params=None, target=None, seed=C.RANDOM_SEED):
    """s11_train.fit_xgboost hardcodes binary:logistic and scale_pos_weight, so
    it cannot express H2 (no reweighting) or H5 (regression head). This mirrors
    it exactly otherwise, so the heads differ only in the named argument.

    `target` overrides the training label -- used by H6, whose second stage
    regresses onto a calibrated probability rather than onto 0/1.
    """
    import xgboost as xgb
    p = dict(objective=objective, tree_method="hist",
             device="cuda" if C.USE_GPU else "cpu", max_bin=C.GPU_MAX_BIN,
             random_state=seed, **(params or {}))
    if objective.startswith("binary"):
        p["eval_metric"] = "aucpr"
        p["scale_pos_weight"] = (
            spw if spw is not None
            else (len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    else:
        p["eval_metric"] = "rmse"

    ttr = ytr if target is None else target
    dtr = xgb.QuantileDMatrix(Xtr, ttr, enable_categorical=True,
                              max_bin=C.GPU_MAX_BIN)
    des = xgb.QuantileDMatrix(Xes, yes, ref=dtr, enable_categorical=True,
                              max_bin=C.GPU_MAX_BIN)
    return xgb.train(p, dtr, num_boost_round=3000, evals=[(des, "es")],
                     early_stopping_rounds=100, verbose_eval=False)


def xgb_predict(bst, X, chunk: int = 500_000):
    """Chunked so a 2.5M-row scoring pass cannot blow up 8 GB of VRAM."""
    import xgboost as xgb
    out = []
    for i in range(0, len(X), chunk):
        d = xgb.DMatrix(X.iloc[i:i + chunk], enable_categorical=True)
        out.append(bst.predict(d, iteration_range=(0, bst.best_iteration + 1)))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Operating point -- the number the product actually consumes
# ---------------------------------------------------------------------------
def hour_key(stay_id: np.ndarray, charttime: np.ndarray) -> np.ndarray:
    """One id per (admission, clock hour).

    Rows are on a greedy irregular grid, so counting flagged ROWS would make a
    densely-charted patient look noisier than a sparsely-charted one. A clinician
    experiences alerts per hour, so that is the unit.
    """
    hr = charttime.astype("datetime64[h]").astype(np.int64)
    hr = hr - hr.min()
    sid = stay_id.astype(np.int64)
    assert hr.max() < 10 ** 7, "hour offset overflows the composite key"
    return sid * 10 ** 7 + hr


def operating_point(y, p, key, t: float) -> dict:
    f = p > t
    tp = int((f & (y == 1)).sum())
    fp = int((f & (y == 0)).sum())
    fn = int(((~f) & (y == 1)).sum())
    total_hours = len(np.unique(key))
    alert_hours = int(len(np.unique(key[f]))) if f.any() else 0
    return dict(threshold=float(t),
                rows_flagged_pct=float(100.0 * f.mean()),
                sensitivity=float(tp / max(tp + fn, 1)),
                ppv=float(tp / max(tp + fp, 1)),
                alert_hours=alert_hours,
                alerts_per_24h=float(24.0 * alert_hours / max(total_hours, 1)))


def threshold_for_budget(y, p, key, budget: float, grid) -> dict:
    """Lowest threshold whose alert rate fits the budget. Alert rate is monotone
    non-increasing in the threshold, so the first hit on an ascending grid wins
    -- a lower threshold means more sensitivity at the same cost."""
    for t in grid:
        op = operating_point(y, p, key, t)
        if op["alerts_per_24h"] <= budget:
            return dict(op, budget_alerts_per_24h=budget)
    return dict(operating_point(y, p, key, grid[-1]),
                budget_alerts_per_24h=budget, note="budget unreachable on this grid")


# ---------------------------------------------------------------------------
def tuned_params(algo: str) -> tuple[dict, str]:
    if BAKEOFF_JSON.exists():
        pb = json.loads(BAKEOFF_JSON.read_text()).get("phase_b", {})
        if algo in pb and "best_params" in pb[algo]:
            return dict(pb[algo]["best_params"]), "tuned"
    return (dict(FIXED["lightgbm"]) if algo == "lightgbm" else dict(XGB_FALLBACK)), "default"


def main(force: bool = False) -> None:
    with cached_stage("s13_calibrate",
                      sources=[C.MODEL_MATRIX_PQ, C.FOLDS_PQ],
                      output=REPORT_JSON, force=force) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    report: dict = {}
    heads: dict = {}
    curves: dict = {}

    X, m = load("full")
    y, sub, stay, fold, split = (m["y"], m["subject_id"], m["stay_id"],
                                 m["fold"], m["split"])
    xp, xp_src = tuned_params("xgboost")
    lp, lp_src = tuned_params("lightgbm")

    # ---------------------------------------------------------------- split
    is_train = split == "train"
    tr = is_train & np.isin(fold, C.TRAIN_FOLDS)
    es = is_train & (fold == C.EARLY_STOP_FOLD)
    cal = is_train & (fold == C.CALIB_FOLD)
    te = split == "test"

    with stage("Split -- four slices, none of them overlapping"):
        for nm, mask in [("train", tr), ("early-stop", es),
                         ("calibrate", cal), ("test", te)]:
            log(f"  {nm:<11} {mask.sum():>9,} rows  "
                f"{len(np.unique(sub[mask])):>6,} patients  "
                f"positive {100 * y[mask].mean():.2f}%")
        s_tr, s_es = set(np.unique(sub[tr])), set(np.unique(sub[es]))
        s_cal, s_te = set(np.unique(sub[cal])), set(np.unique(sub[te]))
        # Assertion 1: the calibrator must never see a patient the model trained
        # on, nor one it will be tested on. Fitting a calibrator on test is the
        # exact defect in data/training_calinerating.ipynb.
        assert not (s_cal & s_tr), "calibration fold shares patients with train"
        assert not (s_cal & s_te), "calibration fold shares patients with test"
        assert not (s_tr & s_te), "train shares patients with test"
        assert not (s_es & s_te), "early-stop fold shares patients with test"
        log("[green]disjoint: train / early-stop / calibrate / test[/green]")
        report["protocol"] = {
            nm: dict(rows=int(mask.sum()),
                     patients=int(len(np.unique(sub[mask]))),
                     positive_rate=float(y[mask].mean()))
            for nm, mask in [("train", tr), ("early_stop", es),
                             ("calibrate", cal), ("test", te)]}
        report["protocol"]["xgboost_params"] = xp_src
        report["protocol"]["lightgbm_params"] = lp_src

    Xtr, Xes, Xcal, Xte = X[tr], X[es], X[cal], X[te]
    ytr, yes, ycal, yte = y[tr], y[es], y[cal], y[te]

    # ------------------------------------------------- H0: the leaky protocol
    with stage("H0 -- reproduce the s12 protocol (early stopping on TEST)"):
        t0 = time.time()
        b0 = fit_xgb(X[is_train], y[is_train], Xte, yte, params=xp)
        p0 = xgb_predict(b0, Xte)
        heads["H0_leaky_early_stopping"], curves["H0_leaky_early_stopping"] = score(yte, p0)
        heads["H0_leaky_early_stopping"].update(
            seconds=round(time.time() - t0, 1), trees=int(b0.best_iteration + 1),
            note="tree count chosen on the test set -- s12_baselines.py:140")
        log(f"  AP={heads['H0_leaky_early_stopping']['average_precision']:.4f} "
            f"({b0.best_iteration + 1} trees)")
        del b0
        gc.collect()

    # ------------------------------------------------------- H1: the control
    with stage("H1 -- XGBoost, scale_pos_weight, raw score (what ships today)"):
        t0 = time.time()
        b1 = fit_xgb(Xtr, ytr, Xes, yes, params=xp)
        p1_te, p1_cal = xgb_predict(b1, Xte), xgb_predict(b1, Xcal)
        heads["H1_xgb_spw_raw"], curves["H1_xgb_spw_raw"] = score(yte, p1_te)
        heads["H1_xgb_spw_raw"].update(seconds=round(time.time() - t0, 1),
                                       trees=int(b1.best_iteration + 1))
        h = heads["H1_xgb_spw_raw"]
        log(f"  AP={h['average_precision']:.4f}  Brier={h['brier']:.4f}  "
            f"BSS={h['brier_skill']:+.3f}  ECE={h['ece']:.4f}  "
            f"mean predicted {h['mean_predicted']:.3f} vs observed {yte.mean():.3f}")
        report["leak_check"] = {
            "leaky_protocol_ap": heads["H0_leaky_early_stopping"]["average_precision"],
            "clean_protocol_ap": h["average_precision"],
            "delta": (heads["H0_leaky_early_stopping"]["average_precision"]
                      - h["average_precision"]),
            "note": "H0 also trains on 40% more patients, so this bounds the "
                    "leak rather than isolating it."}

    # ----------------------------------------- H2: is the reweighting the cause?
    with stage("H2 -- XGBoost with NO scale_pos_weight"):
        t0 = time.time()
        b2 = fit_xgb(Xtr, ytr, Xes, yes, params=xp, spw=1.0)
        p2 = xgb_predict(b2, Xte)
        heads["H2_xgb_no_spw"], curves["H2_xgb_no_spw"] = score(yte, p2)
        heads["H2_xgb_no_spw"]["seconds"] = round(time.time() - t0, 1)
        h = heads["H2_xgb_no_spw"]
        log(f"  AP={h['average_precision']:.4f}  Brier={h['brier']:.4f}  "
            f"BSS={h['brier_skill']:+.3f}  ECE={h['ece']:.4f}  "
            f"mean predicted {h['mean_predicted']:.3f}")
        del b2
        gc.collect()

    # ----------------------------------------------- H3 / H4: post-hoc rescaling
    with stage("H3 / H4 -- Platt and isotonic, fitted on the calibration fold"):
        platt, platt_model, platt_coef = fit_platt(p1_cal, ycal)
        p3 = platt(p1_te)
        heads["H3_xgb_platt"], curves["H3_xgb_platt"] = score(yte, p3)
        heads["H3_xgb_platt"]["platt_slope"] = platt_coef

        iso, iso_model = fit_isotonic(p1_cal, ycal)
        p4 = iso(p1_te)
        heads["H4_xgb_isotonic"], curves["H4_xgb_isotonic"] = score(yte, p4)

        for k in ("H3_xgb_platt", "H4_xgb_isotonic"):
            h = heads[k]
            log(f"  {k:<18} AP={h['average_precision']:.4f}  "
                f"Brier={h['brier']:.4f}  BSS={h['brier_skill']:+.3f}  "
                f"ECE={h['ece']:.4f}  mean predicted {h['mean_predicted']:.4f}")

        # Assertion 2: Platt is strictly monotone, so it CANNOT change the
        # ranking. This is the whole answer in one line -- calibration is free.
        d_ap = abs(heads["H3_xgb_platt"]["average_precision"]
                   - heads["H1_xgb_spw_raw"]["average_precision"])
        assert d_ap < 1e-9, f"Platt moved AP by {d_ap:.2e} -- monotonicity broken"
        log(f"[green]Platt changed AP by {d_ap:.2e} -- ranking provably intact[/green]")

        # Assertion 3: a calibrated forecast's mean must match the base rate.
        gap = abs(heads["H3_xgb_platt"]["mean_predicted"] - float(yte.mean()))
        assert gap < 0.005, f"calibrated mean off the base rate by {gap:.4f}"
        # Assertion 4: it must finally beat the constant forecast.
        assert heads["H3_xgb_platt"]["brier_skill"] > 0, "still worse than a constant"
        log("[green]calibrated mean matches the base rate; BSS > 0[/green]")

    # --------------------------------- H5: the question as literally asked
    with stage("H5 -- XGBoost REGRESSOR on 0/1 (a risk score, Brier-optimal)"):
        t0 = time.time()
        b5 = fit_xgb(Xtr, ytr, Xes, yes, objective="reg:squarederror", params=xp)
        p5 = np.clip(xgb_predict(b5, Xte), 0.0, 1.0)
        heads["H5_xgb_regressor"], curves["H5_xgb_regressor"] = score(yte, p5)
        heads["H5_xgb_regressor"].update(
            seconds=round(time.time() - t0, 1),
            note="squared error on a 0/1 label IS Brier minimisation")
        h = heads["H5_xgb_regressor"]
        log(f"  AP={h['average_precision']:.4f}  Brier={h['brier']:.4f}  "
            f"BSS={h['brier_skill']:+.3f}  ECE={h['ece']:.4f}")
        del b5
        gc.collect()

    # ----------------------------- H6: the bki chain, with a modern base learner
    with stage("H6 -- bki chain structure: classifier -> Platt -> regressor"):
        t0 = time.time()
        # Stage 2 of the chain re-approximates the calibrated probability from
        # the features. Isolating the STRUCTURE means keeping H1 as the base, so
        # the only difference from H3 is this extra regression step.
        tgt_tr = platt(xgb_predict(b1, Xtr))
        tgt_es = platt(xgb_predict(b1, Xes))
        b6 = fit_xgb(Xtr, ytr, Xes, tgt_es, objective="reg:squarederror",
                     params=xp, target=tgt_tr)
        p6 = np.clip(xgb_predict(b6, Xte), 0.0, 1.0)
        heads["H6_bki_chain_modern"], curves["H6_bki_chain_modern"] = score(yte, p6)
        heads["H6_bki_chain_modern"]["seconds"] = round(time.time() - t0, 1)
        h = heads["H6_bki_chain_modern"]
        log(f"  AP={h['average_precision']:.4f}  Brier={h['brier']:.4f}  "
            f"BSS={h['brier_skill']:+.3f}  ECE={h['ece']:.4f}")
        del b6, tgt_tr, tgt_es
        gc.collect()

    # -------------------------------------------------------- H8: LightGBM
    with stage("H8 -- LightGBM raw, then Platt"):
        t0 = time.time()
        lgb_model, _ = fit_lightgbm(Xtr, ytr, Xes, yes, dict(lp), m["cats"])
        p8_te = lgb_model.predict(Xte, num_iteration=lgb_model.best_iteration)
        p8_cal = lgb_model.predict(Xcal, num_iteration=lgb_model.best_iteration)
        heads["H8_lgb_raw"], curves["H8_lgb_raw"] = score(yte, p8_te)
        heads["H8_lgb_raw"]["seconds"] = round(time.time() - t0, 1)
        platt8, _, _ = fit_platt(p8_cal, ycal)
        p8c = platt8(p8_te)
        heads["H8_lgb_platt"], curves["H8_lgb_platt"] = score(yte, p8c)
        for k in ("H8_lgb_raw", "H8_lgb_platt"):
            h = heads[k]
            log(f"  {k:<15} AP={h['average_precision']:.4f}  "
                f"Brier={h['brier']:.4f}  BSS={h['brier_skill']:+.3f}  "
                f"ECE={h['ece']:.4f}")

    # ------------------------------------ H7: the literal sklearn chain, capped
    with stage(f"H7 -- literal sklearn GBC->LR->GBR ({SKLEARN_CHAIN_ROWS:,} rows)"):
        from sklearn.ensemble import (GradientBoostingClassifier,
                                      GradientBoostingRegressor)
        t0 = time.time()
        num = [c for c in X.columns if c not in m["cats"]]
        rng = np.random.default_rng(C.RANDOM_SEED)
        pick = rng.permutation(int(tr.sum()))[:SKLEARN_CHAIN_ROWS]
        A = Xtr[num].to_numpy(dtype=np.float64)[pick]
        b = ytr[pick]
        med = np.nanmedian(A, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        A = np.where(np.isnan(A), med, A)
        T = Xte[num].to_numpy(dtype=np.float64)
        T = np.where(np.isnan(T), med, T)

        gbc = GradientBoostingClassifier(loss="exponential", n_estimators=500,
                                         max_depth=4, min_samples_split=5,
                                         learning_rate=0.01,
                                         random_state=C.RANDOM_SEED).fit(A, b)
        # Orientation corrected: the original takes predict_proba[:, 0], which is
        # P(class 0) -- the modelled quantity runs backwards against "risk".
        s_tr_chain = gbc.predict_proba(A)[:, 1]
        lr7 = LogisticRegression(max_iter=500).fit(s_tr_chain.reshape(-1, 1), b)
        cal_tr = lr7.predict_proba(s_tr_chain.reshape(-1, 1))[:, 1]
        gbr = GradientBoostingRegressor(loss="squared_error", n_estimators=500,
                                        max_depth=4, min_samples_split=5,
                                        learning_rate=0.01,
                                        random_state=C.RANDOM_SEED).fit(A, cal_tr)
        p7 = np.clip(gbr.predict(T), 0.0, 1.0)
        heads["H7_bki_chain_literal"], curves["H7_bki_chain_literal"] = score(yte, p7)
        heads["H7_bki_chain_literal"].update(
            seconds=round(time.time() - t0, 1), rows=int(len(pick)),
            features=len(num),
            note=f"capped at {SKLEARN_CHAIN_ROWS:,} rows and numeric features "
                 "only; sklearn GBC does not scale to 2.5M x 102. Orientation "
                 "corrected and the calibrator fitted on train, so this is the "
                 "chain at its best, not as written.")
        h = heads["H7_bki_chain_literal"]
        log(f"  AP={h['average_precision']:.4f}  Brier={h['brier']:.4f}  "
            f"BSS={h['brier_skill']:+.3f}  ECE={h['ece']:.4f}")
        del A, T, gbc, gbr
        gc.collect()

    # --------------------------------------------- operating point on the winner
    with stage("Operating point -- what `risk > 0.70` actually costs"):
        aux = pl.read_parquet(C.MODEL_MATRIX_PQ, columns=["stay_id", "charttime"])
        ct = aux["charttime"].to_numpy()[te]
        key = hour_key(stay[te], ct)
        total_hours = len(np.unique(key))
        log(f"  test set: {total_hours:,} monitored patient-hours across "
            f"{len(np.unique(stay[te])):,} admissions")

        grid = np.unique(np.quantile(p3, np.linspace(0.50, 0.99999, 400)))
        ops = {"threshold_0.70_raw_H1": operating_point(yte, p1_te, key,
                                                        C.SERVING_THRESHOLD),
               "threshold_0.70_calibrated_H3": operating_point(yte, p3, key,
                                                               C.SERVING_THRESHOLD),
               "budgets": [threshold_for_budget(yte, p3, key, b, grid)
                           for b in C.ALERT_BUDGETS]}
        for nm in ("threshold_0.70_raw_H1", "threshold_0.70_calibrated_H3"):
            o = ops[nm]
            log(f"  {nm:<30} flags {o['rows_flagged_pct']:5.1f}% of rows  "
                f"sens {o['sensitivity']:.3f}  PPV {o['ppv']:.3f}  "
                f"{o['alerts_per_24h']:.1f} alert-hours/day")
        for o in ops["budgets"]:
            log(f"  budget {o['budget_alerts_per_24h']:.0f}/day -> "
                f"threshold {o['threshold']:.4f}  sens {o['sensitivity']:.3f}  "
                f"PPV {o['ppv']:.3f}  {o['alerts_per_24h']:.2f} alert-hours/day")
        # Assertion 7: the alert rate must fall monotonically with the threshold.
        rates = [operating_point(yte, p3, key, t)["alerts_per_24h"]
                 for t in np.quantile(p3, [0.5, 0.8, 0.95, 0.99])]
        assert all(a >= b for a, b in zip(rates, rates[1:])), \
            f"alert rate is not monotone in the threshold: {rates}"
        ops["monitored_patient_hours"] = int(total_hours)
        ops["scored_head"] = "H3_xgb_platt"
        report["operating_points"] = ops

    # ------------------------------------------------------------- bootstrap
    with stage("Patient-level bootstrap on the differences that matter"):
        from sklearn.metrics import brier_score_loss
        pairs = [("H3_platt_vs_H1_raw", p3, p1_te),
                 ("H3_platt_vs_H5_regressor", p3, p5),
                 ("H3_platt_vs_H6_chain", p3, p6),
                 ("H3_platt_vs_H4_isotonic", p3, p4)]
        bs = {}
        for nm, pa, pb in pairs:
            bs[nm] = {
                "brier": bootstrap_metric_diff(yte, pa, pb, sub[te],
                                               brier_score_loss,
                                               higher_is_better=False),
                "ece": bootstrap_metric_diff(yte, pa, pb, sub[te], ece_of,
                                             higher_is_better=False),
                "average_precision": bootstrap_metric_diff(yte, pa, pb, sub[te]),
            }
            b_ = bs[nm]["brier"]
            log(f"  {nm:<28} dBrier={b_['mean']:+.4f} "
                f"[{b_['lo']:+.4f}, {b_['hi']:+.4f}]  "
                f"dAP={bs[nm]['average_precision']['mean']:+.4f}")
        report["bootstrap"] = bs

    # --------------------------------------------------- persist the artifacts
    with stage("Persist model, calibrator and operating point"):
        import joblib
        b1.save_model(str(C.MODEL_XGB))
        lgb_model.save_model(str(C.MODEL_LGB),
                             num_iteration=lgb_model.best_iteration)
        joblib.dump({"kind": "platt", "model": platt_model,
                     "logit_eps": 1e-12,
                     "isotonic": iso_model,
                     "feature_order": list(X.columns),
                     "categorical": m["cats"]}, C.CALIBRATOR_PKL)
        chosen = report["operating_points"]["budgets"][1]  # the 2/day budget
        C.OPERATING_POINT_JSON.write_text(json.dumps({
            "model": str(C.MODEL_XGB.name),
            "calibrator": str(C.CALIBRATOR_PKL.name),
            "apply": "p_calibrated = sigmoid(a * logit(p_raw) + b); "
                     "alert when p_calibrated > threshold",
            "recommended_threshold": chosen["threshold"],
            "recommended_basis": f"{chosen['budget_alerts_per_24h']:.0f} "
                                 "alert-hours per patient-day",
            "sensitivity": chosen["sensitivity"], "ppv": chosen["ppv"],
            "legacy_threshold_0.70": report["operating_points"]
                                           ["threshold_0.70_calibrated_H3"],
        }, indent=2), encoding="utf-8")
        for p in (C.MODEL_XGB, C.MODEL_LGB, C.CALIBRATOR_PKL, C.OPERATING_POINT_JSON):
            log(f"  {p.name:<26} {p.stat().st_size / 1e6:,.1f} MB")

    del X, Xtr, Xes, Xcal, Xte, b1, lgb_model
    gc.collect()

    # ------------------------------ is the label finding an artefact of AP?
    with stage("Is 'warning is a documentation signal' specific to AP?"):
        Xd, md = load("documentation_only")
        dtr = (md["split"] == "train") & np.isin(md["fold"], C.TRAIN_FOLDS)
        des = (md["split"] == "train") & (md["fold"] == C.EARLY_STOP_FOLD)
        dte = md["split"] == "test"
        bd = fit_xgb(Xd[dtr], md["y"][dtr], Xd[des], md["y"][des], params=xp)
        pd_te = xgb_predict(bd, Xd[dte])
        doc, curves["doc_only"] = score(md["y"][dte], pd_te)
        heads["doc_only"] = doc

        prev = float(yte.mean())
        full = heads["H1_xgb_spw_raw"]
        fullc = heads["H3_xgb_platt"]
        share = {
            "average_precision": (doc["average_precision"] - prev)
                                 / (full["average_precision"] - prev),
            "roc_auc": (doc["roc_auc"] - 0.5) / (full["roc_auc"] - 0.5),
        }
        docc, _ = score(md["y"][dte], fit_platt(pd_te, md["y"][dte])[0](pd_te))
        share["brier_skill_calibrated"] = docc["brier_skill"] / fullc["brier_skill"]
        for k, v in share.items():
            log(f"  share of achievable skill reachable from charting alone, "
                f"by {k}: [bold]{100 * v:.1f}%[/bold]")
        report["label_finding"] = {
            "skill_share_from_documentation_only": share,
            "note": "Skill-normalised against each metric's own floor "
                    "(prevalence for AP, 0.5 for ROC-AUC, the base-rate constant "
                    "for Brier skill). The conclusion does not depend on AP -- "
                    "AP is the metric most generous to physiology.",
            "caveat": "brier_skill_calibrated calibrates the documentation-only "
                      "model on the test set itself, so it is an upper bound "
                      "for that row only.",
        }
        del Xd, bd
        gc.collect()

    report["heads"] = heads
    report["reliability_curves"] = curves
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=float),
                           encoding="utf-8")
    log(f"report -> {REPORT_JSON}")

    # ------------------------------------------------------------ the summary
    console.rule("[bold cyan]Rank vs. level, on the same held-out patients")
    log(f"  {'head':<24} {'AP':>7} {'Brier':>8} {'BSS':>7} {'ECE':>7} {'mean p':>8}")
    for k, h in sorted(heads.items(), key=lambda kv: -kv[1]["average_precision"]):
        log(f"  {k:<24} {h['average_precision']:>7.4f} {h['brier']:>8.4f} "
            f"{h['brier_skill']:>+7.3f} {h['ece']:>7.4f} {h['mean_predicted']:>8.4f}")
    log(f"  {'(base-rate constant)':<24} {float(yte.mean()):>7.4f} "
        f"{float(yte.mean() * (1 - yte.mean())):>8.4f} {0.0:>+7.3f} "
        f"{0.0:>7.4f} {float(yte.mean()):>8.4f}")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
