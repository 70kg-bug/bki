"""Gated ablation for the pivot to composite deterioration.

Eight states, applied one change at a time, each measured against the last
ACCEPTED one. A change is kept only if it earns its place; anything that does
not is reverted and the rejection is recorded rather than quietly dropped.

    S0  baseline -- four-arm D, untightened, legacy static block
    S1  + FiO2 escalation must persist to the next measured reading
    S2  + vasopressor flip must be first-of-stay or follow an off-period
    S3  + respiratory arm becomes the label; circulatory reported separately
    S4  + Table 1 built on the final cohort (fills the 22.5% static hole)
    S5  + Charlson severity hierarchy and corrected age bands
    S6  + dobutamine regrouped from vasopressor to inotrope
    S7  + body metrics and tidal volume per kg predicted body weight

WHY NOT AVERAGE PRECISION. AP's floor equals prevalence, so a change that
removes artifact positives lowers AP while making the label cleaner. Judging
on AP would reject every tightening here for the wrong reason. The gate is the
pair of quantities the switch away from `warning` was argued on:

  physiology-attributable skill   AP(full) - AP(documentation_only), and the
                                  same for ROC-AUC. A within-label difference,
                                  so it is unaffected by prevalence moving.
  documentation share             how much of the achievable skill a model
                                  shown only WHICH measurements were charted,
                                  and how stale they are, can reach. Lower is
                                  better; a rise means the label drifted back
                                  towards modelling the nursing chart.

Each state is rebuilt by re-running the data pipeline in a subprocess with the
matching PM_* environment, because config reads those at import. Stage
manifests fingerprint the same constants, so only what actually moved is
recomputed.

    python -m pipeline.gate_pivot              # run the whole sequence
    python -m pipeline.gate_pivot --score S3   # score the current build only
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config as C
from .common import console, log, stage
from .s11_train import load
from .s12_baselines import bootstrap_metric_diff
from .s13_calibrate import fit_platt, fit_xgb, score, tuned_params, xgb_predict

REPORT_JSON = C.REPORTS / "gate_pivot.json"
SCRATCH_DIR = C.SCRATCH / "gate"

# Env deltas applied cumulatively. S0 turns everything off to reproduce the
# state the group approved from, so the sequence starts where the decision was
# actually made rather than where the code now sits.
STEPS: list[tuple[str, str, dict[str, str]]] = [
    ("S0", "baseline: four-arm D, untightened, legacy static block", {
        "PM_TARGET": f"y_composite_{C.PRIMARY_HORIZON_H}h",
        "PM_D_FIO2_PERSIST": "0",
        "PM_D_VASO_STRICT": "0",
        "PM_S02_FINAL_COHORT": "0",
        "PM_CHARLSON_HIERARCHY": "0",
        "PM_STATIC_BODY_METRICS": "0",
        "PM_SPLIT_INOTROPE": "0",
        "PM_ENABLE_PBW": "0",
    }),
    ("S1", "+ FiO2 rise must persist to the next measured reading",
     {"PM_D_FIO2_PERSIST": "1"}),
    ("S2", "+ vasopressor flip must be first-of-stay or follow a 6 h off-period",
     {"PM_D_VASO_STRICT": "1"}),
    ("S3", "respiratory arm becomes the label; circulatory reported separately",
     {"PM_TARGET": f"y_resp_{C.PRIMARY_HORIZON_H}h"}),
    ("S4", "+ Table 1 on the final cohort (fills the static hole)",
     {"PM_S02_FINAL_COHORT": "1"}),
    ("S5", "+ Charlson severity hierarchy and corrected age bands",
     {"PM_CHARLSON_HIERARCHY": "1"}),
    ("S6", "+ dobutamine regrouped from vasopressor to inotrope",
     {"PM_SPLIT_INOTROPE": "1"}),
    ("S7", "+ body metrics and tidal volume per kg PBW",
     {"PM_STATIC_BODY_METRICS": "1", "PM_ENABLE_PBW": "1"}),
]

# A drop inside the noise is not a regression. The gate rejects only when the
# new physiology delta is worse than the last accepted one by more than the
# bootstrap can explain -- i.e. its upper bound sits below the old mean.
DOC_SHARE_TOLERANCE = 0.01   # 1 percentage point


# ---------------------------------------------------------------------------
# Scoring one state
# ---------------------------------------------------------------------------
def score_current(tag: str) -> dict:
    """Fit full and documentation-only on the current build; return the gate metrics."""
    xp, xp_src = tuned_params("xgboost")

    X, m = load("full")
    y = m["y"]
    sub, fold, split = m["subject_id"], m["fold"], m["split"]

    tr_all = split == "train"
    tr = tr_all & np.isin(fold, C.TRAIN_FOLDS)
    es = tr_all & (fold == C.EARLY_STOP_FOLD)
    cal = tr_all & (fold == C.CALIB_FOLD)
    te = split == "test"
    assert not (set(np.unique(sub[tr])) & set(np.unique(sub[te]))), "patient leakage"

    log(f"[bold]{tag}[/bold]  target={C.TARGET}  rows={len(y):,}  "
        f"features={m['n_features']}  "
        f"train {tr.sum():,} / early-stop {es.sum():,} / calib {cal.sum():,} / "
        f"test {te.sum():,}")

    def fit_and_score(Xf, name: str) -> np.ndarray:
        with stage(f"{tag}: {name}"):
            bst = fit_xgb(Xf[tr], y[tr], Xf[es], y[es], params=xp)
            p_te, p_cal = xgb_predict(bst, Xf[te]), xgb_predict(bst, Xf[cal])
            platt, _, _ = fit_platt(p_cal, y[cal])
            p = platt(p_te)
            s, _ = score(y[te], p)
            log(f"  AP={s['average_precision']:.4f}  ROC-AUC={s['roc_auc']:.4f}")
        del bst
        gc.collect()
        return p

    preds = {"full": fit_and_score(X, "full")}
    del X
    gc.collect()

    Xd, md = load("documentation_only")
    # Both loads apply the same censoring mask, so the masks computed above
    # must still address the same rows. If they ever diverge, the two fits are
    # about different patients and the delta between them means nothing.
    assert len(md["y"]) == len(y) and np.array_equal(md["y"], y), (
        "documentation_only load returned a different row set than full")
    preds["documentation_only"] = fit_and_score(Xd, "documentation_only")
    del Xd
    gc.collect()

    prev = float(y[te].mean())
    full, _ = score(y[te], preds["full"])
    doc, _ = score(y[te], preds["documentation_only"])

    d_ap = bootstrap_metric_diff(y[te], preds["full"],
                                 preds["documentation_only"], sub[te])
    d_auc = bootstrap_metric_diff(y[te], preds["full"],
                                  preds["documentation_only"], sub[te],
                                  roc_auc_score)

    return {
        "tag": tag,
        "target": C.TARGET,
        "n_features": int(m["n_features"]),
        "rows_labelled": int(len(y)),
        "rows_test": int(te.sum()),
        "patients_test": int(len(np.unique(sub[te]))),
        "prevalence": prev,
        "xgboost_params": xp_src,
        "flags": {k: os.environ.get(k, "(default)") for k in sorted(
            k for step in STEPS for k in step[2])},
        "full": {"average_precision": full["average_precision"],
                 "roc_auc": full["roc_auc"],
                 # AP minus its floor, rescaled -- comparable when prevalence moves
                 "ap_skill": (full["average_precision"] - prev) / (1.0 - prev)},
        "documentation_only": {"average_precision": doc["average_precision"],
                               "roc_auc": doc["roc_auc"]},
        "documentation_share": {
            "average_precision": ((doc["average_precision"] - prev)
                                  / max(full["average_precision"] - prev, 1e-12)),
            "roc_auc": (doc["roc_auc"] - 0.5) / max(full["roc_auc"] - 0.5, 1e-12)},
        "physiology_delta": {"average_precision": d_ap, "roc_auc": d_auc},
    }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def verdict(new: dict, base: dict | None) -> dict:
    """Accept or reject `new` against the last accepted state."""
    if base is None:
        return {"decision": "BASELINE", "reasons": ["first state, nothing to compare"]}

    reasons, ok = [], True
    for metric, key in (("dAP", "average_precision"), ("dAUC", "roc_auc")):
        n, b = new["physiology_delta"][key], base["physiology_delta"][key]
        if n["mean"] >= b["mean"]:
            reasons.append(f"{metric} {n['mean']:+.4f} >= {b['mean']:+.4f}")
        elif n["hi"] >= b["mean"]:
            reasons.append(
                f"{metric} {n['mean']:+.4f} < {b['mean']:+.4f} but CI hi "
                f"{n['hi']:+.4f} covers it -- within noise")
        else:
            ok = False
            reasons.append(
                f"REGRESSION {metric} {n['mean']:+.4f} [{n['lo']:+.4f}, "
                f"{n['hi']:+.4f}] entirely below {b['mean']:+.4f}")

    ns = new["documentation_share"]["average_precision"]
    bs = base["documentation_share"]["average_precision"]
    if ns <= bs + DOC_SHARE_TOLERANCE:
        reasons.append(f"doc-share {100*ns:.1f}% vs {100*bs:.1f}%")
    else:
        ok = False
        reasons.append(f"REGRESSION doc-share rose {100*bs:.1f}% -> {100*ns:.1f}%")

    return {"decision": "ACCEPT" if ok else "REJECT", "reasons": reasons}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_step(tag: str, env: dict[str, str]) -> dict:
    """Rebuild the data for this state, then score it, both in subprocesses.

    Subprocesses because config resolves PM_* at import time; mutating
    os.environ in this process would not reach an already-imported module.
    """
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out = SCRATCH_DIR / f"{tag}.json"
    out.unlink(missing_ok=True)
    child = {**os.environ, **env}

    for args, label in (
            (["-m", "pipeline.run_all"], "rebuild"),
            (["-m", "pipeline.gate_pivot", "--score", tag], "score")):
        r = subprocess.run([sys.executable, *args], env=child,
                           cwd=str(C.REPO_ROOT))
        if r.returncode != 0:
            raise RuntimeError(f"{tag}: {label} failed with exit {r.returncode}")

    if not out.exists():
        raise RuntimeError(f"{tag}: scorer wrote no result to {out}")
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", metavar="TAG",
                    help="score the CURRENT build and write the result, then exit")
    ap.add_argument("--only", nargs="*", metavar="TAG",
                    help="run just these states (still applied cumulatively)")
    a = ap.parse_args()

    if a.score:
        SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        res = score_current(a.score)
        (SCRATCH_DIR / f"{a.score}.json").write_text(json.dumps(res, indent=2))
        return

    env: dict[str, str] = {}
    results: list[dict] = []
    accepted: dict | None = None
    accepted_env: dict[str, str] = {}

    for tag, desc, delta in STEPS:
        if a.only and tag not in a.only:
            env.update(delta)
            continue
        console.rule(f"[bold cyan]{tag} -- {desc}")
        env.update(delta)
        res = _run_step(tag, env)
        res["description"] = desc
        res["verdict"] = verdict(res, accepted)

        d = res["verdict"]["decision"]
        colour = {"ACCEPT": "green", "REJECT": "red", "BASELINE": "cyan"}[d]
        log(f"[bold {colour}]{d}[/bold {colour}] {tag}")
        for r in res["verdict"]["reasons"]:
            log(f"    {r}")

        if d == "REJECT":
            # Revert: the next state builds on the last one that earned its place.
            log(f"[yellow]reverting {tag}; subsequent states build on "
                f"{accepted['tag'] if accepted else 'S0'}[/yellow]")
            env = dict(accepted_env)
        else:
            accepted, accepted_env = res, dict(env)
        results.append(res)

    _summarise(results)
    REPORT_JSON.write_text(json.dumps(
        {"steps": results,
         "accept_rule": {
             "physiology_delta": "AP and ROC-AUC of full minus documentation_only "
                                 "must not fall by more than the patient-level "
                                 "bootstrap can explain",
             "documentation_share": f"must not rise by more than "
                                    f"{DOC_SHARE_TOLERANCE:.2f}",
             "why_not_average_precision": "AP's floor equals prevalence, so any "
                                          "change that removes artifact positives "
                                          "lowers it without the model worsening"},
         "final": accepted}, indent=2, default=float))
    log(f"report -> {REPORT_JSON}")


def _summarise(results: list[dict]) -> None:
    console.rule("[bold cyan]Gate results")
    log(f"  {'':<4} {'target':<16} {'feat':>5} {'prev':>7} {'AP':>8} {'floor':>7} "
        f"{'AP-skill':>9} {'AUC':>7} {'doc%':>6} {'dAP':>9} {'dAUC':>9}  verdict")
    for r in results:
        f, ds = r["full"], r["documentation_share"]["average_precision"]
        log(f"  {r['tag']:<4} {r['target']:<16} {r['n_features']:>5} "
            f"{100*r['prevalence']:>6.2f}% {f['average_precision']:>8.4f} "
            f"{r['prevalence']:>7.4f} {f['ap_skill']:>9.4f} {f['roc_auc']:>7.4f} "
            f"{100*ds:>5.1f}% "
            f"{r['physiology_delta']['average_precision']['mean']:>+9.4f} "
            f"{r['physiology_delta']['roc_auc']['mean']:>+9.4f}  "
            f"{r['verdict']['decision']}")
    log("  [dim]AP is printed beside its floor because the floor is the "
        "prevalence -- never compare AP down this table[/dim]")


if __name__ == "__main__":
    main()
