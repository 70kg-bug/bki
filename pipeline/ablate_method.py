"""Which part of the current method is the bottleneck? Change one thing at a time.

The claim "PCA(n_components=2) is what caps performance" was inferred, not
measured. This isolates it. Every variant is trained on the same training
patients and scored on the same held-out patients, so the only thing that
differs is the named change.

The current pipeline (data/training_calinerating.ipynb) does:
  1. balance classes by DELETING rows        (iloc[:2*n_pos] -> 3,288 rows)
  2. PCA(n_components=2)                     (11 measurements -> 2 components)
  3. StandardScaler AFTER PCA                (backwards: scaling should precede it)
  4. GradientBoostingClassifier
  5. LogisticRegression calibration
  6. GradientBoostingRegressor on the calibrated probability
  7. predict_proba(...)[:, 0]                (probability of class 0 -- inverted)
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C
from .common import console, log, stage
from .s11_train import load, metrics

OUT = C.REPORTS / "method_ablation.json"
GBC = dict(n_estimators=500, max_depth=4, min_samples_split=5,
           learning_rate=0.01, random_state=C.RANDOM_SEED)


def balanced(y, n_rows, seed=C.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    take = min(len(pos), len(neg), n_rows // 2)
    return np.concatenate([rng.permutation(pos)[:take], rng.permutation(neg)[:take]])


def run(Xtr, ytr, Xte, *, n_components, scale_first, head):
    """One variant. `head='chain'` reproduces the current 3-stage pipeline."""
    steps = []
    if scale_first:
        s = StandardScaler().fit(Xtr)
        Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        steps.append("scale")
    if n_components is not None:
        p = PCA(n_components=n_components).fit(Xtr)
        Xtr, Xte = p.transform(Xtr), p.transform(Xte)
        steps.append(f"pca{n_components}")
    if not scale_first:
        s = StandardScaler().fit(Xtr)
        Xtr, Xte = s.transform(Xtr), s.transform(Xte)
        steps.append("scale")

    clf = GradientBoostingClassifier(loss="exponential", **GBC).fit(Xtr, ytr)
    if head == "direct":
        return clf.predict_proba(Xte)[:, 1], "+".join(steps) + "+gbc"

    # the current 3-stage chain, quirks preserved
    logits_tr = clf.predict_proba(Xtr)[:, 0]
    calib = LogisticRegression(max_iter=500).fit(logits_tr.reshape(-1, 1), ytr)
    calib_tr = calib.predict_proba(logits_tr.reshape(-1, 1))[:, 0]
    reg = GradientBoostingRegressor(loss="squared_error", **GBC).fit(Xtr, calib_tr)
    return 1.0 - np.clip(reg.predict(Xte), 0, 1), "+".join(steps) + "+chain"


def main() -> None:
    results = {}

    X11, m = load("final_only")
    y, tr_m, te_m = m["y"], m["split"] == "train", m["split"] == "test"
    A11 = X11.to_numpy(dtype=np.float64)
    tr_i, te_i = np.flatnonzero(tr_m), np.flatnonzero(te_m)
    Xte11, yte = A11[te_i], y[te_i]
    log(f"train pool {len(tr_i):,} rows | test {len(te_i):,} rows "
        f"({len(np.unique(m['subject_id'][te_i])):,} held-out patients)")

    # (rows, n_components, scale_first, head, label)
    VARIANTS = [
        (3_288,  2,    False, "chain",  "current method, exactly as it is"),
        (3_288,  2,    True,  "chain",  "+ scale before PCA (correct order)"),
        (3_288,  2,    False, "direct", "+ drop the 3-stage chain, use the classifier"),
        (3_288,  3,    True,  "chain",  "PCA 3 components"),
        (3_288,  5,    True,  "chain",  "PCA 5 components"),
        (3_288,  8,    True,  "chain",  "PCA 8 components"),
        (3_288,  11,   True,  "chain",  "PCA 11 = rotation only, no reduction"),
        (3_288,  None, True,  "chain",  "no PCA at all"),
        (3_288,  None, True,  "direct", "no PCA + no chain"),
        (50_000, None, True,  "direct", "no PCA + no chain, 50k rows"),
        (359_148, None, True, "direct", "no PCA + no chain, 359k rows"),
    ]

    with stage("Method ablation on the 11 frozen values"):
        for n_rows, ncomp, scale_first, head, label in VARIANTS:
            idx = tr_i[balanced(y[tr_i], n_rows)]
            t0 = time.time()
            pred, recipe = run(A11[idx], y[idx], Xte11,
                               n_components=ncomp, scale_first=scale_first, head=head)
            r = metrics(yte, pred)
            r.update(rows=len(idx), recipe=recipe, seconds=round(time.time() - t0, 1))
            results[label] = r
            log(f"  {label:<46} AP={r['average_precision']:.4f}  "
                f"ROC-AUC={r['roc_auc']:.4f}  ({len(idx):,} rows, {r['seconds']}s)")
    del A11, X11

    OUT.write_text(json.dumps(results, indent=2, default=float))  # checkpoint

    # Does the feature set matter more than any of the above?
    with stage("Same method, but the full 55 imputed time-series columns"):
        X55, m55 = load("t2_all")
        A55 = X55.to_numpy(dtype=np.float64)
        # GradientBoostingClassifier cannot take NaN (LightGBM/XGBoost can, which
        # is itself part of why they do better). Median-fill so the comparison is
        # about the feature set rather than about NaN support.
        med = np.nanmedian(A55[tr_i], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        A55 = np.where(np.isnan(A55), med, A55)
        for n_rows, label in [(3_288, "no PCA + no chain, 55 features, 3,288 rows"),
                              (50_000, "no PCA + no chain, 55 features, 50k rows")]:
            idx = tr_i[balanced(y[tr_i], n_rows)]
            t0 = time.time()
            pred, recipe = run(A55[idx], y[idx], A55[te_i],
                               n_components=None, scale_first=True, head="direct")
            r = metrics(yte, pred)
            r.update(rows=len(idx), recipe=recipe, features=55,
                     seconds=round(time.time() - t0, 1))
            results[label] = r
            log(f"  {label:<46} AP={r['average_precision']:.4f}  "
                f"ROC-AUC={r['roc_auc']:.4f}  ({len(idx):,} rows, {r['seconds']}s)")

    OUT.write_text(json.dumps(results, indent=2, default=float))
    log(f"report -> {OUT}")

    base = results["current method, exactly as it is"]["average_precision"]
    console.rule("[bold cyan]Lift over the current method")
    for label, r in sorted(results.items(), key=lambda kv: -kv[1]["average_precision"]):
        log(f"  {label:<46} AP={r['average_precision']:.4f}  "
            f"({r['average_precision'] - base:+.4f})")


if __name__ == "__main__":
    main()
