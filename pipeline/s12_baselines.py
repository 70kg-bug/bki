"""Stage 12 -- final evaluation on held-out patients, against the current model.

Scores everything on the SAME held-out patients so the numbers are comparable:

  new            LightGBM on the full assembled matrix
  doc_only       the same model given only "which readings were charted" --
                 no physiological values at all
  baseline_small the CURRENT approach reproduced exactly (3,288-row balanced
                 subsample, PCA to 2 components, gradient boosting -> logistic
                 calibration -> gradient boosting regressor)
  baseline_full  that same method, but given all the training data

Splitting the current approach into two rows separates "the method was weak"
from "the data was small", which is the actual question behind this rebuild.

Note on the existing reported MSE of 0.0041: it is not comparable to anything
here. It measures how closely the final regressor reproduces the pipeline's own
calibrated output -- not how well anything predicts `warning`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C
from .common import console, log, stage
from .s10_assemble import FEATURES_JSON
from .s11_train import (FIXED, fit_lightgbm, fit_xgboost, load, metrics,
                        stay_weights)

REPORT_JSON = C.REPORTS / "final_evaluation.json"
N_BOOTSTRAP = 200
SHAP_SAMPLE = 40_000


def bootstrap_metric_diff(y, pa, pb, subject_id, fn=None, *,
                          higher_is_better: bool = True,
                          n=N_BOOTSTRAP, seed=C.RANDOM_SEED):
    """Bootstrap metric(a) - metric(b) by resampling PATIENTS, not rows.

    Rows within a patient are correlated, so a row-level bootstrap would give
    misleadingly tight intervals.

    `fn(y, p) -> float` defaults to average precision. Pass
    `higher_is_better=False` for losses such as Brier or ECE, so `p_worse` keeps
    meaning "probability that a is worse than b" instead of silently inverting.
    """
    if fn is None:
        from sklearn.metrics import average_precision_score
        fn = average_precision_score
    rng = np.random.default_rng(seed)
    order = np.argsort(subject_id, kind="stable")
    uniq, starts = np.unique(subject_id[order], return_index=True)
    bounds = np.append(starts, len(order))
    blocks = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]

    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if y[idx].sum() == 0:
            continue
        diffs.append(fn(y[idx], pa[idx]) - fn(y[idx], pb[idx]))
    if not diffs:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_worse=float("nan"), resamples=0)
    d = np.array(diffs)
    worse = (d <= 0) if higher_is_better else (d >= 0)
    return dict(mean=float(d.mean()),
                lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                p_worse=float(worse.mean()), resamples=len(d))


def bootstrap_ap_diff(y, pa, pb, subject_id, n=N_BOOTSTRAP, seed=C.RANDOM_SEED):
    """Average-precision difference. Thin wrapper; existing callers are unchanged."""
    return bootstrap_metric_diff(y, pa, pb, subject_id, n=n, seed=seed)


def current_method(Xtr, ytr, Xte, max_rows: int | None, seed=C.RANDOM_SEED):
    """The existing pipeline, reproduced faithfully -- including its quirks.

    Kept deliberately: 1:1 class balancing by deleting rows, PCA to 2 components,
    predict_proba[:, 0], and the calibrator fit on its own training scores. These
    are where the current numbers come from, so a fair comparison keeps them.

    `max_rows` caps the balanced sample. 3,288 reproduces the scale the model is
    trained at today; a larger cap isolates "the method" from "the data volume".
    """
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(ytr == 1)
    neg = np.flatnonzero(ytr == 0)
    take = min(len(pos), len(neg))
    if max_rows is not None:
        take = min(take, max_rows // 2)
    idx = np.concatenate([rng.permutation(pos)[:take], rng.permutation(neg)[:take]])
    Xtr, ytr = Xtr[idx], ytr[idx]

    pca = PCA(n_components=2)
    scaler = StandardScaler()
    Ztr = scaler.fit_transform(pca.fit_transform(Xtr))
    Zte = scaler.transform(pca.transform(Xte))

    clf = GradientBoostingClassifier(n_estimators=500, max_depth=4,
                                     min_samples_split=5, learning_rate=0.01,
                                     loss="exponential", random_state=seed)
    clf.fit(Ztr, ytr)

    # The original takes column 0 -- P(class 0) -- throughout. Preserved.
    logits_tr = clf.predict_proba(Ztr)[:, 0]
    calib = LogisticRegression(max_iter=500).fit(logits_tr.reshape(-1, 1), ytr)
    calib_tr = calib.predict_proba(logits_tr.reshape(-1, 1))[:, 0]

    reg = GradientBoostingRegressor(n_estimators=500, max_depth=4,
                                    min_samples_split=5, learning_rate=0.01,
                                    loss="squared_error", random_state=seed)
    reg.fit(Ztr, calib_tr)
    raw = reg.predict(Zte)
    # Column 0 means the score runs backwards relative to "risk"; flip so the
    # comparison is on the model's best possible orientation rather than its
    # accidental one.
    return 1.0 - np.clip(raw, 0, 1)


def main() -> None:
    report = {}

    # ---------------- new model ----------------
    with stage("Final model on held-out patients"):
        X, m = load("full")
        y, sub, stay = m["y"], m["subject_id"], m["stay_id"]
        tr, te = m["split"] == "train", m["split"] == "test"
        log(f"train {tr.sum():,} rows / {len(np.unique(sub[tr])):,} patients | "
            f"test {te.sum():,} rows / {len(np.unique(sub[te])):,} patients")
        assert not (set(np.unique(sub[tr])) & set(np.unique(sub[te]))), "patient leakage"

        tuned = {}
        best_file = C.REPORTS / "bakeoff_results.json"
        if best_file.exists():
            tuned = json.loads(best_file.read_text()).get("phase_b", {})

        # LightGBM and XGBoost were statistically tied in cross-validation, so
        # both are scored on the held-out patients rather than picking one.
        champions, preds_algo = {}, {}
        for algo, fitter in [("lightgbm", fit_lightgbm), ("xgboost", fit_xgboost)]:
            params = dict(tuned[algo]["best_params"]) if algo in tuned \
                and "best_params" in tuned.get(algo, {}) else dict(FIXED["lightgbm"])
            src = "tuned" if algo in tuned and "best_params" in tuned[algo] else "default"
            t0 = time.time()
            model, p = fitter(X[tr], y[tr], X[te], y[te], params, m["cats"])
            champions[algo], preds_algo[algo] = model, p
            report[algo] = metrics(y[te], p)
            report[algo].update(train_seconds=round(time.time() - t0, 1),
                                features=m["n_features"], params_source=src)

            # inference latency -- this serves a real-time bedside monitor
            sample = X[te].iloc[:200_000]
            t0 = time.time()
            if algo == "lightgbm":
                model.predict(sample, num_iteration=model.best_iteration)
            else:
                import xgboost as xgb
                model.predict(xgb.DMatrix(sample, enable_categorical=True))
            us = (time.time() - t0) / len(sample) * 1e6
            report[algo]["inference_us_per_row"] = round(us, 2)
            log(f"  {algo:<10} AP={report[algo]['average_precision']:.4f}  "
                f"ROC-AUC={report[algo]['roc_auc']:.4f}  "
                f"Brier={report[algo]['brier']:.4f}  "
                f"train {report[algo]['train_seconds']}s  "
                f"inference {us:.2f} us/row  [{src} params]")

        winner = max(champions, key=lambda a: report[a]["average_precision"])
        booster, pred_new = champions[winner], preds_algo[winner]
        report["new"] = dict(report[winner], algorithm=winner)
        log(f"  best on held-out patients: [bold]{winner}[/bold] "
            f"(AP={report['new']['average_precision']:.4f})")

    # ---------------- documentation-only diagnostic ----------------
    with stage("Documentation-only diagnostic"):
        Xd, md = load("documentation_only")
        _, pred_doc = fit_lightgbm(Xd[tr], y[tr], Xd[te], y[te],
                                   dict(FIXED["lightgbm"]), md["cats"])
        report["doc_only"] = metrics(y[te], pred_doc)
        log(f"  documentation-only AP={report['doc_only']['average_precision']:.4f} "
            f"using {md['n_features']} features and no physiological values")
        del Xd

    # ---------------- the current approach ----------------
    with stage("Current approach, reproduced"):
        man = json.loads(FEATURES_JSON.read_text())
        cols = man["sets"]["final_only"]
        Xf = X[cols].to_numpy(dtype=np.float64)
        preds = {}
        # 3,288 reproduces today's training scale exactly; 500k shows whether the
        # same method improves when simply given more data.
        for label, cap in [("baseline_small", 3_288), ("baseline_scaled", 500_000)]:
            t0 = time.time()
            p = current_method(Xf[tr], y[tr], Xf[te], max_rows=cap)
            preds[label] = p
            report[label] = metrics(y[te], p)
            n_used = min(cap, 2 * min(int((y[tr] == 1).sum()), int((y[tr] == 0).sum())))
            report[label].update(train_seconds=round(time.time() - t0, 1),
                                 train_rows=n_used)
            log(f"  {label:<16} AP={report[label]['average_precision']:.4f}  "
                f"ROC-AUC={report[label]['roc_auc']:.4f}  "
                f"({n_used:,} training rows, {report[label]['train_seconds']}s)")

    # ---------------- is the difference real? ----------------
    with stage("Bootstrap over patients (is the gap real?)"):
        # Settle the LightGBM-vs-XGBoost question properly rather than by
        # comparing two means that differ by less than their spread.
        d = bootstrap_ap_diff(y[te], preds_algo["xgboost"], preds_algo["lightgbm"],
                              sub[te])
        report.setdefault("bootstrap", {})["xgboost_vs_lightgbm"] = d
        log(f"  xgboost - lightgbm   AP diff {d['mean']:+.4f}  "
            f"95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
            + ("significant" if d["lo"] > 0 or d["hi"] < 0 else
               "[bold]NOT significant -- they are tied[/bold]"))

        for other, p in [("baseline_small", preds["baseline_small"]),
                         ("baseline_scaled", preds["baseline_scaled"]),
                         ("doc_only", pred_doc)]:
            d = bootstrap_ap_diff(y[te], pred_new, p, sub[te])
            report.setdefault("bootstrap", {})[f"new_vs_{other}"] = d
            verdict = "significant" if d["lo"] > 0 else "NOT significant"
            log(f"  new - {other:<16} AP diff {d['mean']:+.4f}  "
                f"95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]  {verdict}")

    # ---------------- where the signal comes from ----------------
    with stage("SHAP grouped by source table"):
        cols = list(X.columns)
        idx = np.flatnonzero(te)[:SHAP_SAMPLE]
        try:
            import shap
            sv = shap.TreeExplainer(booster).shap_values(X.iloc[idx])
            if isinstance(sv, list):
                sv = sv[1]
            imp = np.abs(sv).mean(axis=0)
        except Exception as e:  # noqa: BLE001
            # SHAP can choke on categorical dtypes depending on the backend;
            # native gain importance answers the same question here.
            log(f"  [yellow]SHAP unavailable ({type(e).__name__}); "
                f"falling back to native gain importance[/yellow]")
            if winner == "lightgbm":
                gains = booster.feature_importance(importance_type="gain")
                imp = np.array([gains[cols.index(c)] if c in cols else 0.0
                                for c in cols], dtype=float)
            else:
                score = booster.get_score(importance_type="gain")
                imp = np.array([score.get(c, 0.0) for c in cols], dtype=float)
            report["shap_method"] = "native_gain_importance"
        groups = man["groups"]
        tot = imp.sum()
        by_group = {}
        for gname, gcols in groups.items():
            s = sum(imp[cols.index(c)] for c in gcols if c in cols)
            by_group[gname] = float(s / tot)
            log(f"  {gname:<18} {100*s/tot:5.1f}% of total attribution")
        # documentation-vs-physiology split inside Table 2
        doc_cols = set(man["sets"]["documentation_only"])
        doc_share = sum(imp[cols.index(c)] for c in cols if c in doc_cols) / tot
        log(f"  [bold]of which, 'was it charted' columns: {100*doc_share:.1f}%[/bold]")
        report["shap"] = dict(by_group=by_group, documentation_share=float(doc_share))
        top = np.argsort(-imp)[:15]
        report["shap"]["top_features"] = [(cols[i], float(imp[i])) for i in top]
        log("  top features: " + ", ".join(cols[i] for i in top[:8]))

    REPORT_JSON.write_text(json.dumps(report, indent=2, default=float))
    log(f"report -> {REPORT_JSON}")


if __name__ == "__main__":
    main()
