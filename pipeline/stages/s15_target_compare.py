"""Stage 15 -- target D versus the incumbent `warning`, strict deltas.

The case for D was construct validity: it is time-varying, built from measured
values and orders rather than a charting flag, and clinically actionable. That is
an argument, not a measurement. This measures it -- including the one outcome that
would sink the switch: D turning out to be mostly a charting signal too.

WHY AVERAGE PRECISION CANNOT BE COMPARED ACROSS THESE LABELS
    Random-guess AP equals the positive rate. `warning` sits at 5.3%, D at 13.8%,
    so D starts from a floor 2.6x higher. A larger raw AP on D would be partly the
    base rate and nothing else. Reported here, but always beside its floor and
    never as the verdict.

    Comparable across labels:  ROC-AUC (prevalence-independent), normalised AP
                               skill (AP - prev)/(1 - prev), Brier skill.
    The decisive number:       the WITHIN-label delta, full minus
                               documentation-only. It is a ratio computed inside
                               one label, so the prevalence problem cannot touch
                               it -- and it is exactly the quantity that condemned
                               `warning` in the first place.

STRICT DELTAS. Every candidate sees the same patients, the same folds, the same
102 features, the same params, the same 4-way clean protocol from s13, and -- the
part that actually makes the comparison legitimate -- THE SAME ROWS. D is
undefined where its forward window overruns the end of the stay, so all labels are
scored on D's observable row set. `warning` is additionally reported on its native
full row set so the result still connects to the published headline.
"""
from __future__ import annotations

import gc
import json

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from .. import config as C
from ..common import align_forward_labels, cached_stage, console, log, stage
from .s11_train import load
from .s12_baselines import bootstrap_metric_diff
from .s13_calibrate import fit_platt, fit_xgb, score, tuned_params, xgb_predict

REPORT_JSON = C.RPT_S15_COMPARISON

H = C.PRIMARY_HORIZON_H
LABELS = [
    (C.LEGACY_TARGET, None, "incumbent -- caregiver documentation flag"),
    ("D_resp", f"y_resp_{H}h", f"SHIPPED: respiratory arm, {H} h look-ahead"),
    ("D_circ", f"y_circ_{H}h", "circulatory arm, reported separately"),
    ("D_composite", f"y_composite_{H}h", f"composite deterioration, {H} h look-ahead"),
    ("D_strict", f"y_strict_{H}h", f"as above, escalation baseline <= "
                                   f"{C.STRICT_BASELINE_AGE_MIN:.0f} min old"),
]


# align_forward_labels now lives in common, shared with s11_train.load() --
# the trainer and this comparison must align labels identically or their
# numbers are not about the same rows.


def coverage(y, stay_id, charttime) -> dict:
    """A target that fires for a fifth of patients cannot drive a ward monitor."""
    stays = np.unique(stay_id)
    hit = np.unique(stay_id[y == 1])

    # Monitored days = the sum of each admission's own span, not the calendar
    # range of the cohort.
    order = np.argsort(stay_id, kind="stable")
    sid, ct = stay_id[order], charttime[order]
    bounds = np.append(np.unique(sid, return_index=True)[1], len(sid))
    total_days = sum(
        (ct[bounds[i]:bounds[i + 1]].max() - ct[bounds[i]:bounds[i + 1]].min())
        / np.timedelta64(1, "D") for i in range(len(bounds) - 1))

    return {"admissions": int(len(stays)),
            "admissions_with_any_positive": int(len(hit)),
            "admissions_any_pct": float(100 * len(hit) / len(stays)),
            "monitored_stay_days": float(total_days),
            "positives_per_stay_day": float(int(y.sum()) / max(total_days, 1e-9))}


def main(force: bool = False) -> None:
    sources = [C.MODEL_MATRIX_PQ, C.FORWARD_TARGETS_PQ, C.FOLDS_PQ]
    with cached_stage("s15_target_compare", sources=sources, output=REPORT_JSON,
                      force=force, extra=C.FP_COMPARE) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    report: dict = {"protocol": {}, "labels": {}, "deltas": {}}
    xp, xp_src = tuned_params("xgboost")

    # Deliberately the INCUMBENT label on the UNFILTERED matrix. load() would
    # otherwise hand back the shipped forward target with censored rows already
    # dropped, and every label here has to be scored on one common row set.
    X, m = load("full", target=C.LEGACY_TARGET, target_source="matrix")
    sub, stay, fold, split = m["subject_id"], m["stay_id"], m["fold"], m["split"]
    fwd = align_forward_labels(len(sub))
    charttime = m["charttime"]

    ys, valid = {C.LEGACY_TARGET: m["y"].astype(np.int8)}, {}
    for name, col, _ in LABELS:
        if col is None:
            continue
        s = fwd[col]
        valid[name] = s.is_not_null().to_numpy()
        ys[name] = s.fill_null(0).to_numpy().astype(np.int8)

    # The comparison row set: where every candidate is observable. Identical
    # for every label, which is the whole point -- otherwise no difference
    # between them is real.
    comparable = np.logical_and.reduce(list(valid.values()))
    with stage("Comparison row set"):
        log(f"  all rows                     {len(sub):>12,}")
        log(f"  D observable (both variants) {comparable.sum():>12,}  "
            f"({100*comparable.mean():.2f}%)")
        log(f"  dropped as unobservable      {(~comparable).sum():>12,}  "
            f"({100*(~comparable).mean():.2f}%)")
        for name, _, _ in LABELS:
            log(f"    {name:<14} positive rate on the shared rows: "
                f"{100*ys[name][comparable].mean():.2f}%")
        report["protocol"] = {
            "rows_total": int(len(sub)),
            "rows_compared": int(comparable.sum()),
            "pct_dropped_unobservable": float(100 * (~comparable).mean()),
            "horizon_hours": H, "xgboost_params": xp_src,
            "censoring": "positive if observed in (t, t+H]; negative only if the "
                         "window fits inside the stay; dropped otherwise",
        }

    def masks(v):
        tr_all = v & (split == "train")
        return (tr_all & np.isin(fold, C.TRAIN_FOLDS),
                tr_all & (fold == C.EARLY_STOP_FOLD),
                tr_all & (fold == C.CALIB_FOLD),
                v & (split == "test"))

    tr, es, cal, te = masks(comparable)
    assert not (set(np.unique(sub[tr])) & set(np.unique(sub[te]))), "patient leakage"
    log(f"train {tr.sum():,} | early-stop {es.sum():,} | calibrate {cal.sum():,} | "
        f"test {te.sum():,} rows ({len(np.unique(sub[te])):,} held-out patients)")

    # ---- fits: full feature set first, then documentation-only ---------------
    preds: dict[tuple[str, str], dict] = {}

    def fit_all(Xf, tag: str, masks4):
        a, b, c, d = masks4
        for name, _, _ in LABELS:
            y = ys[name]
            with stage(f"{name} -- {tag}"):
                bst = fit_xgb(Xf[a], y[a], Xf[b], y[b], params=xp)
                p_te, p_cal = xgb_predict(bst, Xf[d]), xgb_predict(bst, Xf[c])
                platt, _, _ = fit_platt(p_cal, y[c])
                preds[(name, tag)] = {"raw": p_te, "cal": platt(p_te)}
                s, _ = score(y[d], platt(p_te))
                log(f"  AP={s['average_precision']:.4f}  ROC-AUC={s['roc_auc']:.4f}  "
                    f"BSS={s['brier_skill']:+.3f}  ECE={s['ece']:.4f}")
            del bst
            gc.collect()

    fit_all(X, "full", (tr, es, cal, te))
    del X
    gc.collect()

    # Same override as the full matrix above. Without it load() would apply the
    # SHIPPED target's censoring and hand back 3.93M rows, while tr/es/cal/te
    # index the unfiltered 4.20M -- the two feature sets would no longer be
    # about the same rows, which is the one thing this comparison exists to
    # guarantee.
    Xd, md = load("documentation_only", target=C.LEGACY_TARGET,
                  target_source="matrix")
    assert len(Xd) == len(sub), (
        f"doc_only matrix has {len(Xd):,} rows, full had {len(sub):,}")
    fit_all(Xd, "doc_only", (tr, es, cal, te))
    del Xd, md
    gc.collect()

    # ---- score, and compute the delta that actually decides it ---------------
    with stage("Rank, level, and the documentation share"):
        for name, _, note in LABELS:
            y = ys[name]
            prev = float(y[te].mean())
            full, _ = score(y[te], preds[(name, "full")]["cal"])
            doc, _ = score(y[te], preds[(name, "doc_only")]["cal"])
            ap_skill = lambda ap: (ap - prev) / (1.0 - prev)  # noqa: E731
            share_ap = ((doc["average_precision"] - prev)
                        / (full["average_precision"] - prev))
            share_auc = (doc["roc_auc"] - 0.5) / (full["roc_auc"] - 0.5)
            report["labels"][name] = {
                "note": note, "prevalence": prev,
                "full": dict(full, ap_skill=ap_skill(full["average_precision"])),
                "documentation_only": dict(
                    doc, ap_skill=ap_skill(doc["average_precision"])),
                "documentation_share": {"average_precision": share_ap,
                                        "roc_auc": share_auc},
                "physiology_adds": {
                    "average_precision": full["average_precision"]
                                         - doc["average_precision"],
                    "roc_auc": full["roc_auc"] - doc["roc_auc"],
                    "share_of_skill": 1.0 - share_ap},
                "coverage": coverage(y[te], stay[te], charttime[te]),
            }
            log(f"  {name:<14} prev {100*prev:>5.2f}%  AP {full['average_precision']:.4f} "
                f"(floor {prev:.4f})  ROC-AUC {full['roc_auc']:.4f}  "
                f"doc-share {100*share_ap:>5.1f}% AP / {100*share_auc:>5.1f}% AUC")

    with stage("Patient-level bootstrap: how much does physiology add, per label?"):
        for name, _, _ in LABELS:
            y = ys[name]
            d_ap = bootstrap_metric_diff(y[te], preds[(name, "full")]["cal"],
                                         preds[(name, "doc_only")]["cal"], sub[te])
            d_auc = bootstrap_metric_diff(y[te], preds[(name, "full")]["cal"],
                                          preds[(name, "doc_only")]["cal"], sub[te],
                                          roc_auc_score)
            report["deltas"][name] = {"full_minus_doc_ap": d_ap,
                                      "full_minus_doc_auc": d_auc}
            log(f"  {name:<14} dAP={d_ap['mean']:+.4f} "
                f"[{d_ap['lo']:+.4f}, {d_ap['hi']:+.4f}]   "
                f"dAUC={d_auc['mean']:+.4f} "
                f"[{d_auc['lo']:+.4f}, {d_auc['hi']:+.4f}]")

    with stage(f"Do `{C.LEGACY_TARGET}` and D measure the same thing?"):
        # Against the SHIPPED label first, then the composite, which is what
        # the published kappa was computed on and is kept for continuity.
        report["agreement"] = {}
        yw = ys[C.LEGACY_TARGET][te]
        for other in ("D_resp", "D_composite"):
            yd = ys[other][te]
            both = int(((yw == 1) & (yd == 1)).sum())
            a = {"cohen_kappa": float(cohen_kappa_score(yw, yd)),
                 "both_positive": both,
                 f"{C.LEGACY_TARGET}_only": int(((yw == 1) & (yd == 0)).sum()),
                 "D_only": int(((yw == 0) & (yd == 1)).sum()),
                 "neither": int(((yw == 0) & (yd == 0)).sum()),
                 "jaccard": both / max(int(((yw == 1) | (yd == 1)).sum()), 1)}
            report["agreement"][other] = a
            log(f"  {other:<12} kappa = {a['cohen_kappa']:.4f}  |  both positive "
                f"{both:,}  {C.LEGACY_TARGET}-only "
                f"{a[f'{C.LEGACY_TARGET}_only']:,}  D-only {a['D_only']:,}")
        log("  [dim]kappa near 0 means the two labels are not the same event "
            "wearing different names[/dim]")

    report["horizon_sweep"] = json.loads(
        C.RPT_S14_FORWARD.read_text())["variants"]
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=float),
                           encoding="utf-8")
    log(f"report -> {REPORT_JSON}")

    console.rule("[bold cyan]Verdict -- identical patients, rows, features and params")
    log(f"  {'label':<14} {'prev':>7} {'AP':>7} {'APskill':>8} {'ROC-AUC':>8} "
        f"{'doc-share':>10} {'physio adds':>12}")
    for name, _, _ in LABELS:
        r = report["labels"][name]
        log(f"  {name:<14} {100*r['prevalence']:>6.2f}% "
            f"{r['full']['average_precision']:>7.4f} {r['full']['ap_skill']:>8.4f} "
            f"{r['full']['roc_auc']:>8.4f} "
            f"{100*r['documentation_share']['average_precision']:>9.1f}% "
            f"{r['physiology_adds']['average_precision']:>+12.4f}")
    log("  [dim]AP is NOT comparable across rows of this table -- the floors differ. "
        "Compare AP-skill, ROC-AUC and doc-share.[/dim]")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
