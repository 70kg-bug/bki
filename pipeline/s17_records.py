"""Stage 17 -- emit records conforming to the risk output contract.

s16 declares the contract in models/risk_bands_*.json. Until something produces
a record against it, it is a document rather than an interface. This stage is
that producer, and it deliberately CONSUMES the persisted artifact -- the band
machine is loaded with bands.BandMachine.from_json, never re-derived -- so a
successful run is evidence the artifact is loadable and sufficient on its own.

WHY CONTRIBUTORS EXIST
----------------------
The band states severity, not cause. An LLM handed "CRITICAL" and a few raw
ventilator numbers will produce a confident causal story it has no basis for.
Per-reading attributions give it something checkable instead.

Two properties make them safe to hand downstream:

  SPACE  Contributions are rescaled into CALIBRATED-LOGIT space, so
         sigmoid(sum + bias) is exactly the score in the same record. Raw
         TreeSHAP explains the pre-calibration margin, which nobody sees.
         Asserted per row, not assumed. See scoring.Scorer.contributions.

  KIND   Every contributor is tagged physiology or documentation. The 33
         documentation features carry 18.1% of this model's skill, so they
         genuinely surface near the top -- `spo2_delta_t_min` is an honest
         attribution and a catastrophic sentence if narrated as "the patient's
         oxygenation is being monitored less often" in a clinical voice. They
         are LABELLED rather than dropped: hiding them would misrepresent the
         model. Each record also carries its own documentation share, so any
         explanation built from it can be qualified honestly.

SAMPLED, AND SAID SO
--------------------
A seeded sample of admissions, not the whole test fold. This is a golden set for
prompt and retrieval work, not a dump, and disk headroom here is a live
constraint. The count kept and the count dropped are both logged -- a bounded
emit that reads as complete is how a coverage gap ships unnoticed.

DUA
---
Records carry per-patient MIMIC values keyed to stay_id. They go to build/,
which is outside the repo and gitignored. reports/records.json gets aggregates
only -- reports/ is tracked.
"""
from __future__ import annotations

import gzip
import json
import time

import numpy as np

from . import bands as B
from . import config as C
from . import scoring
from .common import cached_stage, log, stage
from .s10_assemble import FEATURES_JSON
from .s11_train import load
from .s16_bands import stay_groups, subsample

REPORT_JSON = C.REPORTS / "records.json"


def _sigmoid(z):
    """Overflow-safe, because contributions can put the logit well past +-40."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -709, 709))),
                    np.exp(np.clip(z, -709, 709)) / (1.0 + np.exp(np.clip(z, -709, 709))))


def feature_kinds(cols: list[str]) -> tuple[dict, dict]:
    """(feature -> 'physiology'|'documentation', feature -> source table).

    Documentation membership comes from features.json's own `documentation_only`
    set rather than from a name pattern -- the set is what every other stage in
    this pipeline measures doc-share against, and a second definition would
    drift from it.
    """
    man = json.loads(FEATURES_JSON.read_text())
    doc = set(man["sets"]["documentation_only"])
    group = {c: g for g, cs in man["groups"].items() for c in cs}
    kind = {c: ("documentation" if c in doc else "physiology") for c in cols}
    return kind, {c: group.get(c, "unknown") for c in cols}


def telemetry_columns(cols: set[str]) -> list[tuple[str, str, str, str, str]]:
    """(param, value_col, age_col, measured_col, locf_col) per frozen parameter.

    `_locf` is carried alongside `_final` to separate THREE epistemic states
    that `measured` alone collapses into two. s07_impute.py:86 defines
    `_final = _locf.fill_null(cohort_reference)`, and where both are present
    they are bit-identical -- so the only thing `_final` adds is a population
    default where this patient has no value at all. That is 20.4% of PEEP
    readings and 13.8% of FiO2. Handing an explanatory layer "PEEP is 8" when
    the 8 is a cohort statistic would state a population fact as a patient
    observation, in a clinical voice.
    """
    out = []
    for p in C.FROZEN_PARAMS:
        v, a, o, l = (f"{p}_final", f"{p}_delta_t_min", f"{p}_observed",
                      f"{p}_locf")
        if v in cols:
            out.append((p, v, a if a in cols else "", o if o in cols else "",
                        l if l in cols else ""))
    return out


def _source(measured, locf_missing) -> str:
    """measured now | this patient's last value | not this patient at all."""
    if measured:
        return "measured"
    return "population_reference" if locf_missing else "carried_forward"


def main(force: bool = False) -> None:
    with cached_stage("s17_records",
                      sources=[C.MODEL_MATRIX_PQ, C.FOLDS_PQ, C.MODEL_XGB,
                               C.CALIBRATOR_PKL, C.BAND_TABLE_JSON],
                      output=C.RECORDS_JSONL, force=force,
                      extra=C.FP_RECORDS) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    report: dict = {}
    t0 = time.time()

    # --------------------------------------------------------- the contract
    with stage("Load the contract that s16 persisted"):
        art = json.loads(C.BAND_TABLE_JSON.read_text())
        machine = B.BandMachine.from_json(art["machine"])
        band_meta = {b["band"]: b for b in art["bands"]}
        base_rate = art["provenance"]["base_rate"]
        assert art["provenance"]["label"] == C.TARGET, (
            f"band table is for {art['provenance']['label']}, config says "
            f"{C.TARGET}")
        log(f"  schema {art['schema_version']}  label {art['provenance']['label']}  "
            f"device {art['provenance']['scoring_device']}")
        log(f"  cuts {'/'.join(f'{c:.4f}' for c in machine.cuts)}")

    # ------------------------------------------------------------- score
    with stage("Re-score the test fold"):
        X, m = load("full")
        y, stay, ct = m["y"], m["stay_id"], m["charttime"]
        te = m["split"] == "test"
        sc = scoring.load_scorer(expect_columns=X.columns)
        assert sc.device == art["provenance"]["scoring_device"], (
            f"band table was fitted on {art['provenance']['scoring_device']} but "
            f"this process scores on {sc.device} -- every cut would shift")
        cols = list(X.columns)

        g_te, t_te = stay_groups(stay, ct, te)
        gs, ts, n_sub = subsample(g_te, t_te, C.RECORD_SAMPLE_STAYS)
        rows = np.sort(np.concatenate(gs))
        log(f"  emitting {n_sub:,} of {len(g_te):,} test admissions "
            f"({len(g_te) - n_sub:,} dropped), {len(rows):,} readings")

        Xs = X.iloc[rows]
        p = sc.score(Xs)
        del X

    # ------------------------------------------------------- attributions
    with stage("Per-reading attributions, rescaled to calibrated-logit space"):
        contribs, bias = sc.contributions(Xs)
        recon = _sigmoid(contribs.sum(axis=1) + bias)
        err = np.abs(recon - p)
        log(f"  reconstruction error  max {err.max():.3e}  "
            f"mean {err.mean():.3e}  (tol {C.CONTRIB_RECON_TOL:.0e})")
        # Assertion 1: these numbers must explain THIS score. Without it the
        # block is "we ran SHAP" rather than an explanation of the record.
        assert err.max() < C.CONTRIB_RECON_TOL, (
            f"contributions do not reconstruct the calibrated score: max error "
            f"{err.max():.3e}. They would explain a number the record does not "
            f"report.")

        kind, group = feature_kinds(cols)
        doc_idx = np.array([kind[c] == "documentation" for c in cols])
        absc = np.abs(contribs)
        tot = absc.sum(axis=1)
        doc_share = np.where(tot > 0, absc[:, doc_idx].sum(axis=1) / tot, 0.0)
        # Assertion 2: doc-share is a share.
        assert (doc_share >= -1e-9).all() and (doc_share <= 1 + 1e-9).all()
        log(f"  documentation share of |attribution|: median "
            f"{np.median(doc_share):.3f}  p90 {np.percentile(doc_share, 90):.3f}")
        # -------------------------------------------- sufficiency signals
        # Two more shares of the SAME denominator as doc_share, over disjoint
        # feature sets, so they compose rather than double-count:
        #
        #   documentation_share   *_observed / *_delta_t_min / *_structurally_*
        #   imputed_share         *_final / *_locf where this patient has no
        #                         value at all -- the number the model split on
        #                         is then a COHORT STATISTIC, not an observation
        #
        # A reading whose score is mostly cohort default is not an assessment of
        # this patient, and a downstream layer must be able to refuse to explain
        # it. Computed here because the denominator is the full 110-column
        # matrix; the record itself carries only the top-8.
        imp_num = np.zeros(len(rows))
        age_num = np.zeros(len(rows))
        age_den = np.zeros(len(rows))
        colset = set(cols)
        for prm in C.FROZEN_PARAMS:
            value_feats = [f"{prm}_final", f"{prm}_locf"]
            if prm == "tidal_volume_observed":
                # Derived from tidal_volume_observed_final, so it inherits that
                # parameter's imputation rather than being independent of it.
                value_feats.append("tidal_volume_ml_per_kg_pbw")
            idx = [cols.index(c) for c in value_feats if c in colset]
            lc, ac = f"{prm}_locf", f"{prm}_delta_t_min"
            if not idx or lc not in colset:
                continue
            w = absc[:, idx].sum(axis=1)
            # population_reference IFF _locf is null: _locf carries the current
            # observation whenever there is one, so a null means this patient
            # was never observed for this parameter up to now.
            never = np.isnan(Xs[lc].to_numpy(dtype=np.float64))
            imp_num += np.where(never, w, 0.0)
            if ac in colset:
                a = Xs[ac].to_numpy(dtype=np.float64)
                ok = ~np.isnan(a)
                age_num += np.where(ok, w * np.nan_to_num(a), 0.0)
                age_den += np.where(ok, w, 0.0)
        imputed_share = np.where(tot > 0, imp_num / np.maximum(tot, 1e-300), 0.0)
        # Attribution-WEIGHTED age, not "oldest value in use". At this grain a
        # row exists because ONE parameter was charted, so something is nearly
        # always stale -- the oldest value in use is 420 min at the median and
        # 2,880 min at p75, which makes any threshold on it suppress most of the
        # set. What matters is whether the values the score actually leaned on
        # are current.
        attr_age = np.where(age_den > 0, age_num / np.maximum(age_den, 1e-300),
                            np.nan)
        # Assertion 3: disjoint shares of one denominator cannot exceed it. This
        # is what stops the two signals silently measuring the same features.
        assert (imputed_share >= -1e-9).all() and (imputed_share <= 1 + 1e-9).all()
        assert (doc_share + imputed_share <= 1 + 1e-9).all(), (
            "documentation and imputed share overlap -- they are defined over "
            "disjoint feature sets and must not double-count")
        log(f"  imputed share of |attribution|: median "
            f"{np.median(imputed_share):.3f}  p90 "
            f"{np.percentile(imputed_share, 90):.3f}")
        _fin = attr_age[~np.isnan(attr_age)]
        log(f"  attribution-weighted age of the values in use: median "
            f"{np.median(_fin):,.0f} min  p90 {np.percentile(_fin, 90):,.0f} min"
            f"  ({len(rows) - len(_fin):,} readings undefined)")

        order = np.argsort(-absc, axis=1)[:, :C.CONTRIB_TOP_K]
        # The tail the record does not list. Without it a consumer holding only
        # the top-k cannot tell a truncated explanation from a wrong one; with
        # it, sum(top_k) + other + bias reconstructs the logit exactly and the
        # explanation is checkable by whoever receives it.
        tail = contribs.sum(axis=1) - np.take_along_axis(contribs, order, 1).sum(axis=1)
        kept = np.take_along_axis(absc, order, 1).sum(axis=1) / np.maximum(tot, 1e-300)
        log(f"  top {C.CONTRIB_TOP_K} carry {np.median(kept):.1%} of |attribution| "
            f"at the median, {np.percentile(kept, 10):.1%} at p10")

    # ------------------------------------------------------------- emit
    with stage(f"Emit records -> {C.RECORDS_JSONL.name}"):
        tele = telemetry_columns(set(cols))
        pos = {r: i for i, r in enumerate(rows)}     # matrix row -> sample index
        # Per-column arrays rather than one float matrix: 9 of the 109 features
        # are categorical (gender, ventilator_mode, ...) and any of them can
        # surface as a top contributor. Coercing those to codes would hand the
        # explanatory layer an integer where it needs "Pressure Support".
        colvals = {}
        for c in cols:
            s = Xs[c]
            colvals[c] = (s.astype("object").where(s.notna(), None).to_numpy()
                          if str(s.dtype) == "category"
                          else s.to_numpy(dtype=np.float64))
        prov = {k: art["provenance"][k] for k in
                ("model", "calibrator", "label", "arm", "horizon_hours",
                 "scoring_device")}
        prov["band_table_schema_version"] = art["schema_version"]

        # Where each parameter's value actually comes from. Reported because a
        # parameter that is mostly population_reference is one the explanatory
        # layer should barely mention, and nothing else in the pipeline says so.
        src_stats = {}
        for prm, vc, ac, oc, lc in tele:
            meas = colvals[oc] > 0.5 if oc else np.zeros(len(rows), bool)
            lnull = np.isnan(colvals[lc]) if lc else np.zeros(len(rows), bool)
            src_stats[prm] = dict(
                measured=float(meas.mean()),
                carried_forward=float((~meas & ~lnull).mean()),
                population_reference=float((~meas & lnull).mean()))

        n, top_counts = 0, {}
        with gzip.open(C.RECORDS_JSONL, "wt", encoding="utf-8") as fh:
            for g, mins in zip(gs, ts):
                stepper = B.BandStepper(machine)
                for ridx, minute in zip(g, mins):
                    i = pos[ridx]
                    v = stepper.push(float(p[i]), float(minute))
                    bm = band_meta[v.displayed]
                    ks = [cols[j] for j in order[i]]
                    top_counts[ks[0]] = top_counts.get(ks[0], 0) + 1
                    fh.write(json.dumps({
                        "schema_version": art["schema_version"],
                        "provenance": prov,
                        "stay_id": int(stay[ridx]),
                        "charttime": str(ct[ridx]),
                        "risk": {"calibrated": float(p[i]), "is_probability": True},
                        "band": {
                            "displayed": v.displayed, "instant": v.instant,
                            "state": v.state,
                            "readings_in_state": v.readings_in_state,
                            "observed_rate": bm["observed_rate"],
                            "base_rate": base_rate,
                            "lift": bm["lift_vs_base_rate"],
                            "envelope": bm["envelope"]},
                        "telemetry": {
                            prm: {
                                "value": _val(colvals[vc][i]),
                                "age_min": _val(colvals[ac][i]) if ac else None,
                                "measured": (bool(colvals[oc][i] > 0.5)
                                             if oc else None),
                                "source": _source(
                                    oc and colvals[oc][i] > 0.5,
                                    bool(lc) and np.isnan(colvals[lc][i]))}
                            for prm, vc, ac, oc, lc in tele},
                        "reasons": [{"code": "MODEL_BAND", "band": v.displayed}],
                        "contributors": [
                            {"feature": cols[j], "value": _val(colvals[cols[j]][i]),
                             "contribution": round(float(contribs[i, j]), 8),
                             "kind": kind[cols[j]], "group": group[cols[j]]}
                            for j in order[i]],
                        "contributors_other": round(float(tail[i]), 8),
                        "contributors_bias": round(float(bias[i]), 8),
                        "documentation_share": round(float(doc_share[i]), 4),
                        "imputed_share": round(float(imputed_share[i]), 4),
                        "attribution_age_min": _val(attr_age[i]),
                        # The denominator the three shares above are taken over.
                        # Without it a consumer holding only the top-8 cannot
                        # express one contributor as a share of the whole model
                        # decision -- only as a share of the 8 it can see, which
                        # overstates every one of them.
                        "attribution_total": round(float(tot[i]), 8),
                    }, separators=(",", ":")) + "\n")
                    n += 1
        gz = C.RECORDS_JSONL.stat().st_size
        raw = sum(len(l) for l in gzip.open(C.RECORDS_JSONL, "rb"))
        # Both sizes, because quoting only the compressed one understates the
        # file 21x and anyone who decompresses it to look gets a surprise.
        log(f"  {n:,} records   {gz / 1e6:,.1f} MB gzipped / "
            f"{raw / 1e6:,.1f} MB raw ({raw / max(n, 1):,.0f} bytes per record)")
        log(f"  read it streaming -- [bold]one JSON object per line[/bold]; "
            f"json.load() on the whole file raises 'Extra data'")

    # ---------------------------------------------------------- aggregates
    with stage("Aggregates -> reports/records.json (no per-patient data)"):
        top = sorted(top_counts.items(), key=lambda kv: -kv[1])[:15]
        for f, c in top[:8]:
            log(f"  top contributor {f:<42} {100 * c / n:5.1f}% of readings  "
                f"[{kind[f]}]")
        doc_top = sum(c for f, c in top_counts.items()
                      if kind[f] == "documentation") / n
        log(f"  readings whose TOP contributor is a documentation feature: "
            f"{100 * doc_top:.1f}%")
        worst_ref = sorted(src_stats.items(),
                           key=lambda kv: -kv[1]["population_reference"])[:4]
        for prm, s in worst_ref:
            log(f"  {prm:<24} measured {s['measured']:6.1%}  carried "
                f"{s['carried_forward']:6.1%}  [red]population "
                f"{s['population_reference']:6.1%}[/red]")
        report.update(
            records=n, admissions=n_sub, admissions_available=len(g_te),
            admissions_dropped=len(g_te) - n_sub,
            bytes_gzipped=int(C.RECORDS_JSONL.stat().st_size),
            bytes_uncompressed=int(raw),
            format="gzipped JSONL -- one complete JSON object per line; parse "
                   "with json.loads() per line, not json.load() on the file",
            schema_version=art["schema_version"],
            reconstruction=dict(max=float(err.max()), mean=float(err.mean()),
                                tolerance=C.CONTRIB_RECON_TOL),
            documentation_share=dict(
                median=float(np.median(doc_share)),
                p90=float(np.percentile(doc_share, 90)),
                mean=float(doc_share.mean()),
                top_contributor_is_documentation=float(doc_top)),
            imputed_share=dict(
                median=float(np.median(imputed_share)),
                p90=float(np.percentile(imputed_share, 90)),
                p99=float(np.percentile(imputed_share, 99)),
                mean=float(imputed_share.mean()),
                max=float(imputed_share.max())),
            attribution_age_min=dict(
                median=float(np.median(_fin)),
                p90=float(np.percentile(_fin, 90)),
                p99=float(np.percentile(_fin, 99)),
                undefined=int(len(rows) - len(_fin))),
            top_contributors=[{"feature": f, "share_of_readings": c / n,
                               "kind": kind[f], "group": group[f]}
                              for f, c in top],
            telemetry_provenance=src_stats,
            seconds=round(time.time() - t0, 1))
        REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"  {REPORT_JSON.name} written")


def _val(x):
    """JSON has no NaN. A missing value must serialise as null, not as a bare
    NaN token that json.loads hands back as float('nan') downstream -- which is
    how an absent measurement becomes a number in a prompt.

    Categorical features stay strings: "Pressure Support" is what an explanation
    needs, not the code the model happened to assign it.
    """
    if x is None:
        return None
    if isinstance(x, str):
        return x
    f = float(x)
    return None if np.isnan(f) else round(f, 4)


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="re-emit even when the manifest is current -- needed "
                          "after a code change, which no fingerprint covers")
    main(force=_ap.parse_args().force)
