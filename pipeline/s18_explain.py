"""Stage 18 -- resolve every record to an explanation, and prove the checker works.

s17 emits records; this stage decides what may be said about each one. It is
model-independent on purpose. No inference stack is installed in this
environment, the golden set is DUA-covered so it cannot be sent to a hosted API,
and the payload's token cost is what should decide the model size -- so the two
pieces built here are the ones that ship whatever model arrives later:

  the payload contract   record -> exactly what a generator is allowed to see
  the grounding verifier text -> what it contradicts in the record

THE BASELINE IS A FLOOR, NOT A FALLBACK
---------------------------------------
explain.baseline() renders a grounded sentence from the payload. It exists to be
the thing a generated explanation has to beat, and to exercise the checker. It is
NOT what gets served when a generator is down: that path returns
EXPLANATION_UNAVAILABLE_TEXT, because fluent clinical prose in a slot meant to
signal absence defeats the purpose of the slot. Worth stating plainly -- this
project has already shipped a template as an explanation once
(legacy/lora_adapters/, a LoRA tuned on four hardcoded paragraphs over seven
patients, still mounted by docker-compose.yml).

WHY THE MUTATION SUITE EXISTS
-----------------------------
The baseline is built from the payload, so it passes the checker by
construction. That is close to tautological and proves nothing about the
checker. The suite below deliberately corrupts baseline text -- swaps the band,
injects a trend claim, quotes a value the patient never had, fabricates a number
-- and fails the stage if any corruption goes uncaught. It also asserts the
FALSE-POSITIVE direction: forward-looking language must NOT be flagged, because
the label is a 6 h forward risk and "risk of deterioration" is the band's
meaning rather than a claim about the past.

There is no pytest and no CI in this project; verification means running the
thing and reading the output, so the suite lives inside the stage.

DUA
---
Explanations quote per-patient MIMIC values and are keyed to stay_id. They go to
build/, never to reports/, which is tracked.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import Counter

import numpy as np

from . import config as C
from . import explain as E
from . import grounding as G
from .common import cached_stage, log, stage

REPORT_JSON = C.REPORTS / "explanations.json"


def policy() -> E.Policy:
    return E.Policy(
        display=C.PARAM_DISPLAY,
        max_imputed_share=C.SUFFICIENCY_MAX_IMPUTED_SHARE,
        max_doc_share=C.SUFFICIENCY_MAX_DOC_SHARE,
        insufficient_text=C.INSUFFICIENT_DATA_TEXT,
        unavailable_text=C.EXPLANATION_UNAVAILABLE_TEXT,
        contributor_k=C.EXPLAIN_CONTRIBUTOR_K)


def records(limit: int | None = None):
    with gzip.open(C.RECORDS_JSONL, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def check(record: dict, text: str) -> list[G.Finding]:
    return G.check(record, text, display=C.PARAM_DISPLAY,
                   band_names=C.BAND_NAMES)


# --------------------------------------------------------------------------
# Adversarial mutations. Each returns (mutated_text, code_that_must_fire) or
# None when the mutation does not apply to this record.
# --------------------------------------------------------------------------
def mut_band(rec, text):
    shown = rec["band"]["displayed"]
    other = next(b for b in C.BAND_NAMES if b != shown)
    return f"{text} Overall assessment: {other}.", G.BAND_MISMATCH


def mut_trend(rec, text):
    return f"{text} FiO2 has been climbing through the shift.", G.TREND_CLAIM


def mut_imputed(rec, text):
    for p, v in rec["telemetry"].items():
        if v.get("source") == "population_reference" and v.get("value") is not None:
            label, unit, dec = C.PARAM_DISPLAY[p]
            return (f"{text} {label} is {v['value']:.{dec}f} {unit}.".strip(),
                    G.IMPUTED_QUOTED)
    return None


def _absent(rec, start: int = 7777) -> int:
    """An integer this record does not entitle anyone to state.

    Searched rather than assumed: a mutation that happens to land on a real
    quantity would be legitimately uncaught, and would fail the stage for the
    wrong reason.
    """
    allowed = G.allowed_numbers(rec, C.PARAM_DISPLAY)
    n = start
    while any(round(a) == n for a in allowed):
        n += 1
    return n


def mut_number(rec, text):
    m = G._NUMBER.search(text)
    if not m:
        return None
    return (text[:m.start()] + str(_absent(rec)) + text[m.end():],
            G.FABRICATED_NUMBER)


def mut_wrong_value(rec, text):
    """A right parameter with a wrong number -- the most dangerous single error
    an explanation can make, and the one a bare number-whitelist misses."""
    for p, v in rec["telemetry"].items():
        val = v.get("value")
        if v.get("source") == "population_reference" or not isinstance(val, (int, float)):
            continue
        label = C.PARAM_DISPLAY[p][0]
        return f"{text} {label} is {_absent(rec, int(val) + 137)}.", None
    return None


def _param_not_used(rec):
    used = {p for c in rec["contributors"] for p in C.PARAM_DISPLAY
            if c["feature"] == p or c["feature"].startswith(p + "_")}
    return next((p for p in C.PARAM_DISPLAY if p not in used), None)


def mut_unsupported(rec, text):
    p = _param_not_used(rec)
    if p is None:
        return None
    return (f"{text} {C.PARAM_DISPLAY[p][0]} is driving this assessment.",
            G.UNSUPPORTED_DRIVER)


def mut_doc_as_physiology(rec, text):
    """A documentation feature narrated in a clinical voice -- the
    `spo2_delta_t_min` -> "oxygenation is monitored less often" failure, which is
    the reason `kind` is on every contributor in the first place."""
    def params(kind):
        return {p for c in rec["contributors"] if c["kind"] == kind
                for p in C.PARAM_DISPLAY
                if c["feature"] == p or c["feature"].startswith(p + "_")}
    only_doc = params("documentation") - params("physiology")
    if not only_doc:
        return None
    label = C.PARAM_DISPLAY[sorted(only_doc)[0]][0]
    return (f"{text} {label} is abnormal and contributes to this assessment.",
            G.DOC_AS_PHYSIOLOGY)


def mut_soft_trend(rec, text):
    return f"{text} The estimate is higher than earlier.", G.POSSIBLE_TREND_CLAIM


def mut_lowercase_band(rec, text):
    shown = rec["band"]["displayed"]
    other = next(b for b in C.BAND_NAMES if b != shown)
    return (f"{text} Overall the patient is at {other.lower()} risk.",
            G.POSSIBLE_BAND_MISMATCH)


def mut_treatment(rec, text):
    return f"{text} Consider titrating support.", G.TREATMENT_LANGUAGE


# Every check in grounding.py appears here. A check with no mutation behind it
# has no evidence it fires at all, and a verifier nobody has seen fail is
# indistinguishable from one that always passes.
MUTATIONS = (("band swapped", mut_band),
             ("trend claim injected", mut_trend),
             ("imputed value quoted", mut_imputed),
             ("number fabricated", mut_number),
             ("wrong value for right parameter", mut_wrong_value),
             ("unsupported driver named", mut_unsupported),
             ("documentation as physiology", mut_doc_as_physiology),
             ("soft trend language", mut_soft_trend),
             ("lowercase band name", mut_lowercase_band),
             ("treatment language", mut_treatment))

# Must NOT fire: the label is a forward risk, so forward-looking language is the
# band's meaning rather than an unsupported claim about the past.
FORWARD_PHRASES = (
    "This indicates a risk of respiratory deterioration within the next 6 hours.",
    "The model estimates elevated risk of deterioration over the coming 6 hours.",
    "This is a forward-looking estimate, not a description of the past.")


def self_check(sample: list[dict], pol: E.Policy) -> dict:
    """Corrupt good text and require the checker to catch it."""
    fired = Counter()
    missed = Counter()
    applied = Counter()
    false_pos = 0
    checked = 0

    for rec in sample:
        if not E.sufficiency(rec, pol).ok:
            continue
        text = E.baseline(rec, pol)
        base_codes = {f.code for f in check(rec, text)}
        checked += 1

        for name, fn in MUTATIONS:
            out = fn(rec, text)
            if out is None:
                continue
            mutated, want = out
            applied[name] += 1
            codes = {f.code for f in check(rec, mutated)}
            new_violations = {f.code for f in check(rec, mutated)
                              if f.severity == G.VIOLATION} - base_codes
            if want is None:
                # mut_wrong_value: either binding check may catch it, but
                # SOMETHING must.
                ok = bool(new_violations)
            else:
                ok = want in codes and want not in base_codes
            (fired if ok else missed)[name] += 1

        # False-positive direction.
        for phrase in FORWARD_PHRASES:
            after = {f.code for f in check(rec, f"{text} {phrase}")
                     if f.severity == G.VIOLATION}
            if after - base_codes:
                false_pos += 1

    return {"records_checked": checked, "applied": dict(applied),
            "caught": dict(fired), "missed": dict(missed),
            "forward_language_false_positives": false_pos}


def main(force: bool = False) -> None:
    with cached_stage("s18_explain",
                      sources=[C.RECORDS_JSONL],
                      output=C.EXPLANATIONS_JSONL, force=force,
                      extra=C.FP_EXPLAIN) as ran:
        if not ran:
            return
        _run()


def _run() -> None:
    report: dict = {}
    t0 = time.time()
    pol = policy()

    # ------------------------------------------------ prove the checker first
    with stage("Adversarial self-check -- corrupt good text, require a catch"):
        sample = list(records(C.EXPLAIN_MUTATION_SAMPLE))
        sc = self_check(sample, pol)
        for name, _fn in MUTATIONS:
            a, c = sc["applied"].get(name, 0), sc["caught"].get(name, 0)
            mark = "[green]" if a and c == a else "[red]"
            log(f"  {mark}{c:>4,}/{a:<5,}[/] {name}")
        log(f"  forward-looking language falsely flagged: "
            f"{sc['forward_language_false_positives']}")
        # Assertion 1: a mutation that goes uncaught means the checker does not
        # do the thing this whole stage is built around.
        assert not sc["missed"], (
            f"the grounding checker missed corrupted text: {sc['missed']}. "
            f"Every mutation must be caught or the verifier is decorative.")
        # Assertion 2: the checker must not punish the label's own semantics.
        assert sc["forward_language_false_positives"] == 0, (
            f"{sc['forward_language_false_positives']} forward-looking "
            f"statements were flagged as violations. The label IS a 6 h forward "
            f"risk; flagging its meaning would force generators to omit it.")

    # ------------------------------------- the two non-generating paths exist
    with stage("Fallback paths"):
        good = next(r for r in sample if E.sufficiency(r, pol).ok)
        bad = next((r for r in records() if not E.sufficiency(r, pol).ok), None)
        assert bad is not None, "no record below the floor -- cannot exercise it"
        una = E.explain(good, pol, generator=None)
        ins = E.explain(bad, pol, generator=lambda pl: "should never be called")
        # Assertion 3: an insufficient record must never reach a generator.
        assert una["status"] == E.GENERATOR_UNAVAILABLE
        assert una["text"] == C.EXPLANATION_UNAVAILABLE_TEXT
        assert ins["status"] == E.INSUFFICIENT_DATA
        assert ins["text"] == C.INSUFFICIENT_DATA_TEXT, (
            "an insufficient record produced generated text -- the gate in "
            "build_payload is not upstream of the generator")
        try:
            E.build_payload(bad, pol)
            raise AssertionError("build_payload accepted a record below the floor")
        except E.InsufficientData:
            pass
        log(f"  below the floor -> [bold]{ins['text']}[/bold]  {ins['reasons']}")
        log(f"  generator down  -> [bold]{una['text']}[/bold]  "
            f"(band, risk and telemetry still emit)")

    # -------------------------------------------------------------- emit
    with stage(f"Explain every record -> {C.EXPLANATIONS_JSONL.name}"):
        status_n = Counter()
        reason_n = Counter()
        finding_n = Counter()
        sup_band = Counter()
        all_band = Counter()
        lengths = []
        worst_examples: dict[str, dict] = {}
        n = 0
        with gzip.open(C.EXPLANATIONS_JSONL, "wt", encoding="utf-8") as fh:
            for rec in records():
                n += 1
                all_band[rec["band"]["displayed"]] += 1
                suf = E.sufficiency(rec, pol)
                if suf.ok:
                    text = E.baseline(rec, pol)
                    blk = E.explain(rec, pol,
                                    generator=lambda _pl, t=text: t,
                                    generator_name="baseline")
                    findings = check(rec, text)
                    lengths.append(len(text))
                else:
                    blk = E.explain(rec, pol)          # no generator -> refused
                    text, findings = blk["text"], []
                    sup_band[rec["band"]["displayed"]] += 1
                status_n[blk["status"]] += 1
                for r in suf.reasons:
                    reason_n[r] += 1
                for f in findings:
                    finding_n[(f.code, f.severity)] += 1
                    # Code and severity ONLY. A finding's message and evidence
                    # quote the explanation, which quotes this patient's values,
                    # and reports/ is tracked. The full findings live beside the
                    # text they describe in build/, which is not.
                    worst_examples.setdefault(f.code, {"code": f.code,
                                                       "severity": f.severity})
                fh.write(json.dumps({
                    "schema_version": C.EXPLAIN_SCHEMA_VERSION,
                    "record_schema_version": rec["schema_version"],
                    "provenance": rec["provenance"],
                    "stay_id": rec["stay_id"],
                    "charttime": rec["charttime"],
                    "band": rec["band"]["displayed"],
                    "status": blk["status"],
                    "generator": blk["generator"],
                    "reasons": blk["reasons"],
                    "text": blk["text"],
                    "findings": [f.to_json() for f in findings],
                }, separators=(",", ":")) + "\n")
        gz = C.EXPLANATIONS_JSONL.stat().st_size
        log(f"  {n:,} explanations   {gz / 1e6:,.1f} MB gzipped")
        for s, k in status_n.most_common():
            log(f"  {s:<24} {k:>7,}  {k / n:6.1%}")

    # -------------------------------------------------------- aggregates
    with stage("Aggregates -> reports/explanations.json (no per-patient data)"):
        viol = {c: k for (c, sev), k in finding_n.items() if sev == G.VIOLATION}
        warn = {c: k for (c, sev), k in finding_n.items() if sev == G.WARNING}
        # Assertion 4: the baseline is built from the payload, so a violation
        # against its own record is a real defect in one of the two modules --
        # never a reason to relax the check.
        assert not viol, (
            f"the template baseline violated its own record: {viol}. Either the "
            f"baseline states something the record does not support, or the "
            f"checker is wrong. Fix whichever it is; do not loosen the rule.")
        for c, k in sorted(warn.items(), key=lambda kv: -kv[1]):
            log(f"  warning  {c:<26} {k:>7,}  {k / max(n, 1):6.2%} of readings")
        if not warn:
            log("  no warnings raised")
        log(f"  suppressed by band: " + "  ".join(
            f"{b} {sup_band[b]:,}/{all_band[b]:,} ({sup_band[b] / max(all_band[b], 1):.1%})"
            for b in C.BAND_NAMES))
        ln = np.array(lengths) if lengths else np.array([0])
        report.update(
            explanations=n,
            schema_version=C.EXPLAIN_SCHEMA_VERSION,
            record_schema_version=sample[0]["schema_version"],
            format="gzipped JSONL -- one complete JSON object per line; parse "
                   "with json.loads() per line, not json.load() on the file",
            bytes_gzipped=int(gz),
            status=dict(status_n),
            insufficient_reasons=dict(reason_n),
            floor=dict(max_imputed_share=C.SUFFICIENCY_MAX_IMPUTED_SHARE,
                       max_doc_share=C.SUFFICIENCY_MAX_DOC_SHARE,
                       staleness_gate=None,
                       staleness_note=(
                           "deliberately absent -- readings whose driving "
                           "values are >48 h old carry 1.38x the forward-label "
                           "prevalence of the rest, so gating on staleness "
                           "would suppress readings at ABOVE the base "
                           "deterioration rate. Age is disclosed per parameter "
                           "instead.")),
            suppressed_by_band={b: {"suppressed": sup_band[b],
                                    "total": all_band[b],
                                    "share": sup_band[b] / max(all_band[b], 1)}
                                for b in C.BAND_NAMES},
            violations=viol,
            warnings=warn,
            warning_codes_seen=[worst_examples[c] for c in warn],
            baseline_length=dict(median=float(np.median(ln)),
                                 p10=float(np.percentile(ln, 10)),
                                 p90=float(np.percentile(ln, 90))),
            self_check=sc,
            fallback_text=dict(insufficient=C.INSUFFICIENT_DATA_TEXT,
                               generator_unavailable=C.EXPLANATION_UNAVAILABLE_TEXT),
            seconds=round(time.time() - t0, 1))
        REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"  {REPORT_JSON.name} written")


def show(k: int) -> None:
    """Print payloads and baselines. The real check on whether the contract says
    enough to explain a reading is a judgement, not an assertion."""
    pol = policy()
    shown = 0
    for rec in records():
        suf = E.sufficiency(rec, pol)
        print("=" * 78)
        print(f"stay {rec['stay_id']}  {rec['charttime']}  "
              f"band {rec['band']['displayed']}  risk "
              f"{rec['risk']['calibrated']:.4f}")
        if not suf.ok:
            print(f"\n  {C.INSUFFICIENT_DATA_TEXT}   {list(suf.reasons)}")
            print(f"  imputed_share {rec['imputed_share']}  "
                  f"documentation_share {rec['documentation_share']}")
        else:
            print("\n--- payload ---")
            print(json.dumps(E.build_payload(rec, pol), indent=1))
            text = E.baseline(rec, pol)
            print("\n--- baseline ---")
            print(text)
            f = check(rec, text)
            print(f"\n--- findings: {len(f)} ---")
            for x in f:
                print(f"  [{x.severity}] {x.code}: {x.message}")
        shown += 1
        if shown >= k:
            return


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--force", action="store_true",
                     help="re-emit even when the manifest is current -- needed "
                          "after a code change, which no fingerprint covers")
    _ap.add_argument("--sample", type=int, metavar="N",
                     help="print N payloads and baselines instead of running")
    _a = _ap.parse_args()
    if _a.sample:
        show(_a.sample)
    else:
        main(force=_a.force)
