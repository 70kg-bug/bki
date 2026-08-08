"""The explanation contract -- pure logic. No I/O, no config reads, no numpy.

Importable by serving without dragging in the pipeline, exactly as bands.py is.
The policy it needs (display names, the data floor, the fallback strings) arrives
as a Policy argument rather than being read from config, for the same reason
BandMachine takes its cuts instead of looking them up.

WHAT THIS MODULE DECIDES
------------------------
Three outcomes, and they are different conditions with different causes:

  ok                     the reading is fit to explain and a generator produced
                         text for it
  insufficient_data      the reading does not clear the data floor. Pulsemind
                         says "Insufficient data" and explains nothing. LEGAL:
                         degrade explicitly -- never put a synthesised rationale
                         in a slot that is meant to signal absence.
  generator_unavailable  the reading is fine and the generator is down. Band,
                         risk, telemetry and contributors all still emit; only
                         the narrative is withheld.

THE GATE IS UPSTREAM OF EVERY GENERATOR
---------------------------------------
build_payload() raises on an insufficient record, so no generator -- template,
LLM, or anything added later -- can be handed one. A rule enforced at the point
where the data is assembled cannot be forgotten by a caller; a rule enforced in
each generator has to be re-implemented every time one is added.

WHAT THE PAYLOAD WITHHOLDS, AND WHY
-----------------------------------
A parameter whose source is `population_reference` reaches the payload WITHOUT
ITS VALUE. s07_impute defines `_final = _locf.fill_null(cohort_reference)`, so
where the patient has no value at all the number is a cohort statistic -- 82.6%
of EtCO2 readings, 79% of the I:E ratios, 61% of flow rate. A generator that
cannot see the number cannot quote it, which retires a whole class of failure
rather than leaving it to be detected afterwards. What is lost -- that the model
did lean on it -- comes back as `imputed: true` on the contributor.

THERE ARE NO TREND FEATURES
---------------------------
All 109 features are point-in-time values plus staleness. "FiO2 has been
climbing" is unsupported by the record however naturally it reads, so the
payload carries an explicit no_trend_data marker and the system prompt forbids
backward-looking trajectory claims. grounding.py enforces it independently.
"""
from __future__ import annotations

from dataclasses import dataclass

# Why a reading was refused. Enumerated, never prose -- the same reason codes go
# into reports and into the record, and a free-text reason cannot be counted.
IMPUTED_SHARE_ABOVE_FLOOR = "IMPUTED_SHARE_ABOVE_FLOOR"
DOC_SHARE_ABOVE_FLOOR = "DOC_SHARE_ABOVE_FLOOR"

OK = "ok"
INSUFFICIENT_DATA = "insufficient_data"
GENERATOR_UNAVAILABLE = "generator_unavailable"
STATUSES = (OK, INSUFFICIENT_DATA, GENERATOR_UNAVAILABLE)

# The five derived forms of every frozen parameter, and what each one IS in
# words. Rendering `peep_locf` as "PEEP" would be a lie of omission: the value is
# the last one charted, not a reading taken now.
SUFFIX_PHRASE = {
    "_final": "{p}",
    "_locf": "{p} (last charted value)",
    "_observed": "whether {p} was charted at this reading",
    "_delta_t_min": "time since {p} was last charted",
    "_structurally_missing_in_stay": "whether {p} was ever charted in this stay",
}
# Forms that carry the parameter's VALUE, as opposed to describing its charting.
# Only these can be imputed -- `spo2_observed = 0` is true whether or not the
# patient was ever measured.
VALUE_SUFFIXES = ("_final", "_locf")


class InsufficientData(ValueError):
    """Raised by build_payload for a record below the floor.

    An exception rather than a None return: a caller that ignores it gets a
    traceback, where a caller that ignores a None gets an explanation built from
    an empty payload.
    """


@dataclass(frozen=True, slots=True)
class Policy:
    """Everything this module needs that is a choice rather than a fact."""
    display: dict          # param -> (label, unit, decimals)
    max_imputed_share: float
    max_doc_share: float
    insufficient_text: str
    unavailable_text: str
    contributor_k: int = 5


@dataclass(frozen=True, slots=True)
class Sufficiency:
    ok: bool
    reasons: tuple[str, ...]


# --------------------------------------------------------------------------
# The floor
# --------------------------------------------------------------------------
def sufficiency(record: dict, policy: Policy) -> Sufficiency:
    """Is this reading substantially an assessment of THIS patient?

    Both signals are shares of the same |attribution| denominator over disjoint
    feature sets (see s17_records): how much of the score came from cohort
    defaults the patient never had, and how much came from charting behaviour
    rather than physiology. Above either threshold the score is largely not
    about the patient in front of the clinician.

    Deliberately NOT gated on staleness. `attribution_age_min` is in the record
    and reads like an obvious third condition, but measured on the golden set
    readings whose driving values are more than 48 h old carry 1.38x the forward
    label prevalence of the rest -- gating on it would suppress readings at
    ABOVE the base deterioration rate. Age is disclosed per parameter in the
    payload instead.
    """
    reasons = []
    if float(record.get("imputed_share") or 0.0) > policy.max_imputed_share:
        reasons.append(IMPUTED_SHARE_ABOVE_FLOOR)
    if float(record.get("documentation_share") or 0.0) > policy.max_doc_share:
        reasons.append(DOC_SHARE_ABOVE_FLOOR)
    return Sufficiency(not reasons, tuple(reasons))


# --------------------------------------------------------------------------
# Rendering helpers -- deterministic, so the verifier can re-derive them
# --------------------------------------------------------------------------
def split_feature(feature: str, display: dict) -> tuple[str | None, str | None]:
    """(parameter, suffix) for a frozen-parameter feature, (None, None) else.

    Longest suffix first: `spo2_structurally_missing_in_stay` also ends in
    nothing else, but `_final` vs `_delta_t_min` would collide on a naive scan
    if a parameter name ever ended in one of them.
    """
    for suf in sorted(SUFFIX_PHRASE, key=len, reverse=True):
        if feature.endswith(suf):
            p = feature[: -len(suf)]
            if p in display:
                return p, suf
    return None, None


def feature_label(feature: str, display: dict) -> str:
    """A phrase a clinician can read, or the raw name when there isn't one.

    Static and intervention features keep their column names with underscores
    opened up. Inventing a clinical gloss for `first_careunit` would be writing
    an interpretation into what is supposed to be a transcription.
    """
    p, suf = split_feature(feature, display)
    if p is not None:
        return SUFFIX_PHRASE[suf].format(p=display[p][0])
    return feature.replace("_", " ")


def age_phrase(age_min: float | None) -> str:
    """Staleness in words. The unit changes with the magnitude because '2,880
    min' is a number a reader has to convert and '2.0 days' is one they don't."""
    if age_min is None:
        return "age not available"
    if age_min < 1:
        return "measured at this reading"
    if age_min < 90:
        return f"last charted {age_min:.0f} min ago"
    if age_min < 2880:
        return f"last charted {age_min / 60:.1f} h ago"
    return f"last charted {age_min / 1440:.1f} days ago"


def _pct(x: float | None) -> str | None:
    """A fraction as a percentage string, rounded ONCE, here.

    One decimal place because that is what grounding.allowed_numbers will accept
    back: it matches a stated number against any record quantity rounded to the
    precision the text used, so a value rendered at 1 dp always round-trips.
    """
    return None if x is None else f"{x * 100:.1f}%"


def format_value(param: str, value, display: dict) -> str:
    label, unit, dec = display[param]
    if value is None:
        return "not charted"
    return f"{value:.{dec}f}{(' ' + unit) if unit else ''}"


# --------------------------------------------------------------------------
# The payload
# --------------------------------------------------------------------------
def build_payload(record: dict, policy: Policy) -> dict:
    """Exactly what a generator is allowed to see. Raises InsufficientData."""
    suf = sufficiency(record, policy)
    if not suf.ok:
        raise InsufficientData(
            f"record does not clear the data floor: {', '.join(suf.reasons)}")

    d = policy.display
    band, prov = record["band"], record["provenance"]
    total = float(record.get("attribution_total") or 0.0)

    # Which parameters are a cohort default for this patient. Needed twice: to
    # withhold the value below, and to flag the contributors that lean on it.
    tel = record["telemetry"]
    imputed_params = {p for p, v in tel.items()
                      if v.get("source") == "population_reference"}

    telemetry = []
    for p in d:
        if p not in tel:
            continue
        v = tel[p]
        label, unit, _ = d[p]
        src = v.get("source")
        if src == "population_reference":
            # No value, on purpose. See the module docstring.
            telemetry.append({
                "parameter": label, "unit": unit, "value": None,
                "status": "not_charted_for_this_patient",
                "note": ("this patient has no charted value; the model used a "
                         "cohort default, which is not an observation of them")})
        else:
            telemetry.append({
                "parameter": label, "unit": unit,
                "value": v["value"],
                "formatted": format_value(p, v["value"], d),
                "status": "measured" if src == "measured" else "carried_forward",
                "age_min": v.get("age_min"),
                "age": age_phrase(v.get("age_min"))})

    contributors = []
    for c in record["contributors"][: policy.contributor_k]:
        p, suffix = split_feature(c["feature"], d)
        imputed = bool(p in imputed_params and suffix in VALUE_SUFFIXES)
        # `{p}_locf` normally renders as "(last charted value)", which is a lie
        # when there IS no charted value: for an imputed parameter _locf is null
        # and _final is a cohort statistic. A generator handed that label will
        # faithfully repeat it, and no numeric check can catch a false claim
        # made entirely in words.
        label = (f"{d[p][0]} (no charted value for this patient; cohort default)"
                 if imputed else feature_label(c["feature"], d))
        contributors.append({
            "feature": c["feature"],
            "label": label,
            "direction": "raises" if c["contribution"] > 0 else "lowers",
            # NAMED share_of_score, not `share`. It is the fraction of the
            # model's decision this feature accounts for. It is NOT a change in
            # the patient's risk, and "lowers the risk by 18%" is how a reader
            # will take it unless the field says otherwise.
            #
            # PRE-FORMATTED, deliberately. Handing over 0.1027 and asking for a
            # percentage makes the generator do arithmetic, and a generator
            # doing arithmetic gets it wrong: measured at n=200, it rendered
            # 0.1027 as "10.28%" and produced a number that is in no field of
            # the record. Every contributor share in that same sentence was
            # correct -- the errors are conversions, not inventions. Never ask
            # it to compute anything that can be computed exactly here.
            "share_of_score": _pct(
                (abs(c["contribution"]) / total) if total > 0 else 0.0),
            "value": c["value"],
            "kind": c["kind"],
            # The model leaned on a cohort default here. Not the same as a
            # documentation feature and not the same as physiology: it is the
            # absence of this patient's data acting as if it were data.
            "imputed": imputed,
        })

    return {
        "band": {
            "name": band["displayed"],
            "state": band["state"],
            "readings_in_state": band["readings_in_state"],
            "observed_rate": band["observed_rate"],
            "base_rate": band["base_rate"],
            "lift": band["lift"],
            "envelope": band["envelope"],
            "meaning": (
                f"{band['observed_rate']:.1%} of readings in this band met the "
                f"{prov['horizon_hours']}-hour {prov['arm']} deterioration "
                f"label, against {band['base_rate']:.1%} across the cohort"),
        },
        "risk": record["risk"],
        "label": {"arm": prov["arm"], "horizon_hours": prov["horizon_hours"]},
        "telemetry": telemetry,
        "contributors": contributors,
        "contributors_shown": len(contributors),
        "contributors_total": len(record["contributors"]),
        # Pre-formatted for the same reason as share_of_score above.
        "documentation_share": _pct(record["documentation_share"]),
        "imputed_share": _pct(record.get("imputed_share")),
        "constraints": {
            "no_trend_data": True,
            "note": ("every input is point-in-time plus staleness -- there are "
                     "no deltas or slopes, so no claim may be made about which "
                     "direction any value has been moving"),
        },
    }


# The prohibitions alone produced text that broke no rule and said very little:
# it never named the band, quoted no contributor value or share, and dropped
# both the staleness and the documentation share. Zero violations, strictly less
# informative than the template. The checker catches FALSE statements, not
# MISSING ones, so completeness has to be demanded rather than assumed.
SYSTEM_PROMPT = """\
You explain a ventilation risk assessment to an ICU clinician. You are given a \
structured record and you may state ONLY what is in it.

YOU MUST INCLUDE, in this order:

1. The band, named exactly as it appears in the record, and what it means: the \
percentage of readings in that band that met the label, against the cohort \
percentage.
2. The two or three largest contributors. Give each one's value and the share \
of the score it accounts for. A share is how much of the model's decision that \
feature accounts for; it is NOT a change in the patient's risk. Write "accounts \
for X% of the score", never "lowers the risk by X%". If a contributor is marked \
imputed, say the model used a cohort default there and give NO value for it.
3. Any parameter carried forward rather than measured: give its value and how \
old it is.
4. Any parameter not charted for this patient, by name.
5. The documentation share: how much of the score came from charting patterns \
rather than physiology.

Every percentage in the record is already written out for you. Quote it exactly \
as it appears. Do NOT convert, re-round, add or average any number: the record \
is arithmetic-free by design and every figure you need is present in the form \
you should say it.

Name a parameter as a contributor ONLY if it appears in the contributors list, \
and give each one its own share exactly as written. Never group several \
parameters behind one figure, never combine or apportion shares between them, \
and never say "respectively".

YOU MUST NOT, and all of this is checked mechanically after you answer:

- State any number that is not in the record.
- Say which direction any value has been moving. The model has no trend inputs, \
so "rising", "falling", "worsening" and "improving" are unsupported. The risk \
itself is forward-looking; the measurements are not.
- Give a value for a parameter marked not_charted_for_this_patient.
- Name any band other than the one in the record.
- Describe a value charted hours ago as a current reading.

Write three to five plain sentences for a clinician reading at the bedside. No \
preamble, no bullet points, no restating these instructions.\
"""


def render_prompt(payload: dict) -> list[dict]:
    """Chat messages. The system prompt states the rules grounding.py enforces --
    two expressions of one contract, deliberately kept side by side."""
    import json
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=1)}]


# --------------------------------------------------------------------------
# The template baseline -- the floor an LLM has to beat, and NOT a fallback
# --------------------------------------------------------------------------
def baseline(record: dict, policy: Policy) -> str:
    """A grounded sentence built only from the payload.

    This is the comparison floor for a generated explanation. It is deliberately
    NOT what gets served when a generator is down -- that path returns
    policy.unavailable_text, because a fallback slot that emits fluent clinical
    prose defeats the purpose of having one.
    """
    pl = build_payload(record, policy)
    b, d = pl["band"], policy.display
    parts = [f"{b['name']}. {b['meaning']}."]

    if b["state"] == "confirmed":
        n = b["readings_in_state"]
        parts.append(f"Band confirmed over {n} consecutive "
                     f"{'reading' if n == 1 else 'readings'}.")
    elif b["state"] == "provisional":
        parts.append("A higher band is pending confirmation.")
    else:
        parts.append("The band is being held above the current score while it "
                     "clears the demotion deadband.")

    drivers = [c for c in pl["contributors"] if c["kind"] == "physiology"]
    if drivers:
        bits = []
        for c in drivers[:3]:
            v = c["value"]
            if c["imputed"]:
                # The model DID lean on this, and the number it leaned on is a
                # cohort statistic. Naming the parameter is honest; printing the
                # number would state a population fact as this patient's.
                shown, imp = "", ", on a cohort default rather than a charted value"
            else:
                shown = f" {v:g}" if isinstance(v, (int, float)) else (
                    f" {v}" if isinstance(v, str) else "")
                imp = ""
            bits.append(f"{c['label']}{shown} ({c['direction']} the score, "
                        f"{c['share_of_score']} of it{imp})")
        parts.append("Weighted most: " + "; ".join(bits) + ".")

    stale = [t for t in pl["telemetry"]
             if t["status"] == "carried_forward" and (t["age_min"] or 0) >= 90]
    if stale:
        parts.append("Carried forward rather than measured now: "
                     + "; ".join(f"{t['parameter']} {t['formatted']}, {t['age']}"
                                 for t in stale[:3]) + ".")

    gaps = [t["parameter"] for t in pl["telemetry"]
            if t["status"] == "not_charted_for_this_patient"]
    if gaps:
        parts.append(f"No charted value for this patient: {', '.join(gaps)} -- "
                     f"the model used cohort defaults there.")

    doc = pl["documentation_share"]
    if doc is not None:
        parts.append(f"Documentation features rather than physiology account "
                     f"for {doc} of this score.")
    return " ".join(parts)


# --------------------------------------------------------------------------
# The whole decision, in one place that serving and s18 both call
# --------------------------------------------------------------------------
def explain(record: dict, policy: Policy, *, generator=None,
            generator_name: str | None = None) -> dict:
    """Resolve a record to an explanation block.

    `generator` is a callable taking the payload and returning text, or None
    when no generator is available. The three-way outcome is decided here rather
    than by each caller, so serving and the pipeline cannot disagree about when
    Pulsemind is allowed to explain something.
    """
    suf = sufficiency(record, policy)
    if not suf.ok:
        return {"status": INSUFFICIENT_DATA, "text": policy.insufficient_text,
                "generator": None, "reasons": list(suf.reasons)}
    if generator is None:
        return {"status": GENERATOR_UNAVAILABLE, "text": policy.unavailable_text,
                "generator": None, "reasons": []}
    return {"status": OK, "text": generator(build_payload(record, policy)),
            "generator": generator_name, "reasons": []}
