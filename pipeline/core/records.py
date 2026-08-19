"""Turn one scored row into the record shape s16 declares and s18 consumes.

The scalar restatement of s17_records; `tools.serving_parity` diffs every number
here against what s17 emitted.

The three shares share ONE denominator, the sum of |contribution| over all 109
features:

  documentation_share  charting behaviour rather than physiology
  imputed_share        cohort defaults this patient never had
  attribution_total    the denominator itself, so a consumer holding only the
                       top-8 can express a share of the whole decision

Their feature sets are disjoint, so they compose rather than double-count, and
`documentation_share + imputed_share <= 1` is checked here as s17 checks it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import ServingAssets

# The logit at which exp() overflows a float. s17 clips here too, because
# contributions can put the logit well past the range sigmoid needs.
_LOGIT_LIMIT = 709.0


@dataclass(frozen=True, slots=True)
class Telemetry:
    """One parameter as the record reports it: value, age, and where it came from."""

    parameter: str
    value: float | None
    age_min: float | None
    measured: bool
    source: str          # measured | carried_forward | population_reference


def telemetry_from_frame(frame, assets: ServingAssets) -> list[Telemetry]:
    """Read the eleven parameters back out of the assembled feature row.

    s07_impute.py:86 makes `_final = _locf.fill_null(median)`, so a null `_locf`
    means the number the model split on is a population statistic -- the third
    state `measured` alone cannot express.
    """
    row = frame.iloc[0]
    out = []
    for param in assets.frozen_params:
        locf = row[f"{param}_locf"]
        measured = bool(row[f"{param}_observed"] > 0.5)
        never_seen = _isnan(locf)
        age = row[f"{param}_delta_t_min"]
        out.append(Telemetry(
            parameter=param,
            value=_clean(row[f"{param}_final"]),
            # The raw age, matching s17_records.py:314. NOT always None for a
            # cohort default: past the 240-minute LOCF cutoff `_locf` is null, so
            # the source reads `population_reference` while the age is real. The
            # dictionary says null there; this follows the emitter.
            age_min=_clean(age),
            measured=measured,
            source=("measured" if measured
                    else "population_reference" if never_seen
                    else "carried_forward"),
        ))
    return out


def attribution(frame, contributions: np.ndarray, bias: float,
                assets: ServingAssets) -> dict:
    """Contributors, the signed tail, and the three shares, for one row.

    `contributions` must already be in calibrated-logit space, so
    sigmoid(sum + tail + bias) is the score this same record reports.
    """
    columns = list(assets.feature_order)
    magnitude = np.abs(contributions)
    total = float(magnitude.sum())

    order = np.argsort(-magnitude)[:assets.contrib_top_k]
    contributors = [{
        "feature": columns[j],
        "value": _clean(frame.iloc[0][columns[j]]),
        "contribution": round(float(contributions[j]), 8),
        "kind": assets.kind_of(columns[j]),
        "group": assets.feature_group.get(columns[j], "unknown"),
    } for j in order]

    # SIGNED, not a magnitude. Offsetting terms in the tail cancel, which is why
    # a coverage ratio must never be taken against it -- that flatters the top-k.
    tail = float(contributions.sum() - contributions[order].sum())

    documentation = sum(magnitude[j] for j, c in enumerate(columns)
                        if c in assets.documentation_features)

    # A value feature rests on a default exactly when `_locf` is null.
    imputed = 0.0
    weighted_age, age_weight = 0.0, 0.0
    row = frame.iloc[0]
    for param in assets.frozen_params:
        value_features = [f"{param}_final", f"{param}_locf"]
        if param == "tidal_volume_observed":
            # Derived from tidal_volume_observed_final, so it inherits that
            # parameter's imputation rather than being independent of it.
            value_features.append("tidal_volume_ml_per_kg_pbw")
        weight = sum(magnitude[columns.index(c)]
                     for c in value_features if c in assets.feature_order)
        if _isnan(row[f"{param}_locf"]):
            imputed += weight

        # Independent of the imputation branch: past the LOCF cutoff a parameter
        # counts as imputed, yet its `_delta_t_min` is real and is the point.
        age = row[f"{param}_delta_t_min"]
        if not _isnan(age):
            weighted_age += weight * float(age)
            age_weight += weight

    doc_share = documentation / total if total > 0 else 0.0
    imp_share = imputed / total if total > 0 else 0.0
    if doc_share + imp_share > 1.0 + 1e-9:
        raise ValueError(
            "documentation and imputed share overlap -- they are defined over "
            "disjoint feature sets and must not double-count")

    return {
        "contributors": contributors,
        "contributors_other": round(tail, 8),
        "contributors_bias": round(float(bias), 8),
        "documentation_share": round(doc_share, 4),
        "imputed_share": round(imp_share, 4),
        # Attribution-WEIGHTED age, not the oldest value in use: a row exists
        # because ONE parameter was charted, so something is nearly always stale.
        "attribution_age_min": (round(weighted_age / age_weight, 4)
                                if age_weight > 0 else None),
        "attribution_total": round(total, 8),
    }


def reconstructs(record: dict, calibrated: float, tolerance: float = 1e-5) -> bool:
    """Does the record add up to the score it reports? Checkable without the model."""
    logit = (sum(c["contribution"] for c in record["contributors"])
             + record["contributors_other"] + record["contributors_bias"])
    # Clipped, as s17 clips: this must return False, not raise, on an extreme.
    clipped = max(-_LOGIT_LIMIT, min(_LOGIT_LIMIT, logit))
    return abs(1.0 / (1.0 + math.exp(-clipped)) - calibrated) < tolerance


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or (
        hasattr(x, "__float__") and math.isnan(float(x)))


def _clean(x):
    """JSON has no NaN, and a bare NaN token becomes a number in a prompt."""
    if x is None:
        return None
    if isinstance(x, str):
        return x
    value = float(x)
    return None if math.isnan(value) else round(value, 4)
