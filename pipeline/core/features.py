"""Build the 109-column feature row for one reading, at serving time.

s07/s08/s02 assemble these once over the whole cohort with window functions; a
service holds one bed, so the definitions are restated here as a per-stay state
machine. `tools.push_parity` diffs all 109 columns against the model matrix.

TWELVE CANNOT BE SERVED AS TRAINED, and take their causal reading instead:

  `{p}_structurally_missing_in_stay` (11)  `observed.sum().over("stay_id") == 0`
      is an aggregate over the whole stay, future included.
  `vent_hours`  `date_diff('hour', vent_start, vent_end)` is known only once
      ventilation has ended.

They sit inside the 33 documentation features carrying 18.1% of this model's
skill, so the drift is measured rather than assumed small.

Imports no stage; `charlson.CHARLSON` is reused rather than copied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .charlson import CHARLSON, GRADED_PAIRS

# s02_table1_static.py:30. Here because PBW is defined in inches.
IN_TO_CM = 2.54

# s08_table3_interventions.py:35-43. The five groups, in the order the feature
# names appear. `inotrope` is separate from `vasopressor` on purpose --
# dobutamine raises cardiac output rather than vascular resistance.
DRUG_GROUPS = ("vasopressor", "sedative", "opioid", "paralytic", "inotrope")

# What s08 writes when no interval matches. Not zero: zero minutes since start
# would mean an infusion beginning exactly now.
NO_INFUSION = -1.0


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ServingAssets:
    """Everything models/ knows that the booster file does not.

    Written by `tools.export_serving_assets`. XGBoost splits on the integer code
    behind "MICU", not the string, so a level list built from a different row set
    is a different model that still returns a probability.
    """

    feature_order: tuple[str, ...]
    categorical: dict[str, tuple[str, ...]]
    reference_medians: dict[str, float]
    frozen_params: tuple[str, ...]
    locf_cutoff_min: float
    contrib_top_k: int
    charlson_hierarchy: bool
    documentation_features: frozenset
    feature_group: dict[str, str]
    provenance: dict

    @classmethod
    def load(cls, path) -> "ServingAssets":
        d = json.loads(path.read_text())
        return cls(
            feature_order=tuple(d["feature_order"]),
            categorical={k: tuple(v) for k, v in d["categorical"].items()},
            reference_medians=dict(d["reference_medians"]),
            frozen_params=tuple(d["frozen_params"]),
            locf_cutoff_min=float(d["locf_cutoff_min"]),
            contrib_top_k=int(d["contrib_top_k"]),
            charlson_hierarchy=bool(d["charlson_hierarchy"]),
            documentation_features=frozenset(d["documentation_features"]),
            feature_group=dict(d["feature_group"]),
            provenance=dict(d["provenance"]),
        )

    def kind_of(self, feature: str) -> str:
        """'documentation' when the feature describes charting, not the patient.

        A generated explanation must never narrate a documentation contributor as
        a change in the patient.
        """
        return "documentation" if feature in self.documentation_features else "physiology"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Infusion:
    """One infusion interval, as the drug chart records it.

    s08 tests `COALESCE(endtime, starttime) > charttime`, so `ended_at=None` can
    never be running: an open-ended infusion counts as stopped, not forever.
    """

    group: str
    started_at: datetime
    ended_at: datetime | None
    rate: float = 0.0

    def running_at(self, now: datetime) -> bool:
        end = self.ended_at if self.ended_at is not None else self.started_at
        return self.started_at < now < end


@dataclass(frozen=True, slots=True)
class PatientContext:
    """The static half of the row: HIS/EMR, ADT and the coded diagnosis list.

    `pbw_kg`, the seventeen `cci_*` flags and the three `charlson_*` columns are
    derived here, so a caller supplies only what a hospital system holds.
    """

    sex: str
    age_at_icu: float
    admission_type: str
    admission_location: str
    race: str
    first_careunit: str
    ventilation_start: datetime
    height_cm: float | None = None
    weight_kg: float | None = None
    insurance: str | None = None
    language: str | None = None
    marital_status: str | None = None
    # None, not 0.0: NULL on 44.56% of admissions, and the booster learned a
    # direction for that NaN. Zero is an in-range observed value.
    hours_admit_to_icu: float | None = None
    ed_minutes: float | None = None
    prior_icu_stays: int = 0
    # (code, icd_version). Codes are stored without dots, as MIMIC stores them.
    icd_codes: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Reading:
    """One tick from the bedside.

    `values` carries only what this tick measured. Absent is not zero and not
    null: it is unobserved, and the previous value is carried forward.
    """

    observed_at: datetime
    values: dict[str, float] = field(default_factory=dict)
    infusions: tuple[Infusion, ...] = ()
    ventilator_mode: str | None = None


# ---------------------------------------------------------------------------
# Static derivations
# ---------------------------------------------------------------------------
def predicted_body_weight(height_cm: float | None, sex: str | None) -> float | None:
    """ARDSNet PBW, from height and sex only (s02_table1_static.py:140).

    Never actual weight -- it moves with fluid balance, which would make the
    tidal-volume denominator a treatment effect.
    """
    if height_cm is None or sex is None:
        return None
    inches = height_cm / IN_TO_CM
    base = 50.0 if sex == "M" else 45.5
    return base + 2.3 * (inches - 60.0)


def charlson_flags(icd_codes) -> dict[str, int]:
    """The seventeen `cci_*` flags, from (code, icd_version) pairs.

    The Python form of `charlson.category_sql()`.
    """
    flags = {name: 0 for name in CHARLSON}
    for code, version in icd_codes:
        code = str(code).upper().replace(".", "")
        for name, (icd9, icd10, _weight) in CHARLSON.items():
            prefixes = icd9 if int(version) == 9 else icd10
            if any(code.startswith(p) for p in prefixes):
                flags[name] = 1
    return flags


def charlson_age_points(age: float | None) -> int:
    """Standard age adjustment, INCLUSIVE upper bounds (charlson.age_points_sql).

    The exclusive form overcharged every patient aged exactly 50/60/70/80 by a
    point, and scored unknown age 4 instead of 0.
    """
    if age is None:
        return 0
    for limit, points in ((50, 0), (60, 1), (70, 2), (80, 3)):
        if age <= limit:
            return points
    return 4


def charlson_comorbidity_score(flags: dict[str, int], hierarchy: bool = True) -> int:
    """Weighted sum with the graded-pair hierarchy (charlson.score_sql).

    Summing both halves of a graded pair explained 91.94% of the disagreement
    against MIMIC's own derived table.
    """
    severe_of = {mild: severe for mild, severe in GRADED_PAIRS} if hierarchy else {}
    total = 0
    for name, (_i9, _i10, weight) in CHARLSON.items():
        present = flags.get(name, 0)
        if name in severe_of and flags.get(severe_of[name], 0) == 1:
            present = 0
        total += weight * present
    return total


def static_row(context: PatientContext, hierarchy: bool = True) -> dict[str, object]:
    """The 36 Table-1 columns for one admission, minus `vent_hours`.

    `vent_hours` is the one static column that is not static -- `StayFeatures`
    supplies its causal form per reading.
    """
    flags = charlson_flags(context.icd_codes)
    score = charlson_comorbidity_score(flags, hierarchy)
    age_points = charlson_age_points(context.age_at_icu)

    row: dict[str, object] = {
        "gender": context.sex,
        "age_at_icu": context.age_at_icu,
        "admission_type": context.admission_type,
        "admission_location": context.admission_location,
        "insurance": context.insurance,
        "language": context.language,
        "marital_status": context.marital_status,
        "race": context.race,
        "first_careunit": context.first_careunit,
        "hours_admit_to_icu": context.hours_admit_to_icu,
        "ed_minutes": context.ed_minutes,
        "prior_icu_stays": context.prior_icu_stays,
        "height_cm": context.height_cm,
        "weight_kg": context.weight_kg,
        "pbw_kg": predicted_body_weight(context.height_cm, context.sex),
        "charlson_comorbidity_score": score,
        "charlson_age_points": age_points,
        "charlson_index": score + age_points,
    }
    row.update({f"cci_{name}": value for name, value in flags.items()})
    return row


# ---------------------------------------------------------------------------
# The per-stay state machine
# ---------------------------------------------------------------------------
class StayFeatures:
    """One instance per ventilated stay. Feed it readings in time order.

    One per bed alongside one `bands.BandStepper`. `snapshot()`/`restore()`
    matter: a handler restarted without its state loses every carried-forward
    value at once and every parameter reads as never measured.
    """

    def __init__(self, context: PatientContext, assets: ServingAssets) -> None:
        self.context = context
        self.assets = assets
        self.static = static_row(context, assets.charlson_hierarchy)
        # Last value seen per parameter, and when. Absent = never observed.
        self._last_value: dict[str, float] = {}
        self._last_seen: dict[str, datetime] = {}
        self._last_mode: str | None = None
        self._last_mode_at: datetime | None = None
        self._last_reading_at: datetime | None = None
        # Not yet visible to a row, oldest first. A mode carries its OWN
        # charttime in s08's ASOF join and need not coincide with a reading.
        self._pending_modes: list[tuple[datetime, str]] = []

    # -- state ------------------------------------------------------------
    def snapshot(self) -> dict:
        """Serialisable state. Timestamps as ISO strings so this is JSON."""
        return {
            "last_value": dict(self._last_value),
            "last_seen": {p: t.isoformat() for p, t in self._last_seen.items()},
            "last_mode": self._last_mode,
            "last_mode_at": (self._last_mode_at.isoformat()
                             if self._last_mode_at else None),
            "last_reading_at": (self._last_reading_at.isoformat()
                                if self._last_reading_at else None),
            "pending_modes": [[t.isoformat(), m] for t, m in self._pending_modes],
        }

    def restore(self, state: dict) -> "StayFeatures":
        self._last_value = dict(state.get("last_value", {}))
        self._last_seen = {p: datetime.fromisoformat(t)
                           for p, t in state.get("last_seen", {}).items()}
        self._last_mode = state.get("last_mode")
        at = state.get("last_mode_at")
        self._last_mode_at = datetime.fromisoformat(at) if at else None
        seen = state.get("last_reading_at")
        self._last_reading_at = datetime.fromisoformat(seen) if seen else None
        self._pending_modes = [(datetime.fromisoformat(t), m)
                               for t, m in state.get("pending_modes", [])]
        return self

    def observe_mode(self, mode: str, at: datetime) -> None:
        """A ventilator mode charted between readings, with its own timestamp.

        Queued rather than applied: s08's ASOF join is strictly-before, so the
        row at the mode's own instant must not see it.
        """
        latest = (self._pending_modes[-1][0] if self._pending_modes
                  else self._last_mode_at)
        if latest is not None and at < latest:
            raise ValueError(
                f"mode at {at.isoformat()} precedes the last one at "
                f"{latest.isoformat()} -- out of order would age it from the future")
        self._pending_modes.append((at, mode))

    # -- the five companion columns ---------------------------------------
    def _timeseries_row(self, reading: Reading) -> dict[str, object]:
        """s07_impute.build_expressions, one row at a time.

        Order matters: the reading updates `last_seen` BEFORE the age is taken,
        so a parameter measured now has age 0, not the gap since the last one.
        """
        row: dict[str, object] = {}
        for param in self.assets.frozen_params:
            value = reading.values.get(param)
            observed = value is not None
            if observed:
                self._last_value[param] = float(value)
                self._last_seen[param] = reading.observed_at

            seen_at = self._last_seen.get(param)
            if seen_at is None:
                age = None
                locf = None
            else:
                age = (reading.observed_at - seen_at).total_seconds() / 60.0
                # A setting from more than four hours ago is not "current".
                locf = self._last_value[param] if age <= self.assets.locf_cutoff_min else None

            row[f"{param}_observed"] = 1 if observed else 0
            row[f"{param}_delta_t_min"] = age
            row[f"{param}_locf"] = locf
            # CAUSAL FORM. As trained this asks whether the parameter was ever
            # measured anywhere in the stay, including after now.
            row[f"{param}_structurally_missing_in_stay"] = 0 if seen_at is not None else 1
            row[f"{param}_final"] = (locf if locf is not None
                                     else self.assets.reference_medians[param])
        return row

    # -- interventions -----------------------------------------------------
    def _intervention_row(self, reading: Reading) -> dict[str, object]:
        """s08_table3_interventions, strictly-before intervals only.

        Treatments are given *because* a patient is deteriorating, so an infusion
        starting at the reading's instant leaks the label.
        """
        now = reading.observed_at
        row: dict[str, object] = {}
        for group in DRUG_GROUPS:
            active = [i for i in reading.infusions
                      if i.group == group and i.running_at(now)]
            row[f"{group}_running"] = 1 if active else 0
            row[f"{group}_rate"] = float(sum(i.rate for i in active))
            row[f"{group}_minutes_since_start"] = (
                max((now - i.started_at).total_seconds() / 60.0 for i in active)
                if active else NO_INFUSION)

        # STRICTLY BEFORE, unlike the eleven parameters: s08 joins with
        # `ASOF LEFT JOIN ... AND d.charttime > m.charttime`, and
        # `ventilator_mode_age_min == 0` occurs ZERO times in 4.2M training rows
        # (minimum 1.0). Draining on the strict inequality reproduces that and
        # keeps each event's own charttime.
        if reading.ventilator_mode is not None:
            self.observe_mode(reading.ventilator_mode, now)
        while self._pending_modes and self._pending_modes[0][0] < now:
            self._last_mode_at, self._last_mode = self._pending_modes.pop(0)

        row["ventilator_mode"] = self._last_mode
        row["ventilator_mode_age_min"] = (
            (now - self._last_mode_at).total_seconds() / 60.0
            if self._last_mode_at is not None else NO_INFUSION)
        return row

    # -- assembly ----------------------------------------------------------
    def push(self, reading: Reading) -> pd.DataFrame:
        """Advance the stay by one reading and return its 109-column frame.

        Nine columns are `category` with the FULL training level set attached.
        Building one from the row's own value gives it a single level coded 0,
        which scores cleanly and means something else entirely.
        """
        if self._last_reading_at is not None and reading.observed_at < self._last_reading_at:
            raise ValueError(
                f"reading at {reading.observed_at.isoformat()} precedes the last one at "
                f"{self._last_reading_at.isoformat()} -- out of order would give a negative "
                f"age and carry a value back from the future")
        if reading.observed_at < self.context.ventilation_start:
            raise ValueError(
                f"reading at {reading.observed_at.isoformat()} precedes ventilation start "
                f"{self.context.ventilation_start.isoformat()}")

        row = dict(self.static)
        row.update(self._timeseries_row(reading))
        row.update(self._intervention_row(reading))

        # CAUSAL FORM. As trained: vent_end - vent_start, a completed episode.
        elapsed = reading.observed_at - self.context.ventilation_start
        row["vent_hours"] = elapsed.total_seconds() // 3600.0

        pbw = row.get("pbw_kg")
        tidal = row.get("tidal_volume_observed_final")
        row["tidal_volume_ml_per_kg_pbw"] = (
            tidal / pbw if pbw and pbw > 0 and tidal is not None else None)

        missing = set(self.assets.feature_order) - set(row)
        extra = set(row) - set(self.assets.feature_order)
        if missing or extra:
            raise ValueError(
                f"feature row does not match the model's columns -- "
                f"missing {sorted(missing)}, unexpected {sorted(extra)}")

        frame = pd.DataFrame([row], columns=list(self.assets.feature_order))
        for column, levels in self.assets.categorical.items():
            supplied = row[column]
            frame[column] = pd.Categorical(frame[column], categories=list(levels))
            # An unseen value becomes code -1 -- indistinguishable from missing,
            # scored without complaint, and a different answer.
            if supplied is not None and frame[column].isna().iloc[0]:
                raise ValueError(
                    f"{column}={supplied!r} is not one of the {len(levels)} levels "
                    f"the model was trained on")
        for column in frame.columns:
            if column not in self.assets.categorical:
                frame[column] = frame[column].astype(np.float32)

        self._last_reading_at = reading.observed_at
        return frame
