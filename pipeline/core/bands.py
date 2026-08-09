"""Risk bands and hysteresis -- pure logic. No I/O, no config reads, no numpy.

Importable by serving code without dragging in the pipeline, exactly as
charlson.py is pure logic used by s02.

WHY THIS IS A STEPPER
---------------------
The product scores a live ventilator telemetry stream, so the machine that ships
consumes one reading at a time and carries state forward. This module is written
that way, and the offline sweep in s16_bands.py loops over the SAME step_latch()
that serving would call. Calibrating a whole-array implementation and shipping a
streaming one would mean the measured flip rate and lost lead time describe a
machine that never runs.

FOUR STATES FROM THREE PERSISTING THRESHOLDS
--------------------------------------------
Each boundary is an independent two-state latch with its own deadband and dwell.
The displayed band is the NUMBER OF LATCHES THAT ARE ON.

That composition is not cosmetic -- it is what makes the cuts separable. A
single four-band machine couples them: demoting out of HIGH would read the gap
down to MEDIUM for its margin, so no cut could be solved without the others, and
the budget search would be three-dimensional. As independent latches, crossings
of boundary k depend on cut k alone and each cut solves against its own alert
budget.

Monotonicity (latch k+1 on => latch k on) holds by construction whenever the
cuts ascend, the dwell settings are shared, and the margin fraction is < 1:
    demote_at(k) = (1-f)*cut(k) + f*cut(k-1)  lies in (cut(k-1), cut(k)]
so demote_at(k) <= cut(k) < demote_at(k+1) and the lower latch always releases
later. It is asserted anyway -- a hand-edited band table is exactly the case the
invariant would not survive.

Boundary convention is `p > cut`, matching operating_point.json's documented
"alert when p_calibrated > threshold" and s13_calibrate.operating_point().
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

# The three values band.state can take. `confirmed` means the badge agrees with
# the current score; the other two are the honest disclosure that it does not.
CONFIRMED = "confirmed"      # displayed == instant
PROVISIONAL = "provisional"  # instant is higher; a promotion is pending
DEMOTING = "demoting"        # instant is lower; the badge is being held up
STATES = (CONFIRMED, PROVISIONAL, DEMOTING)


# ---------------------------------------------------------------------------
# Configuration -- built once, read in the hot loop
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Boundary:
    """One persisting threshold.

    `floor_below` is the next cut down (0.0 for the lowest), used only to size
    the deadband: a margin expressed as a fraction of the band below scales with
    how much room that band actually has, where one absolute margin would be
    generous at the top of the scale and overwhelming at the bottom.
    """
    cut: float
    floor_below: float
    promote_dwell_min: float
    promote_min_readings: int
    demote_margin_frac: float
    demote_dwell_min: float

    @property
    def demote_at(self) -> float:
        """Deadband floor. Falling below `cut` is not enough to drop the badge;
        the score has to clear the margin too."""
        return self.cut - self.demote_margin_frac * (self.cut - self.floor_below)

    def to_json(self) -> dict:
        return {"cut": self.cut, "floor_below": self.floor_below,
                "demote_at": self.demote_at,
                "promote_dwell_min": self.promote_dwell_min,
                "promote_min_readings": self.promote_min_readings,
                "demote_margin_frac": self.demote_margin_frac,
                "demote_dwell_min": self.demote_dwell_min}


def boundaries_from_cuts(cuts, *, promote_dwell_min, promote_min_readings,
                         demote_margin_frac, demote_dwell_min) -> tuple[Boundary, ...]:
    """Build the latch set from ascending cuts and one shared dwell config.

    Shared dwell is what guarantees the monotone-prefix invariant: a higher latch
    can never promote before the one below it, because both start pending on the
    same reading.
    """
    cuts = list(cuts)
    if cuts != sorted(cuts):
        raise ValueError(f"cuts must ascend, got {cuts}")
    if not 0.0 <= demote_margin_frac < 1.0:
        raise ValueError(f"demote_margin_frac must be in [0, 1), got "
                         f"{demote_margin_frac}")
    below = [0.0] + cuts[:-1]
    return tuple(Boundary(c, b, promote_dwell_min, promote_min_readings,
                          demote_margin_frac, demote_dwell_min)
                 for c, b in zip(cuts, below))


@dataclass(frozen=True, slots=True)
class BandMachine:
    """Names ascend in severity; `boundaries[k]` is the floor of `names[k+1]`."""
    names: tuple[str, ...]
    boundaries: tuple[Boundary, ...]
    max_gap_min: float

    def __post_init__(self) -> None:
        if len(self.boundaries) != len(self.names) - 1:
            raise ValueError(f"{len(self.names)} bands need "
                             f"{len(self.names) - 1} boundaries, got "
                             f"{len(self.boundaries)}")
        d = [b.demote_at for b in self.boundaries]
        if d != sorted(d) or len(set(d)) != len(d):
            raise ValueError(f"deadband floors are not strictly ascending: {d} "
                             "-- the latches cannot stay a monotone prefix")

    @property
    def cuts(self) -> list[float]:
        return [b.cut for b in self.boundaries]

    def assign(self, p: float) -> int:
        """Instant band: how many cuts the score is strictly above."""
        return bisect_left(self.cuts, p)

    def to_json(self) -> dict:
        return {"names": list(self.names), "max_gap_min": self.max_gap_min,
                "boundaries": [b.to_json() for b in self.boundaries]}

    @classmethod
    def from_json(cls, d: dict) -> "BandMachine":
        return cls(names=tuple(d["names"]),
                   boundaries=tuple(
                       Boundary(b["cut"], b["floor_below"],
                                b["promote_dwell_min"], b["promote_min_readings"],
                                b["demote_margin_frac"], b["demote_dwell_min"])
                       for b in d["boundaries"]),
                   max_gap_min=d["max_gap_min"])


# ---------------------------------------------------------------------------
# The primitive. Everything else is a loop over this.
# ---------------------------------------------------------------------------
# Latch state is a plain tuple, not a dataclass: this runs once per reading per
# boundary, and the offline sweep calls it tens of millions of times. The
# readable wrapper is BandStepper below.
#
#   (on: bool, pend_since: float | None, pend_readings: int)
LATCH_INIT = (False, None, 0)


def step_latch(b: Boundary, state: tuple, p: float, t: float,
               max_gap_min: float, gap: float | None) -> tuple:
    """Advance one persisting threshold by one reading.

    `t` and `gap` are minutes. A gap longer than `max_gap_min` clears any pending
    transition -- "sustained for 30 minutes" cannot be claimed across a hole in
    the charting. It does NOT drop a latch that is already on: silently dropping
    a patient's badge because nobody charted for four hours would be its own lie.
    """
    on, pend_since, pend_readings = state
    if gap is not None and gap > max_gap_min:
        pend_since, pend_readings = None, 0

    if not on:
        if p > b.cut:
            if pend_since is None:
                pend_since = t
            pend_readings += 1
            if (t - pend_since >= b.promote_dwell_min
                    and pend_readings >= b.promote_min_readings):
                return (True, None, 0)
            return (False, pend_since, pend_readings)
        # any reading at or below the cut breaks continuity
        return (False, None, 0)

    if p < b.demote_at:
        if pend_since is None:
            pend_since = t
        pend_readings += 1
        if t - pend_since >= b.demote_dwell_min:
            return (False, None, 0)
        return (True, pend_since, pend_readings)
    # inside the deadband, or back above the cut: the fall was not sustained
    return (True, None, 0)


@dataclass(slots=True)
class BandView:
    """What one scored reading yields. Mirrors the `band` block of the record
    schema in models/risk_bands_*.json."""
    displayed_index: int
    displayed: str
    instant_index: int
    instant: str
    state: str
    readings_in_state: int


class BandStepper:
    """Serving-facing wrapper: one instance per ventilated stay.

    Mutable on purpose -- a stream handler holds one of these per bed and calls
    push() as telemetry arrives. State is small and serialisable via snapshot()
    / restore() so a handler can be restarted without resetting every badge to
    LOW, which would silently de-escalate the whole unit.
    """

    __slots__ = ("m", "latches", "last_t", "displayed", "readings_in_state")

    def __init__(self, machine: BandMachine) -> None:
        self.m = machine
        self.latches = [LATCH_INIT] * len(machine.boundaries)
        self.last_t: float | None = None
        self.displayed = 0
        self.readings_in_state = 0

    def push(self, p: float, t: float) -> BandView:
        """`p` is the CALIBRATED probability; `t` is minutes on any consistent
        origin (s16 uses minutes since the first reading of the stay)."""
        m = self.m
        gap = None if self.last_t is None else t - self.last_t
        on_count = 0
        prev_on = True
        for i, b in enumerate(m.boundaries):
            s = step_latch(b, self.latches[i], p, t, m.max_gap_min, gap)
            self.latches[i] = s
            if s[0]:
                if not prev_on:
                    raise AssertionError(
                        f"latch {i} is on while a lower one is off -- the band "
                        f"table is not monotone: {[l[0] for l in self.latches]}")
                on_count += 1
            else:
                prev_on = False
        self.last_t = t

        if on_count != self.displayed:
            self.displayed = on_count
            self.readings_in_state = 1
        else:
            self.readings_in_state += 1

        inst = m.assign(p)
        st = (CONFIRMED if inst == on_count
              else PROVISIONAL if inst > on_count else DEMOTING)
        return BandView(on_count, m.names[on_count], inst, m.names[inst], st,
                        self.readings_in_state)

    def snapshot(self) -> dict:
        return {"latches": [list(l) for l in self.latches], "last_t": self.last_t,
                "displayed": self.displayed,
                "readings_in_state": self.readings_in_state}

    def restore(self, d: dict) -> "BandStepper":
        self.latches = [tuple(l) for l in d["latches"]]
        self.last_t = d["last_t"]
        self.displayed = d["displayed"]
        self.readings_in_state = d["readings_in_state"]
        return self


# ---------------------------------------------------------------------------
# Offline helpers -- the sweep in s16 calls these; both loop over step_latch, so
# what is measured offline is what runs online.
# ---------------------------------------------------------------------------
def latch_series(b: Boundary, p, minutes, max_gap_min: float) -> list[bool]:
    """One boundary over one stay's readings, in charttime order.

    Used to solve a single cut against its alert budget without paying for the
    other two latches -- which is legitimate precisely because the latches are
    independent.
    """
    out = []
    st = LATCH_INIT
    last = None
    for pi, ti in zip(p, minutes):
        st = step_latch(b, st, pi, ti, max_gap_min,
                        None if last is None else ti - last)
        last = ti
        out.append(st[0])
    return out


def run(machine: BandMachine, p, minutes) -> list[BandView]:
    """The full machine over one stay's readings, in charttime order."""
    s = BandStepper(machine)
    return [s.push(pi, ti) for pi, ti in zip(p, minutes)]
