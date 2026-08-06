"""Central configuration for the Pulsemind MIMIC-IV pipeline.

Everything path-, code- and threshold-related lives here so no stage hardcodes
a magic number. See the approved plan for the reasoning behind each choice.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
#
# This repo holds code and results. Everything large or credentialed lives in the
# workspace beside it and is never committed:
#
#   <workspace>/data     MIMIC-IV 3.1 -- ~97 GB, PhysioNet DUA, not redistributable
#   <workspace>/build    pipeline artifacts -- ~420 MB, regenerable
#   <workspace>/models   fitted models -- ~152 MB, regenerable
#   <repo>/reports       measured results -- small, tracked, the deliverable
#
# Every root is environment-overridable, so a checkout can keep its data
# somewhere else entirely without editing this file.
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("PM_WORKSPACE", REPO_ROOT.parent))

DATA_ROOT = Path(os.getenv("PM_DATA_ROOT", WORKSPACE / "data"))
MIMIC = DATA_ROOT / "mimic-iv-3.1"


# MIMIC-IV ships each table inside a directory of the same name. chartevents is
# nested one level deeper than the rest -- easy to get wrong, so resolve it once.
def mimic_csv(module: str, table: str) -> Path:
    """Resolve a MIMIC table to its CSV, tolerating the repeated-directory nesting.

    Returns the expected path even when the file is absent, so that importing this
    module never fails on a checkout without the 97 GB source tree -- the reports
    and the code should be readable without it. Stages that actually read MIMIC
    call `require_mimic()` first, which fails with a useful message.
    """
    base = MIMIC / module / f"{table}.csv"
    candidates = [
        base / f"{table}.csv" / f"{table}.csv",  # chartevents is triple-nested
        base / f"{table}.csv",
        base,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[1]  # the usual layout; require_mimic() reports it properly


CHARTEVENTS = mimic_csv("icu", "chartevents")
ICUSTAYS = mimic_csv("icu", "icustays")
PROCEDUREEVENTS = mimic_csv("icu", "procedureevents")
INPUTEVENTS = mimic_csv("icu", "inputevents")
D_ITEMS = mimic_csv("icu", "d_items")
PATIENTS = mimic_csv("hosp", "patients")
ADMISSIONS = mimic_csv("hosp", "admissions")
DIAGNOSES_ICD = mimic_csv("hosp", "diagnoses_icd")
TRANSFERS = mimic_csv("hosp", "transfers")

MIMIC_TABLES = {
    "icu/chartevents": CHARTEVENTS, "icu/icustays": ICUSTAYS,
    "icu/procedureevents": PROCEDUREEVENTS, "icu/inputevents": INPUTEVENTS,
    "icu/d_items": D_ITEMS, "hosp/patients": PATIENTS,
    "hosp/admissions": ADMISSIONS, "hosp/diagnoses_icd": DIAGNOSES_ICD,
    "hosp/transfers": TRANSFERS,
}


def require_mimic() -> None:
    """Fail early and legibly when the MIMIC source tree is not where we expect.

    Called once by run_all before any stage runs, so a misconfigured data root
    surfaces immediately rather than as a DuckDB IO error six minutes in.
    """
    missing = sorted(n for n, p in MIMIC_TABLES.items() if not p.is_file())
    if missing:
        raise FileNotFoundError(
            f"MIMIC-IV source tables not found under {MIMIC}\n"
            f"  missing: {', '.join(missing)}\n"
            "  Set PM_DATA_ROOT to the directory holding mimic-iv-3.1/, or "
            "PM_WORKSPACE to its parent."
        )


# Reference export used to verify the local extract reproduces BigQuery semantics.
BQ_EXPORT = DATA_ROOT / "raw-query.csv"

# Pipeline outputs. Derived and large -- deliberately outside the repo.
BUILD = Path(os.getenv("PM_BUILD_ROOT", WORKSPACE / "build"))
COHORT_PQ = BUILD / "cohort.parquet"
TS_LONG_PQ = BUILD / "ts_long.parquet"          # the cache -- the only 42 GB scan feeds this
T1_STATIC_PQ = BUILD / "t1_static.parquet"
T2_WIDE_PQ = BUILD / "t2_timeseries.parquet"
T2_IMPUTED_PQ = BUILD / "t2_imputed.parquet"
T3_INTERV_PQ = BUILD / "t3_interventions.parquet"
T4_OUTCOMES_PQ = BUILD / "t4_outcomes.parquet"
MODEL_MATRIX_PQ = BUILD / "model_matrix.parquet"
FOLDS_PQ = BUILD / "folds.parquet"
REFERENCE_STATS_JSON = BUILD / "reference_stats.json"
MANIFEST_DIR = BUILD / "manifests"
REPORTS = REPO_ROOT / "reports"                                  # tracked in git
MODELS = Path(os.getenv("PM_MODELS_ROOT", WORKSPACE / "models"))  # derived, not tracked

SCRATCH = BUILD / "scratch"

for _d in (BUILD, MANIFEST_DIR, REPORTS, MODELS, SCRATCH):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Keep every temporary file on D:. The C: drive is tight (~13 GB), and a
# spilling DuckDB query or a CatBoost training directory can be many GB.
# This must run before duckdb/matplotlib/numba/catboost are first used.
# --------------------------------------------------------------------------
for _var in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_var] = str(SCRATCH)
os.environ.setdefault("MPLCONFIGDIR", str(SCRATCH / "mpl"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(SCRATCH / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRATCH / "xdg"))

import tempfile as _tempfile  # noqa: E402

_tempfile.tempdir = str(SCRATCH)

CATBOOST_TRAIN_DIR = SCRATCH / "catboost"  # never let CatBoost write to CWD/C:

# --------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------
DUCKDB_MEMORY_LIMIT = os.getenv("PM_DUCKDB_MEM", "20GB")
DUCKDB_THREADS = int(os.getenv("PM_DUCKDB_THREADS", "24"))
DUCKDB_TEMP_DIR = BUILD / "duckdb_tmp"
DUCKDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
DUCKDB_MAX_TEMP = os.getenv("PM_DUCKDB_MAX_TEMP", "12GB")

# --------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------
ITEM_INVASIVE_VENT = 225792
ITEM_NONINVASIVE_VENT = 225794
# Items used by the "wider" cohort definition (detection only -- still ventilated patients)
COHORT_HINT_ITEMS = {
    220339: "PEEP set",
    223849: "Ventilator Mode",
    224695: "Peak Insp. Pressure",
    224685: "Tidal Volume (observed)",
}
VENT_WINDOW_GRACE_MIN = 0  # rows strictly inside [vent_start, vent_end]

# --------------------------------------------------------------------------
# THE FROZEN ELEVEN -- the model's only time-series inputs. Immutable.
# --------------------------------------------------------------------------
FROZEN_PARAMS: dict[str, int] = {
    "spo2": 220277,
    "fio2": 223835,
    "flow_rate": 224691,
    "peep": 220339,
    "pip": 224695,
    "respiratory_rate_total": 224690,
    "minute_volume": 224687,
    "tidal_volume_observed": 224685,
    "etco2": 228640,
    "inspiratory_ratio": 226873,
    "expiratory_ratio": 226871,
}
# Column order must match data/raw-query.csv exactly for the verification diff.
FROZEN_ORDER = list(FROZEN_PARAMS.keys())

# Some frozen columns are sourced from more than one itemid, taken in preference
# order (first non-null wins).
#
# `peep`: verified against data/raw-query.csv -- on all 14,288 rows where our
# PEEP-set-only value disagreed with the export, the export matched Total PEEP
# Level (224700) exactly, and none matched PEEP set (220339). So the frozen
# BigQuery definition prefers Total PEEP, which is also the better measure
# clinically: it is set PEEP plus any auto-PEEP, i.e. the pressure actually in
# the lung. 224700 is already in the cache, so this costs no rescan.
FROZEN_PARAM_SOURCES: dict[str, list[int]] = {n: [i] for n, i in FROZEN_PARAMS.items()}
FROZEN_PARAM_SOURCES["peep"] = [224700, 220339]

FROZEN_SOURCE_ITEMIDS: list[int] = sorted(
    {i for ids in FROZEN_PARAM_SOURCES.values() for i in ids})

# --------------------------------------------------------------------------
# Cache superset -- STORED, NOT TRAINED ON.
# Extra codes are ~free during a scan that already reads every byte, but adding
# one later would cost another full pass. This does not change the model.
# --------------------------------------------------------------------------
EXTRA_CACHED_ITEMS: dict[str, int] = {
    "o2_flow": 223834,
    "total_peep": 224700,
    "respiratory_rate": 220210,
    "plateau_pressure": 224696,
    "mean_airway_pressure": 224697,
    "compliance": 229661,
    "tidal_volume_set": 224684,
    "spont_tidal_volume": 224421,
    "spont_respiratory_rate": 224422,
    "pf_ratio_value": 229393,
    "heart_rate": 220045,
    "temperature_f": 223761,
    "temperature_c": 223762,
    "arterial_bp_mean": 220052,
    "wbc": 220546,
    "resistance": 220283,
    "ventilator_mode": 223849,
    "ventilator_type": 223848,
    "o2_delivery_device": 226732,
}
CACHE_ITEMS: dict[str, int] = {**FROZEN_PARAMS, **EXTRA_CACHED_ITEMS}
CACHE_ITEMIDS: list[int] = sorted(set(CACHE_ITEMS.values()))
ITEMID_TO_NAME: dict[int, str] = {v: k for k, v in CACHE_ITEMS.items()}

# Items whose payload is text rather than a number (valuenum is null for these).
TEXT_ITEMIDS = {223849, 223848, 226732}

# --------------------------------------------------------------------------
# Physiologic plausibility ranges. Out-of-range -> NULL, so bad values flow into
# the missingness machinery instead of corrupting medians. Absent from the
# original pipeline entirely.
# --------------------------------------------------------------------------
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "spo2": (0.0, 100.0),
    "fio2": (21.0, 100.0),
    "flow_rate": (0.0, 150.0),
    "peep": (0.0, 35.0),
    "pip": (0.0, 80.0),
    "respiratory_rate_total": (0.0, 70.0),
    "minute_volume": (0.0, 40.0),
    "tidal_volume_observed": (0.0, 2000.0),
    "etco2": (0.0, 100.0),
    "inspiratory_ratio": (0.0, 10.0),
    "expiratory_ratio": (0.0, 20.0),
}

# FiO2 is charted both as a fraction (0.21-1.0) and a percentage (21-100).
# Values at or below this bound are treated as fractions and scaled up.
FIO2_FRACTION_MAX = 1.0

# --------------------------------------------------------------------------
# Imputation
# --------------------------------------------------------------------------
LOCF_CUTOFF_MIN = 240.0  # a setting from >4h ago is not "current"
SUFFIXES = ("_observed", "_delta_t_min", "_locf", "_structurally_missing_in_stay", "_final")

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
TARGET = "warning"
GROUP_KEY = "subject_id"   # patient, NOT stay -- one patient can have several stays
N_FOLDS = 5
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Forward-looking targets (stage 14) -- candidate D, composite deterioration
#
# `warning` is a caregiver documentation flag: 84.6% of its achievable AP skill
# (95.2% of ROC-AUC skill) is reachable from charting pattern alone. D asks a
# different question -- "will this patient deteriorate in the next H hours?" --
# built from measured values and orders rather than the act of charting.
#
# Two of the four components are clinician RESPONSES rather than patient states,
# so D carries some reverse-causation exposure. Components are therefore scored
# individually, not just as an OR, so the exposure is measured and not asserted.
# --------------------------------------------------------------------------
FORWARD_TARGETS_PQ = BUILD / "targets_forward.parquet"

HORIZONS_H = (2, 6, 12)      # 6 is trained; 2 and 12 justify the choice
PRIMARY_HORIZON_H = 6

D_FIO2_RISE = 20.0           # percentage points above the setting in force
D_PEEP_RISE = 3.0            # cmH2O above the setting in force
D_SPO2_BELOW = 88.0          # the one component that is a patient state
STRICT_BASELINE_AGE_MIN = 60.0  # `y_strict`: escalation judged only against a fresh setting

# Baselines come from the LOCF value (the setting currently in force, capped at
# LOCF_CUTOFF_MIN) because that is what a clinician actually sees. Forward values
# must be MEASURED, never imputed -- an event has to be observed to count.

# --------------------------------------------------------------------------
# Calibration (stage 13)
#
# AP and ROC-AUC measure RANK and are invariant under any monotone rescaling of
# the score. The product does not consume a rank -- bki/backend/main.py applies
# `risk_prob > 0.70`, which is a test on the LEVEL. Stage 13 measures the level
# and fits the map that makes it mean something.
#
# Two folds are carved out of the training patients. Fold 3 drives early
# stopping; fold 4 fits the calibrator. Neither is ever the test set: fitting a
# calibrator on test is precisely the defect in training_calinerating.ipynb.
# --------------------------------------------------------------------------
EARLY_STOP_FOLD = 3        # chooses the tree count
CALIB_FOLD = 4             # fits the calibrator -- unseen by the model
TRAIN_FOLDS = (0, 1, 2)

SERVING_THRESHOLD = 0.70   # the constant in bki/backend/main.py:85

# Alerts per ventilated patient-day. A bedside monitor competes for attention,
# so the threshold is chosen against an alert budget rather than picked round.
ALERT_BUDGETS = (1.0, 2.0, 4.0)
ALERT_DEDUP_MINUTES = 60   # rows are irregularly spaced; count alert-hours, not rows

MODEL_XGB = MODELS / "xgboost_warning.ubj"
MODEL_LGB = MODELS / "lightgbm_warning.txt"
CALIBRATOR_PKL = MODELS / "calibrator.joblib"
OPERATING_POINT_JSON = MODELS / "operating_point.json"

# GPU: RTX 5060 (Blackwell sm_120), 8 GB VRAM with ~6.5 GB usable.
# Keep max_bin low so the quantised matrix stays comfortably inside VRAM.
GPU_MAX_BIN = 64
USE_GPU = os.getenv("PM_USE_GPU", "1") == "1"
N_CPU_THREADS = int(os.getenv("PM_THREADS", "30"))
