"""Loading the persisted scorer -- one place, because the DEVICE is part of it.

s13 fits and persists; s16 and s17 both re-score from what it wrote. They must
do it identically, and the reason is not tidiness:

  XGBoost's CPU and CUDA predictors disagree on this model by 4.85e-03 on the
  raw score on average and 1.25e-01 at worst, across 417k of 765k test rows.
  The persisted artifacts reproduce reports/calibration.json to EXACTLY zero on
  CUDA and not on CPU. A bare `xgb.Booster()` loaded from file predicts on CPU.

So a band table, an operating point, or a set of attributions is only valid for
the device it was produced on. Putting `set_param({"device": ...})` in one
function is what stops that becoming a silent, untraceable shift downstream.

ATTRIBUTIONS ARE RESCALED INTO CALIBRATED-LOGIT SPACE
-----------------------------------------------------
`pred_contribs` attributes the RAW margin m, with sum(c_i) + bias = m. Platt is
p_cal = sigmoid(a*m + b), which is affine in m, so

    a*m + b = sum(a*c_i) + (a*bias + b)

decomposes the CALIBRATED logit exactly -- not approximately, because an affine
map distributes over a sum. Skipping the rescale would hand downstream an
explanation of a pre-calibration number nobody sees, and anyone adding the
contributions up would get a different answer from the score in the record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config as C
from .xgb_math import _logit, xgb_predict

CONTRIB_CHUNK = 200_000     # (rows x 110 float32) per pass; keeps VRAM bounded


@dataclass(frozen=True, slots=True)
class Scorer:
    """The persisted model + calibrator, pinned to the device they were fitted on."""
    booster: object
    platt: object                  # the fitted sklearn LogisticRegression
    slope: float                   # Platt `a`
    intercept: float               # Platt `b`
    feature_order: list[str]
    device: str

    # -- scores ------------------------------------------------------------
    def raw(self, X) -> np.ndarray:
        return xgb_predict(self.booster, X)

    def calibrate(self, p_raw) -> np.ndarray:
        return self.platt.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]

    def score(self, X) -> np.ndarray:
        return self.calibrate(self.raw(X))

    # -- attributions ------------------------------------------------------
    def contributions(self, X, chunk: int = CONTRIB_CHUNK):
        """Exact per-row TreeSHAP, rescaled into calibrated-logit space.

        Native `pred_contribs` rather than the `shap` package: it is the same
        TreeSHAP, costs one extra predict pass, runs on the same device as the
        prediction, and adds no dependency.

        `iteration_range` must match `xgb_predict`'s or the contributions
        decompose a different number of trees than the score does.

        Returns (contribs, bias) where
            sigmoid(contribs.sum(axis=1) + bias) == the calibrated probability.
        """
        import xgboost as xgb
        rng = (0, self.booster.best_iteration + 1)
        out = []
        for i in range(0, len(X), chunk):
            d = xgb.DMatrix(X.iloc[i:i + chunk], enable_categorical=True)
            out.append(self.booster.predict(d, pred_contribs=True,
                                            iteration_range=rng))
        c = np.concatenate(out).astype(np.float64)
        # Last column is the bias term, not a feature.
        return c[:, :-1] * self.slope, c[:, -1] * self.slope + self.intercept


def load_scorer(expect_columns=None) -> Scorer:
    """Load model + calibrator from models/, pinned to the fitting device.

    `expect_columns` is the loaded matrix's column order. The calibrator records
    the order it was fitted against; scoring a re-ordered matrix silently
    produces a different model, so the check is not optional when a caller has
    the columns to hand.
    """
    import joblib
    import xgboost as xgb

    bst = xgb.Booster()
    bst.load_model(str(C.MODEL_XGB))
    device = "cuda" if C.USE_GPU else "cpu"
    bst.set_param({"device": device})

    payload = joblib.load(C.CALIBRATOR_PKL)
    assert payload["kind"] == "platt", f"unexpected calibrator {payload['kind']}"
    order = list(payload["feature_order"])
    if expect_columns is not None:
        assert order == list(expect_columns), (
            "calibrator feature_order does not match the loaded matrix -- "
            "models/ and build/features.json are out of step")
    lr = payload["model"]
    return Scorer(booster=bst, platt=lr,
                  slope=float(lr.coef_[0][0]), intercept=float(lr.intercept_[0]),
                  feature_order=order, device=device)
