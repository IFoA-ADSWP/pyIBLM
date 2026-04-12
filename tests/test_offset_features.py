"""Tests for offset handling in IBLM."""

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, ExplainIBLM, load_freMTPLmini, split_into_train_validate_test
from iblm._model import _build_design_matrix

SEED = 42


# ---------------------------------------------------------------------------
# Property: beta_coeff reconstruction with offset matches predict()
# ---------------------------------------------------------------------------


def test_beta_coeff_equals_predict_with_offset(model_poisson, splits):
    """beta_coeff reconstruction must match predict() at unit exposure (offset=0).

    data_beta_coeff explains the rate-level prediction (exposure=1). Comparing
    against predict() with offset zeroed out makes the property exact.
    """
    test_data = splits["test"]
    ex = ExplainIBLM(model_poisson, test_data)

    cat_vars = model_poisson.predictor_vars["categorical"]

    coeff_mult = pd.DataFrame({"bias": 1.0}, index=test_data.index)
    for col in model_poisson.predictor_vars["all"]:
        coeff_mult[col] = 1.0 if col in cat_vars else test_data[col].to_numpy(dtype=float)

    bc = ex.data_beta_coeff[list(coeff_mult.columns)]
    predict_from_coeff = np.exp((bc * coeff_mult).sum(axis=1).to_numpy())

    # predict() at unit exposure (offset = 0) to match beta_coeff decomposition
    offset_var = model_poisson.offset_var
    test_unit = test_data.copy()
    if offset_var:
        test_unit[offset_var] = 0.0  # log(1) = 0 → Exposure = 1
    predict_direct = model_poisson.predict(test_unit)

    max_diff = np.max(np.abs(predict_from_coeff / predict_direct - 1))
    assert max_diff < 1e-5, f"Max relative diff (with offset model): {max_diff}"


# ---------------------------------------------------------------------------
# Property: offset_var present vs absent in predict() newdata
# ---------------------------------------------------------------------------


def test_predict_zero_offset_equals_no_offset_col():
    """predict() with offset col = 0 must equal predict() when offset col absent."""
    df = load_freMTPLmini()
    df["LogExposure"] = np.log(df["Exposure"])
    splits = split_into_train_validate_test(df.drop(columns=["Exposure"]), seed=SEED)

    m = IBLM()
    m.fit(
        splits,
        response_var="ClaimNb",
        offset_var="LogExposure",
        family="poisson",
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
    )

    test_data = splits["test"]
    # Version A: no offset column → offset defaults to 0
    test_no_offset = test_data.drop(columns=["LogExposure"])
    preds_no_col = m.predict(test_no_offset)

    # Version B: offset column present but all zeros → same as no offset
    test_zero_offset = test_no_offset.copy()
    test_zero_offset["LogExposure"] = 0.0
    preds_zero = m.predict(test_zero_offset)

    max_diff = np.max(np.abs(preds_no_col / preds_zero - 1))
    assert max_diff < 1e-8, f"Zero-offset vs absent-offset differ: {max_diff}"


# ---------------------------------------------------------------------------
# Property: doubling offset (log-scale) doubles response-scale prediction
# ---------------------------------------------------------------------------


def test_doubling_offset_doubles_prediction():
    """For a log-link model, adding log(2) to offset must double predictions."""
    df = load_freMTPLmini()
    df["LogExposure"] = np.log(df["Exposure"])
    splits = split_into_train_validate_test(df.drop(columns=["Exposure"]), seed=SEED)

    m = IBLM()
    m.fit(
        splits,
        response_var="ClaimNb",
        offset_var="LogExposure",
        family="poisson",
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
    )

    test_data = splits["test"]
    preds_base = m.predict(test_data)

    # Increase LogExposure by log(2) → predictions should double
    test_doubled = test_data.copy()
    test_doubled["LogExposure"] = test_data["LogExposure"] + np.log(2)
    preds_doubled = m.predict(test_doubled)

    ratio = preds_doubled / preds_base
    max_diff = np.max(np.abs(ratio - 2.0))
    assert max_diff < 1e-6, f"Doubling offset should double predictions; max diff: {max_diff}"
