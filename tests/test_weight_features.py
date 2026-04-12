"""Tests for weight handling in IBLM."""

import warnings

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, ExplainIBLM, load_freMTPLmini, split_into_train_validate_test

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_weighted_splits(seed=SEED):
    """Return splits with ClaimRate as response and Exposure as weight."""
    df = load_freMTPLmini()
    df = df.assign(ClaimRate=df["ClaimNb"] / df["Exposure"]).drop(
        columns=["ClaimNb", "LogExposure"], errors="ignore"
    )
    return split_into_train_validate_test(df, seed=seed)


def _fit(splits, response_var, family, weight_var=None, **kwargs):
    m = IBLM()
    m.fit(
        splits,
        response_var=response_var,
        family=family,
        weight_var=weight_var,
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
        **kwargs,
    )
    return m


# ---------------------------------------------------------------------------
# Smoke: weighted fit completes for each log-link family
# ---------------------------------------------------------------------------


def test_weighted_poisson_smoke():
    """Poisson model with Exposure weight fits and predicts without error."""
    splits = _make_weighted_splits()
    model = _fit(splits, "ClaimRate", "poisson", weight_var="Exposure")
    preds = model.predict(splits["test"])
    assert len(preds) == len(splits["test"])
    assert np.all(preds > 0)


def test_weighted_gamma_smoke():
    """Gamma model with Exposure weight fits and predicts without error."""
    df = load_freMTPLmini()
    df = df.assign(ClaimRate=(df["ClaimNb"] / df["Exposure"]).clip(lower=0.01)).drop(
        columns=["ClaimNb", "LogExposure"], errors="ignore"
    )
    splits = split_into_train_validate_test(df, seed=SEED)
    model = _fit(splits, "ClaimRate", "gamma", weight_var="Exposure")
    preds = model.predict(splits["test"])
    assert len(preds) == len(splits["test"])
    assert np.all(preds > 0)


def test_weighted_gaussian_smoke():
    """Gaussian model with Exposure weight fits and predicts without error."""
    splits = _make_weighted_splits()
    model = _fit(splits, "ClaimRate", "gaussian", weight_var="Exposure")
    preds = model.predict(splits["test"])
    assert len(preds) == len(splits["test"])


# ---------------------------------------------------------------------------
# Smoke: ExplainIBLM works with weighted model
# ---------------------------------------------------------------------------


def test_explain_weighted_poisson_smoke():
    """ExplainIBLM runs without error on a weighted Poisson model."""
    splits = _make_weighted_splits()
    model = _fit(splits, "ClaimRate", "poisson", weight_var="Exposure")
    ex = ExplainIBLM(model, splits["test"])
    ex.overall_correction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex.bias_density()


# ---------------------------------------------------------------------------
# Property: weighted model predictions are positive and finite
# ---------------------------------------------------------------------------


def test_weighted_predictions_finite_positive():
    """Weighted model predictions must be finite and positive for log-link families."""
    splits = _make_weighted_splits()
    for family in ("poisson", "gamma"):
        if family == "gamma":
            df = load_freMTPLmini()
            df = df.assign(ClaimRate=(df["ClaimNb"] / df["Exposure"]).clip(lower=0.01)).drop(
                columns=["ClaimNb", "LogExposure"], errors="ignore"
            )
            s = split_into_train_validate_test(df, seed=SEED)
        else:
            s = splits
        model = _fit(s, "ClaimRate", family, weight_var="Exposure")
        preds = model.predict(s["test"])
        assert np.all(np.isfinite(preds)), f"{family}: predictions contain non-finite values"
        assert np.all(preds > 0), f"{family}: predictions contain non-positive values"


# ---------------------------------------------------------------------------
# Property: weight_var column is preserved in test data after predict()
# ---------------------------------------------------------------------------


def test_weight_col_not_modified_by_predict():
    """predict() must not modify or drop the weight column in newdata."""
    splits = _make_weighted_splits()
    model = _fit(splits, "ClaimRate", "poisson", weight_var="Exposure")
    test_data = splits["test"].copy()
    original_exposure = test_data["Exposure"].copy()
    _ = model.predict(test_data)
    pd.testing.assert_series_equal(test_data["Exposure"], original_exposure)


# ---------------------------------------------------------------------------
# Property: beta_coeff reconstruction matches predict() for weighted model
# ---------------------------------------------------------------------------


def test_beta_coeff_equals_predict_weighted():
    """beta_coeff reconstruction must equal predict() for a weighted Poisson model."""
    splits = _make_weighted_splits()
    model = _fit(splits, "ClaimRate", "poisson", weight_var="Exposure")
    test_data = splits["test"]
    ex = ExplainIBLM(model, test_data)

    cat_vars = model.predictor_vars["categorical"]

    coeff_mult = pd.DataFrame({"bias": 1.0}, index=test_data.index)
    for col in model.predictor_vars["all"]:
        coeff_mult[col] = 1.0 if col in cat_vars else test_data[col].to_numpy(dtype=float)

    bc = ex.data_beta_coeff[list(coeff_mult.columns)]
    predict_from_coeff = np.exp((bc * coeff_mult).sum(axis=1).to_numpy())
    predict_direct = model.predict(test_data)

    max_diff = np.max(np.abs(predict_from_coeff / predict_direct - 1))
    assert max_diff < 1e-5, f"Max relative diff (weighted): {max_diff}"
