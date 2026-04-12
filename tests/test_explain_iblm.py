"""Smoke and property tests for ExplainIBLM."""

import warnings

import numpy as np
import pandas as pd
import pytest

from iblm import (
    IBLM,
    ExplainIBLM,
    load_freMTPLmini,
    split_into_train_validate_test,
)

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit(df_or_splits, response_var, family, **kwargs):
    if isinstance(df_or_splits, dict):
        splits = df_or_splits
    else:
        splits = split_into_train_validate_test(df_or_splits, seed=SEED)
    m = IBLM()
    m.fit(
        splits,
        response_var=response_var,
        family=family,
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
        **kwargs,
    )
    return m, splits


def _smoke_explain(model, test_data, cat_var=None, cont_var=None):
    ex = ExplainIBLM(model, test_data)
    if cat_var:
        ex.beta_corrected_scatter(cat_var)
        ex.beta_corrected_density(cat_var)
    if cont_var:
        ex.beta_corrected_scatter(cont_var)
        ex.beta_corrected_density(cont_var)
    ex.overall_correction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ex.bias_density()
    return ex


# ---------------------------------------------------------------------------
# Smoke: variable-type combinations
# ---------------------------------------------------------------------------


def test_explain_one_cat_one_cont():
    """explain_iblm runs without error with one categorical and one continuous."""
    df = load_freMTPLmini()[["VehBrand", "VehPower", "ClaimNb"]]
    model, splits = _fit(df, "ClaimNb", "poisson")
    _smoke_explain(model, splits["test"], cat_var="VehBrand", cont_var="VehPower")


def test_explain_cat_only():
    """explain_iblm runs without error with categorical predictors only."""
    df = load_freMTPLmini()[["VehBrand", "Area", "ClaimNb"]]
    model, splits = _fit(df, "ClaimNb", "poisson")
    _smoke_explain(model, splits["test"], cat_var="VehBrand")


def test_explain_cont_only():
    """explain_iblm runs without error with continuous predictors only."""
    df = load_freMTPLmini()[["VehPower", "VehAge", "DrivAge", "BonusMalus", "ClaimNb"]]
    model, splits = _fit(df, "ClaimNb", "poisson")
    _smoke_explain(model, splits["test"], cont_var="VehPower")


def test_explain_with_weight():
    """explain_iblm runs without error when a weight variable is used."""
    df = load_freMTPLmini()
    df = df.assign(ClaimRate=df["ClaimNb"] / df["Exposure"]).drop(
        columns=["ClaimNb", "LogExposure"], errors="ignore"
    )
    splits = split_into_train_validate_test(df, seed=SEED)
    model, splits = _fit(splits, "ClaimRate", "poisson", weight_var="Exposure")
    _smoke_explain(model, splits["test"], cat_var="Area", cont_var="VehPower")


def test_explain_no_zero_values():
    """explain_iblm runs without error when no continuous feature equals zero (no bias migration)."""
    df = load_freMTPLmini()[["VehPower", "VehAge", "DrivAge", "BonusMalus", "ClaimNb"]].copy()
    for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus"]:
        df[col] = df[col].clip(lower=1)
    model, splits = _fit(df, "ClaimNb", "poisson")
    _smoke_explain(model, splits["test"], cont_var="VehPower")


# ---------------------------------------------------------------------------
# Smoke: families
# ---------------------------------------------------------------------------


def test_explain_gaussian(model_gaussian, splits_with_exposure):
    """explain_iblm smoke-test with gaussian family."""
    s = {
        key: (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
            .drop(columns=["ClaimNb", "LogExposure"])
        )
        for key in ("train", "validate", "test")
    }
    _smoke_explain(model_gaussian, s["test"], cat_var="Area", cont_var="VehPower")


def test_explain_gamma(model_gamma, splits_with_exposure):
    """explain_iblm smoke-test with gamma family."""
    s = {
        key: (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: (x["ClaimNb"] / x["Exposure"]).clip(lower=0.01))
            .drop(columns=["ClaimNb", "LogExposure"])
        )
        for key in ("train", "validate", "test")
    }
    _smoke_explain(model_gamma, s["test"], cat_var="Area", cont_var="VehPower")


def test_explain_tweedie(model_tweedie, splits_with_exposure):
    """explain_iblm smoke-test with tweedie family."""
    s = {
        key: (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
            .drop(columns=["ClaimNb", "LogExposure"])
        )
        for key in ("train", "validate", "test")
    }
    _smoke_explain(model_tweedie, s["test"], cat_var="Area", cont_var="VehPower")


def test_explain_custom_objective():
    """Overriding the default XGBoost objective runs without error."""
    df = load_freMTPLmini().drop(columns=["LogExposure"], errors="ignore")
    splits = split_into_train_validate_test(df, seed=SEED)
    model = IBLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            splits,
            response_var="ClaimNb",
            family="poisson",
            params={"objective": "reg:squarederror", "seed": 0, "max_depth": 3},
            nrounds=200,
            early_stopping_rounds=20,
            verbose=0,
        )
    _smoke_explain(model, splits["test"], cat_var="Area", cont_var="VehPower")


# ---------------------------------------------------------------------------
# Property: migrate_reference_to_bias TRUE vs FALSE → same predictions
# ---------------------------------------------------------------------------


def test_migrate_vs_no_migrate_same_predictions():
    """Toggling migrate_reference_to_bias must not change reconstructed predictions."""
    df = load_freMTPLmini().drop(columns=["LogExposure"], errors="ignore")
    splits = split_into_train_validate_test(df, seed=SEED)
    model, splits = _fit(splits, "ClaimNb", "poisson")
    test_data = splits["test"]

    ex_migrate = ExplainIBLM(model, test_data, migrate_reference_to_bias=True)
    ex_no_migrate = ExplainIBLM(model, test_data, migrate_reference_to_bias=False)

    # Build coeff_multiplier: bias=1, categorical=1, continuous=feature value
    cont_vars = model.predictor_vars["continuous"]
    cat_vars = model.predictor_vars["categorical"]

    coeff_mult = pd.DataFrame({"bias": 1.0}, index=test_data.index)
    for col in model.predictor_vars["all"]:
        coeff_mult[col] = 1.0 if col in cat_vars else test_data[col].to_numpy(dtype=float)

    def reconstruct(ex):
        bc = ex.data_beta_coeff[list(coeff_mult.columns)]
        return np.exp((bc * coeff_mult).sum(axis=1).to_numpy())

    preds_migrate = reconstruct(ex_migrate)
    preds_no_migrate = reconstruct(ex_no_migrate)

    max_diff = np.max(np.abs(preds_migrate / preds_no_migrate - 1))
    assert max_diff < 1e-10, f"Max relative difference: {max_diff}"
