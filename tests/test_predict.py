"""Property tests for IBLM.predict()."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from iblm import IBLM, ExplainIBLM, load_freMTPLmini, split_into_train_validate_test
from iblm._model import _build_design_matrix, _get_xgb_feature_df

SEED = 42


# ---------------------------------------------------------------------------
# Helper: alternative "base_margin" prediction (mirrors R's test helper)
# ---------------------------------------------------------------------------


def _base_margin_predict(model: IBLM, newdata: pd.DataFrame) -> np.ndarray:
    """Feed GLM link predictions as XGBoost base_margin, then predict directly."""
    offset_var = model.offset_var
    offset = (
        newdata[offset_var].to_numpy(dtype=float)
        if offset_var and offset_var in newdata.columns
        else np.zeros(len(newdata))
    )
    X_glm = _build_design_matrix(
        newdata,
        model.predictor_vars["continuous"],
        model.predictor_vars["categorical"],
        model.cat_levels["all"],
        model.cat_levels["reference"],
    )
    glm_links = np.asarray(
        model.glm_model.predict(exog=X_glm, offset=offset, which="linear")
    )
    X_xgb = _get_xgb_feature_df(newdata, model.predictor_vars, model.cat_levels["all"])
    dmat = xgb.DMatrix(X_xgb, base_margin=glm_links, enable_categorical=True)
    return model.booster_model.predict(dmat)


# ---------------------------------------------------------------------------
# Property: beta-coeff reconstruction == predict()  (multiplicative/log-link)
# ---------------------------------------------------------------------------


def test_beta_coeff_equals_predict(model_poisson, splits):
    """Reconstruct predictions from data_beta_coeff and compare to predict()."""
    test_data = splits["test"]
    ex = ExplainIBLM(model_poisson, test_data)

    cont_vars = model_poisson.predictor_vars["continuous"]
    cat_vars = model_poisson.predictor_vars["categorical"]

    coeff_mult = pd.DataFrame({"bias": 1.0}, index=test_data.index)
    for col in model_poisson.predictor_vars["all"]:
        coeff_mult[col] = 1.0 if col in cat_vars else test_data[col].to_numpy(dtype=float)

    bc = ex.data_beta_coeff[list(coeff_mult.columns)]
    predict_from_coeff = np.exp((bc * coeff_mult).sum(axis=1).to_numpy())

    # predict() at unit exposure (offset=0) — data_beta_coeff doesn't include offset
    offset_var = model_poisson.offset_var
    test_unit = test_data.copy()
    if offset_var and offset_var in test_unit.columns:
        test_unit[offset_var] = 0.0
    predict_direct = model_poisson.predict(test_unit)

    max_diff = np.max(np.abs(predict_from_coeff / predict_direct - 1))
    assert max_diff < 1e-5, f"Max relative difference: {max_diff}"


# ---------------------------------------------------------------------------
# Property: base_margin method == predict()  (poisson, gaussian, gamma)
# ---------------------------------------------------------------------------


def _assert_base_margin_vs_predict(model, test_data, tol=1e-4):
    preds_direct = model.predict(test_data)
    preds_base_margin = _base_margin_predict(model, test_data)
    max_diff = np.max(np.abs(preds_base_margin / preds_direct - 1))
    mean_diff = np.mean(preds_base_margin / preds_direct - 1)
    assert max_diff < tol, f"Max relative difference: {max_diff}"
    assert abs(mean_diff) < 1e-6, f"Mean relative difference: {mean_diff}"


def test_base_margin_equals_predict_poisson(model_poisson, splits):
    _assert_base_margin_vs_predict(model_poisson, splits["test"])


def test_base_margin_equals_predict_gaussian(model_gaussian, splits_with_exposure):
    s = {
        key: (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
            .drop(columns=["ClaimNb", "LogExposure"])
        )
        for key in ("train", "validate", "test")
    }
    _assert_base_margin_vs_predict(model_gaussian, s["test"])


def test_base_margin_equals_predict_gamma(model_gamma, splits_with_exposure):
    s = {
        key: (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: (x["ClaimNb"] / x["Exposure"]).clip(lower=0.01))
            .drop(columns=["ClaimNb", "LogExposure"])
        )
        for key in ("train", "validate", "test")
    }
    _assert_base_margin_vs_predict(model_gamma, s["test"])


# ---------------------------------------------------------------------------
# Property: response == exp(link)  for log-link families
# ---------------------------------------------------------------------------


def test_link_vs_response(model_poisson, splits):
    """predict(type='response') must equal exp(predict(type='link'))."""
    test_data = splits["test"]
    preds_response = model_poisson.predict(test_data, type="response")
    preds_link = model_poisson.predict(test_data, type="link")
    max_diff = np.max(np.abs(preds_response / np.exp(preds_link) - 1))
    assert max_diff < 1e-6, f"Max relative difference: {max_diff}"


# ---------------------------------------------------------------------------
# Property: trim parameter contracts booster corrections toward 1
# ---------------------------------------------------------------------------


def test_trim_contracts_booster(model_poisson, splits):
    """Predictions with trim=0 must all equal GLM predictions (booster capped to 1)."""
    test_data = splits["test"]

    preds_notrimmed = model_poisson.predict(test_data, trim=None)
    preds_trimmed = model_poisson.predict(test_data, trim=0)

    # With trim=0, booster is clipped to [1, 1] then renormalised → all 1s
    # So prediction == GLM prediction
    X_glm = _build_design_matrix(
        test_data,
        model_poisson.predictor_vars["continuous"],
        model_poisson.predictor_vars["categorical"],
        model_poisson.cat_levels["all"],
        model_poisson.cat_levels["reference"],
    )
    offset = test_data[model_poisson.offset_var].to_numpy(dtype=float)
    glm_preds = np.asarray(model_poisson.glm_model.predict(exog=X_glm, offset=offset))

    max_diff_trim0 = np.max(np.abs(preds_trimmed / glm_preds - 1))
    assert max_diff_trim0 < 1e-10, (
        f"trim=0 should equal GLM predictions; max diff: {max_diff_trim0}"
    )


def test_trim_invalid_type_raises(model_poisson, splits):
    """predict(type='invalid') must raise ValueError."""
    with pytest.raises(ValueError):
        model_poisson.predict(splits["test"], type="invalid")
