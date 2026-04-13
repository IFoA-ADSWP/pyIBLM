"""Anchored regression tests against R v1.0.3 baseline values.

Test suite 1 — Poisson, with offset, explicit seed=0
=====================================================
Split strategy mirrors the R test exactly:
  rep(c("train","train","train","validate","test"), times=5000)
→ deterministic, row-order-based, no random sampling.

XGBoost params (seed=0, tree_method="hist", nthread=1) guarantee
bit-identical XGBoost results within Python across platforms.

Tolerances
----------
GLM coefficients  — 1e-4 relative
    Most agree to ~1e-8. VehBrandB12 is the worst at ~4e-6 (near-zero
    coefficient in a small cell; shallow curvature makes the IRLS saddle
    points of R stats::glm and statsmodels differ more proportionally).
    1e-4 covers the worst case with room to spare.

homog / glm / iblm deviance  — 1e-8 relative
    All three depend only on the GLM + XGBoost best_iteration trees.
    Agreement with R is ~1e-10 to 1e-11.

    Root cause of prior divergence (now fixed): Python's
    xgb.callback.EarlyStopping retains *all* trained trees in the booster
    (best_iteration + patience rounds = 39 trees), while R's xgb.train
    with early_stopping_rounds trims the booster to best_iteration trees
    after training.  The fix is EarlyStopping(save_best=True), which makes
    Python match R's behaviour exactly.

Test suite 2 — Gaussian, no offset, modified response, explicit seed=42
========================================================================
Uses the same deterministic split but:
  • ClaimNb is modified: sign-flip on odd R rows + (DrivAge * VehAge)/1000,
    making it a continuous regression target.
  • No LogExposure offset.
  • family="gaussian" (identity link, additive relationship).
  • Explicit seed=42, no subsampling → XGBoost paths are bit-identical
    between R xgboost 3.x and Python xgboost 2.x.

Tolerances
----------
GLM coefficients  — 1e-8 relative
    OLS is solved via exact linear algebra; agreement with R is ~1e-13 to
    1e-14.  VehBrandB4 is the worst at ~1.6e-9 (near-zero coefficient ~9e-8;
    even tiny absolute differences produce large relative gaps).

best_iteration    — exact integer equality (both = 99)
best_score        — exact floating-point equality (both = 0.24031355492261605)
    With nthread=1 and an explicit seed, the XGBoost C++ RNG is deterministic
    across R and Python for the same xgboost shared library.

beta_corrections colSums — 1e-8 relative
    SHAP values from XGBoost agree to machine precision (~0 to 2.2e-16);
    observed worst case is 2.2e-16 (VehPower, one ULP).

data_beta_coeff colSums — 1e-8 relative
    GLM-derived; agreement is ~1e-13 to 1e-14.

homog / glm / iblm deviance — 1e-8 relative
    All three agree with R to floating-point precision (~0 to 6.7e-16).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, get_pinball_scores, load_freMTPLmini
from iblm._explain import ExplainIBLM


# ---------------------------------------------------------------------------
# Deterministic split (mirrors R's rep() assignment)
# ---------------------------------------------------------------------------


def _make_deterministic_splits():
    df = load_freMTPLmini()
    n = len(df)
    pattern = ["train", "train", "train", "validate", "test"]
    assert n % len(pattern) == 0
    labels = np.tile(pattern, n // len(pattern))
    df = df.assign(split=labels, LogExposure=np.log(df["Exposure"])).drop(columns=["Exposure"])
    return {
        k: df[df["split"] == k].drop(columns=["split"]).reset_index(drop=True)
        for k in ("train", "validate", "test")
    }


def _fit_anchor_model(splits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IBLM()
        model.fit(
            splits,
            response_var="ClaimNb",
            offset_var="LogExposure",
            family="poisson",
            params={
                "objective": "count:poisson",
                "seed": 0,
                "tree_method": "hist",
                "nthread": 1,
            },
            nrounds=1000,
            early_stopping_rounds=25,
            verbose=0,
        )
    return model


# ---------------------------------------------------------------------------
# R v1.0.3 anchor values
# ---------------------------------------------------------------------------

_R_GLM_COEFFS = {
    "(Intercept)": -3.8594708173404872,
    "AreaB":       -0.11190174836580087,
    "AreaC":       -0.444917144255031,
    "AreaD":       -0.2484358290731008,
    "AreaE":       -0.1478457642136018,
    "BonusMalus":   0.021559157295387592,
    "DrivAge":       0.008528529315074463,
    "VehAge":       -0.042704795183391375,
    "VehBrandB12":   0.0011925765912662008,
    "VehBrandB2":   -0.023898926969100273,
    "VehBrandB3":    0.07594077341978828,
    "VehBrandB4":   -0.26545705985518625,
    "VehBrandB5":    0.29149098741982354,
    "VehBrandB6":   -0.004878175985223516,
    "VehPower":      0.05894266359794995,
}

_R_BOOSTER_SCORE     = 0.18048168787918548
_R_BOOSTER_ITERATION = 13

_R_BETA_CORRECTIONS_COLSUMS = {
    "bias":        48.13453136815224,
    "BonusMalus":  -0.20176153548366793,
    "DrivAge":     -5.775991371225044,
    "VehAge":      -173.63288308576503,
    "VehPower":     5.272630189453419,
    "AreaA":        0.0,
    "AreaB":        10.82191365386825,
    "AreaC":        27.738016427087132,
    "AreaD":       -32.716441521886736,
    "AreaE":        -1.529093350225594,
    "VehBrandB1":   0.0,
    "VehBrandB12": -197.57205333554884,
    "VehBrandB2":  -21.20856424618978,
    "VehBrandB3":   -9.44511840166524,
    "VehBrandB4":   -0.21352318883873522,
    "VehBrandB5":   -6.173628297285177,
    "VehBrandB6":  -20.332186447805725,
}

_R_DATA_BETA_COEFF_COLSUMS = {
    "bias":       -19249.219555334283,
    "Area":        -1289.2881240635083,
    "BonusMalus":   107.5940249414543,
    "DrivAge":       36.86665520414727,
    "VehAge":      -387.15685900272194,
    "VehBrand":    -237.55534919289332,
    "VehPower":     299.98594817920315,
}

_R_PINBALL = pd.DataFrame({
    "model":            ["homog", "glm", "iblm"],
    "poisson_deviance": [0.27083016955873457, 0.26744791989655214, 0.25633025817301236],
    "pinball_score":    [0.0,                 0.012488452330451705, 0.05353875976722611],
})


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anchor_results():
    splits = _make_deterministic_splits()
    model = _fit_anchor_model(splits)
    ps = get_pinball_scores(splits["test"], model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = ExplainIBLM(model, splits["test"])
    return model, ps, explainer


# ---------------------------------------------------------------------------
# Test 1: Booster best_score and best_iteration vs R anchor
# ---------------------------------------------------------------------------


def test_booster_score_and_iteration_vs_r(anchor_results):
    """Booster best_score and best_iteration must match R v1.0.3 anchor.

    best_iteration is an integer and must be identical (13).
    best_score differs by ~1e-9 relative — the GLM base_margin divergence
    flows into the Poisson deviance metric; 1e-6 relative covers it.
    """
    model, _, _ex = anchor_results
    b = model.booster_model

    assert b.best_iteration == _R_BOOSTER_ITERATION, (
        f"best_iteration: got {b.best_iteration}, expected {_R_BOOSTER_ITERATION}"
    )

    rel_diff = abs(b.best_score / _R_BOOSTER_SCORE - 1)
    assert rel_diff < 1e-6, (
        f"best_score: got {b.best_score!r}, R={_R_BOOSTER_SCORE!r}, "
        f"relative diff {rel_diff:.2e} (threshold 1e-6)"
    )


# ---------------------------------------------------------------------------
# Test 2: GLM coefficients vs R anchor  (tolerance 1e-4 relative)
# ---------------------------------------------------------------------------


def test_glm_coefficients_vs_r(anchor_results):
    """GLM coefficients must agree with R v1.0.3 anchor to 1e-4 relative.

    Most coefficients match to ~1e-8.  VehBrandB12 is the worst at ~4e-6
    (near-zero value, small cell count).  Tolerance 1e-4 covers all cases.
    """
    model, _, _ex = anchor_results
    params = model.glm_model.params

    for name, r_val in _R_GLM_COEFFS.items():
        py_val = float(params[name])
        rel_diff = abs(py_val / r_val - 1)
        assert rel_diff < 1e-4, (
            f"GLM coeff '{name}': got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e} (threshold 1e-4)"
        )


# ---------------------------------------------------------------------------
# Test 2: Pinball scores vs R anchor
# ---------------------------------------------------------------------------


def test_pinball_scores_vs_r(anchor_results):
    """Pinball scores must agree with R v1.0.3 anchor within stated tolerances.

    All models — 1e-8 relative / absolute.
    The GLM-only models (homog, glm) match R to ~1e-11.
    iblm matches R to ~1e-10 once the EarlyStopping(save_best=True) fix is
    applied (Python now uses only best_iteration trees, same as R).
    """
    _, ps, _ex = anchor_results
    ps_dict = {row["model"]: row for _, row in ps.iterrows()}

    # Deviance: relative tolerance — all three match R to ~1e-10 or better.
    dev_tol_rel = {"homog": 1e-8, "glm": 1e-8, "iblm": 1e-8}

    # Pinball score: absolute tolerance — same level of agreement.
    pin_tol_abs = {"homog": 0.0, "glm": 1e-8, "iblm": 1e-8}

    for _, r_row in _R_PINBALL.iterrows():
        model_name = r_row["model"]
        py_row = ps_dict[model_name]

        r_dev = r_row["poisson_deviance"]
        py_dev = float(py_row["poisson_deviance"])
        rel_diff = abs(py_dev / r_dev - 1)
        tol_d = dev_tol_rel[model_name]
        assert rel_diff < tol_d, (
            f"{model_name} poisson_deviance: got {py_dev!r}, R={r_dev!r}, "
            f"relative diff {rel_diff:.2e} (threshold {tol_d})"
        )

        r_pin = r_row["pinball_score"]
        py_pin = float(py_row["pinball_score"])
        tol_p = pin_tol_abs[model_name]
        assert abs(py_pin - r_pin) <= tol_p, (
            f"{model_name} pinball_score: got {py_pin!r}, R={r_pin!r}, "
            f"absolute diff {abs(py_pin - r_pin):.2e} (threshold {tol_p})"
        )


# ---------------------------------------------------------------------------
# Test 4: beta_corrections column sums vs R anchor  (tolerance 1e-5 relative)
# ---------------------------------------------------------------------------


def test_beta_corrections_colsums_vs_r(anchor_results):
    """beta_corrections column sums must agree with R v1.0.3 anchor to 1e-5 relative.

    SHAP values from XGBoost agree to ~2e-6 relative in the worst case
    (bias, AreaE, VehBrandB4).  Zero columns (AreaA, VehBrandB1 — reference
    levels) are checked with an absolute tolerance of 0.
    1e-5 relative covers the worst case with room to spare.
    """
    _, _, explainer = anchor_results
    bc_sums = explainer.beta_corrections.sum()

    for col, r_val in _R_BETA_CORRECTIONS_COLSUMS.items():
        py_val = float(bc_sums[col])
        if r_val == 0.0:
            assert py_val == 0.0, (
                f"beta_corrections['{col}'] colsum: got {py_val!r}, expected 0.0"
            )
        else:
            rel_diff = abs(py_val / r_val - 1)
            assert rel_diff < 1e-5, (
                f"beta_corrections['{col}'] colsum: got {py_val!r}, R={r_val!r}, "
                f"relative diff {rel_diff:.2e} (threshold 1e-5)"
            )


# ---------------------------------------------------------------------------
# Test 5: data_beta_coeff column sums vs R anchor  (tolerance 1e-7 relative)
# ---------------------------------------------------------------------------


def test_data_beta_coeff_colsums_vs_r(anchor_results):
    """data_beta_coeff column sums must agree with R v1.0.3 anchor to 1e-7 relative.

    These sums depend only on the GLM coefficients and the test data, so they
    match R to ~1e-8 relative.  1e-7 is a comfortable guard.
    """
    _, _, explainer = anchor_results
    dbc_sums = explainer.data_beta_coeff.sum()

    for col, r_val in _R_DATA_BETA_COEFF_COLSUMS.items():
        py_val = float(dbc_sums[col])
        rel_diff = abs(py_val / r_val - 1)
        assert rel_diff < 1e-7, (
            f"data_beta_coeff['{col}'] colsum: got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e} (threshold 1e-7)"
        )


# ===========================================================================
# Test suite 2: Gaussian family, no offset, alternating-sign ClaimNb
# ===========================================================================


# ---------------------------------------------------------------------------
# Data prep + model fit (mirrors R test "mini 2")
# ---------------------------------------------------------------------------


def _make_gaussian_splits():
    """Mirror R's data preparation for the Gaussian anchored test.

    R code::
        freMTPLmini |>
          mutate(train_validate_test = rep(c("train","train","train","validate","test"), times=5000)) |>
          mutate(ClaimNb = if_else(row_number() %% 2 == 1, 1, -1) * ClaimNb
                           + (DrivAge * VehAge) / 1000) |>
          select(-Exposure) |>
          split(~train_validate_test) |>
          map(function(x) select(x, -train_validate_test))

    R's row_number() is 1-indexed: rows 1, 3, 5, … (odd) → +1; rows 2, 4, 6, … → -1.
    Python (0-indexed): index 0, 2, 4, … → R row 1, 3, 5, … → positive.
    """
    df = load_freMTPLmini()
    n = len(df)
    pattern = ["train", "train", "train", "validate", "test"]
    assert n % len(pattern) == 0
    labels = np.tile(pattern, n // len(pattern))
    sign = np.where(np.arange(n) % 2 == 0, 1, -1)
    df = df.assign(
        split=labels,
        ClaimNb=sign * df["ClaimNb"] + (df["DrivAge"] * df["VehAge"]) / 1000,
    ).drop(columns=["Exposure"])
    return {
        k: df[df["split"] == k].drop(columns=["split"]).reset_index(drop=True)
        for k in ("train", "validate", "test")
    }


def _fit_anchor_model_gauss(splits):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = IBLM()
        model.fit(
            splits,
            response_var="ClaimNb",
            family="gaussian",
            params={
                "seed":        42,
                "tree_method": "hist",
                "nthread":     1,
                "eta":         0.05,
                "lambda":      2.0,
            },
            nrounds=1000,
            early_stopping_rounds=100,
            verbose=0,
        )
    return model


# ---------------------------------------------------------------------------
# R v1.0.3 anchor values — Gaussian test
# ---------------------------------------------------------------------------

_R2_GLM_COEFFS = {
    "(Intercept)":  -0.2025413916786566,
    "AreaB":        -0.023471030505357086,
    "AreaC":        -0.012140587927466915,
    "AreaD":        -0.010996950580295135,
    "AreaE":        -0.007953646029726714,
    "BonusMalus":   -0.0002338431728971184,
    "DrivAge":       0.0058192444991261374,
    "VehAge":        0.04071492793289529,
    "VehBrandB12":  -0.004293462462660974,
    "VehBrandB2":   -0.002182175681227498,
    "VehBrandB3":    0.002837980312847773,
    "VehBrandB4":    9.35937469434587e-08,
    "VehBrandB5":    0.007794208317519717,
    "VehBrandB6":   -0.002785564379060474,
    "VehPower":     -0.0020011279950285315,
}

_R2_BOOSTER_SCORE     = 0.24031355492261605
_R2_BOOSTER_ITERATION = 99

_R2_BETA_CORRECTIONS_COLSUMS = {
    "bias":        4.142185995416639,
    "BonusMalus": -0.024477002752042034,
    "DrivAge":    -0.1165565642249349,
    "VehAge":     -4.385297857851767,
    "VehPower":   -0.13806569857106246,
    "AreaA":       0.0,
    "AreaB":       0.27601326090530165,
    "AreaC":      -0.18794983586803937,
    "AreaD":      -0.5743794106633686,
    "AreaE":      -0.006268041955934223,
    "VehBrandB1":  0.0,
    "VehBrandB12": 1.3853131091244677,
    "VehBrandB2": -1.196247227967433,
    "VehBrandB3": -0.35058218140443387,
    "VehBrandB4": -0.15855255764290632,
    "VehBrandB5":  0.29630929163431574,
    "VehBrandB6": -0.013268982924955708,
}

_R2_DATA_BETA_COEFF_COLSUMS = {
    "bias":       -1008.5647723978664,
    "Area":         -54.9651351375618,
    "BonusMalus":    -1.1936928672376341,
    "DrivAge":       28.97966593140575,
    "VehAge":       199.18934180662467,
    "VehBrand":      -6.515258712434496,
    "VehPower":     -10.14370567371372,
}

_R2_PINBALL = pd.DataFrame({
    "model":             ["homog", "glm",                     "iblm"],
    "gaussian_deviance": [0.09755753957408,
                          0.04795265336814491,
                          0.044498189422960646],
    "pinball_score":     [0.0,
                          0.508467991530965,
                          0.5438774940693221],
})


# ---------------------------------------------------------------------------
# Shared fixture — Gaussian suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anchor_results_gauss():
    splits = _make_gaussian_splits()
    model = _fit_anchor_model_gauss(splits)
    ps = get_pinball_scores(splits["test"], model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = ExplainIBLM(model, splits["test"])
    return model, ps, explainer


# ---------------------------------------------------------------------------
# Test G1: GLM coefficients vs R anchor  (tolerance 1e-8 relative)
# ---------------------------------------------------------------------------


def test_glm_coefficients_vs_r_gauss(anchor_results_gauss):
    """Gaussian OLS coefficients must agree with R v1.0.3 anchor to 1e-8 relative.

    OLS is solved via exact linear algebra; Python and R agree to ~1e-13 to 1e-14.
    VehBrandB4 is the worst at ~1.6e-9 (near-zero coefficient ~9e-8).
    1e-8 covers the worst case with room to spare.
    """
    model, _, _ex = anchor_results_gauss

    for name, r_val in _R2_GLM_COEFFS.items():
        py_val = float(model.glm_model.params[name])
        rel_diff = abs(py_val / r_val - 1)
        assert rel_diff < 1e-8, (
            f"GLM coeff '{name}': got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e} (threshold 1e-8)"
        )


# ---------------------------------------------------------------------------
# Test G2: Booster best_score and best_iteration vs R anchor  (exact)
# ---------------------------------------------------------------------------


def test_booster_score_and_iteration_vs_r_gauss(anchor_results_gauss):
    """Booster best_score and best_iteration must be exactly equal to R v1.0.3 anchor.

    With nthread=1 and explicit seed=42, XGBoost's C++ RNG is fully
    deterministic, so Python and R produce bit-identical booster results.
    best_iteration (99) and best_score (0.24031355492261605) are both exact.
    """
    model, _, _ex = anchor_results_gauss
    b = model.booster_model

    assert b.best_iteration == _R2_BOOSTER_ITERATION, (
        f"best_iteration: got {b.best_iteration}, expected {_R2_BOOSTER_ITERATION}"
    )
    assert b.best_score == _R2_BOOSTER_SCORE, (
        f"best_score: got {b.best_score!r}, expected {_R2_BOOSTER_SCORE!r}"
    )


# ---------------------------------------------------------------------------
# Test G3: beta_corrections column sums vs R anchor  (tolerance 1e-8 relative)
# ---------------------------------------------------------------------------


def test_beta_corrections_colsums_vs_r_gauss(anchor_results_gauss):
    """beta_corrections column sums must agree with R v1.0.3 anchor to 1e-8 relative.

    With seed=42 and no subsampling, SHAP values are bit-identical between
    R and Python; observed worst case is ~2.2e-16 (one ULP, VehPower).
    Reference-level columns (AreaA, VehBrandB1) are asserted exactly zero.
    """
    _, _, explainer = anchor_results_gauss
    bc_sums = explainer.beta_corrections.sum()

    for col, r_val in _R2_BETA_CORRECTIONS_COLSUMS.items():
        py_val = float(bc_sums[col])
        if r_val == 0.0:
            assert py_val == 0.0, (
                f"beta_corrections['{col}'] colsum: got {py_val!r}, expected 0.0"
            )
        else:
            rel_diff = abs(py_val / r_val - 1)
            assert rel_diff < 1e-8, (
                f"beta_corrections['{col}'] colsum: got {py_val!r}, R={r_val!r}, "
                f"relative diff {rel_diff:.2e} (threshold 1e-8)"
            )


# ---------------------------------------------------------------------------
# Test G4: data_beta_coeff column sums vs R anchor  (tolerance 1e-8 relative)
# ---------------------------------------------------------------------------


def test_data_beta_coeff_colsums_vs_r_gauss(anchor_results_gauss):
    """data_beta_coeff column sums must agree with R v1.0.3 anchor to 1e-8 relative.

    GLM-derived; agreement is ~1e-13 to 1e-14.  1e-8 is a conservative guard.
    """
    _, _, explainer = anchor_results_gauss
    dbc_sums = explainer.data_beta_coeff.sum()

    for col, r_val in _R2_DATA_BETA_COEFF_COLSUMS.items():
        py_val = float(dbc_sums[col])
        rel_diff = abs(py_val / r_val - 1)
        assert rel_diff < 1e-8, (
            f"data_beta_coeff['{col}'] colsum: got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e} (threshold 1e-8)"
        )


# ---------------------------------------------------------------------------
# Test G5: Pinball scores vs R anchor  (tolerance 1e-8)
# ---------------------------------------------------------------------------


def test_pinball_scores_vs_r_gauss(anchor_results_gauss):
    """Pinball scores must agree with R v1.0.3 anchor to 1e-8 relative / absolute.

    With seed=42 and no subsampling, XGBoost predictions are bit-identical;
    all deviances and pinball scores agree with R to floating-point precision
    (~0 to 6.7e-16 observed).
    """
    _, ps, _ex = anchor_results_gauss
    ps_dict = {row["model"]: row for _, row in ps.iterrows()}

    dev_tol_rel = {"homog": 1e-8, "glm": 1e-8, "iblm": 1e-8}
    pin_tol_abs = {"homog": 0.0,  "glm": 1e-8, "iblm": 1e-8}

    for _, r_row in _R2_PINBALL.iterrows():
        model_name = r_row["model"]
        py_row = ps_dict[model_name]

        r_dev = r_row["gaussian_deviance"]
        py_dev = float(py_row["gaussian_deviance"])
        rel_diff = abs(py_dev / r_dev - 1)
        tol_d = dev_tol_rel[model_name]
        assert rel_diff < tol_d, (
            f"{model_name} gaussian_deviance: got {py_dev!r}, R={r_dev!r}, "
            f"relative diff {rel_diff:.2e} (threshold {tol_d})"
        )

        r_pin = r_row["pinball_score"]
        py_pin = float(py_row["pinball_score"])
        tol_p = pin_tol_abs[model_name]
        assert abs(py_pin - r_pin) <= tol_p, (
            f"{model_name} pinball_score: got {py_pin!r}, R={r_pin!r}, "
            f"absolute diff {abs(py_pin - r_pin):.2e} (threshold {tol_p})"
        )
