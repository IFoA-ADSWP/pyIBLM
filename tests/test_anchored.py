"""Anchored regression tests against R v1.0.3 baseline values.

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

homog / glm deviance  — 1e-8 relative
    Both depend only on the GLM; agreement with R is ~1e-11.

iblm deviance  — 0.05 relative (5 %)
    The GLM base_margin fed into XGBoost differs from R's by ~1e-9.
    These tiny differences compound over many boosting rounds, producing
    different tree paths (R: 0.2563, Python: 0.2607, Δ ≈ 1.7 %).
    A 5 % tolerance acknowledges this structural divergence while still
    detecting large regressions.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, get_pinball_scores, load_freMTPLmini


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
    return model, ps


# ---------------------------------------------------------------------------
# Test 1: GLM coefficients vs R anchor  (tolerance 1e-4 relative)
# ---------------------------------------------------------------------------


def test_glm_coefficients_vs_r(anchor_results):
    """GLM coefficients must agree with R v1.0.3 anchor to 1e-4 relative.

    Most coefficients match to ~1e-8.  VehBrandB12 is the worst at ~4e-6
    (near-zero value, small cell count).  Tolerance 1e-4 covers all cases.
    """
    model, _ = anchor_results
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

    homog / glm  — 1e-8 relative  (GLM-only; matches R to ~1e-11)
    iblm deviance     — 0.05 relative  (XGBoost paths diverge due to ~1e-9 GLM
                                       base_margin diff; actual gap ~1.7 %)
    iblm pinball_score — 0.02 absolute (1.7 % deviance gap amplifies to ~30 %
                                       in pinball space; absolute diff ~0.016)
    """
    _, ps = anchor_results
    ps_dict = {row["model"]: row for _, row in ps.iterrows()}

    # Deviance: relative tolerance
    #   homog / glm match R to ~1e-11 so 1e-8 is generous.
    #   iblm diverges by ~1.7 % due to XGBoost path differences; 0.05 covers it.
    dev_tol_rel = {"homog": 1e-8, "glm": 1e-8, "iblm": 0.05}

    # Pinball score: absolute tolerance
    #   homog is identically 0; glm matches R to ~1e-11.
    #   iblm: the 1.7 % deviance gap magnifies to ~30 % in pinball space
    #   (because pinball = 1 - dev/homog_dev amplifies near-1 ratios).
    #   Absolute diff is ~0.016 so 0.02 is the right measure here.
    pin_tol_abs = {"homog": 0.0, "glm": 1e-8, "iblm": 0.02}

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
