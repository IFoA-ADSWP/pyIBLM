"""Anchored regression tests against a fixed Python baseline.

Split strategy mirrors the R v1.0.3 test exactly:
  rep(c("train","train","train","validate","test"), times=5000)
→ deterministic, row-order-based, no random sampling.

XGBoost params (seed=0, tree_method="hist", nthread=1) guarantee
bit-identical results within Python across platforms.

GLM coefficient comparison vs R v1.0.3 anchor
----------------------------------------------
Most coefficients agree to ~1e-8 relative.  VehBrandB12 and VehBrandB6 are
the worst at ~4e-6 and ~2e-7 respectively — both are small-cell categorical
levels where the IRLS implementations (R stats::glm vs statsmodels) converge
to slightly different saddle points in a shallow curvature region.

Pinball score comparison vs R v1.0.3 anchor
--------------------------------------------
  homog  : matches R to 11 s.f. ✓   (pure arithmetic)
  glm    : matches R to 11 s.f. ✓   (GLM only, same data)
  iblm   : DIVERGES from R           (GLM base_margin ~1e-9 diff → different
            XGBoost tree paths after many rounds → 0.2563 in R vs 0.2607 in Py)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, get_pinball_scores, load_freMTPLmini


# ---------------------------------------------------------------------------
# Deterministic split helper (mirrors R's rep() assignment)
# ---------------------------------------------------------------------------


def _make_deterministic_splits():
    """Return splits using the same row-order pattern as the R anchor test."""
    df = load_freMTPLmini()
    n = len(df)
    pattern = ["train", "train", "train", "validate", "test"]
    assert n % len(pattern) == 0, f"Expected {n} to be divisible by {len(pattern)}"
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
# Python baseline — established from iblm v0.1.0
# ---------------------------------------------------------------------------

_PYTHON_GLM_COEFFS = {
    "(Intercept)": -3.859470828513518,
    "BonusMalus":   0.021559157382855673,
    "DrivAge":       0.008528529397239502,
    "VehAge":       -0.04270479533443885,
    "VehPower":      0.05894266370576127,
    "AreaB":        -0.11190174628227387,
    "AreaC":        -0.4449171445888056,
    "AreaD":        -0.24843582848868,
    "AreaE":        -0.14784576311798525,
    "VehBrandB12":   0.0011925718609876648,
    "VehBrandB2":   -0.023898926567742354,
    "VehBrandB3":    0.07594077373013756,
    "VehBrandB4":   -0.2654570613041361,
    "VehBrandB5":    0.2914909877798007,
    "VehBrandB6":   -0.004878175146042017,
}

_PYTHON_PINBALL = {
    "homog": {"poisson_deviance": 0.27083016955146094, "pinball_score": 0.0},
    "glm":   {"poisson_deviance": 0.26744791988591143, "pinball_score": 0.012488452343219603},
    "iblm":  {"poisson_deviance": 0.2606682683432493,  "pinball_score": 0.037521304310525694},
}

# R v1.0.3 anchor — for cross-language correspondence checks
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

_R_PINBALL = {
    "homog": {"poisson_deviance": 0.27083016955873457, "pinball_score": 0.0},
    "glm":   {"poisson_deviance": 0.26744791989655214, "pinball_score": 0.012488452330451705},
    "iblm":  {"poisson_deviance": 0.25633025817301236, "pinball_score": 0.05353875976722611},
}


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
# Test 1: GLM coefficients — Python regression anchor (tight)
# ---------------------------------------------------------------------------


def test_glm_coefficients_python_anchor(anchor_results):
    """GLM coefficients must match the Python v0.1.0 baseline to 1e-10 absolute."""
    model, _ = anchor_results
    params = model.glm_model.params  # pd.Series

    for name, expected in _PYTHON_GLM_COEFFS.items():
        actual = float(params[name])
        assert abs(actual - expected) < 1e-10, (
            f"GLM coeff '{name}': got {actual!r}, expected {expected!r}, "
            f"diff {abs(actual - expected):.2e}"
        )


# ---------------------------------------------------------------------------
# Test 2: GLM coefficients — cross-language correspondence with R
# ---------------------------------------------------------------------------


def test_glm_coefficients_vs_r_anchor(anchor_results):
    """Python GLM coefficients must agree with R v1.0.3 anchor to 1e-4 relative.

    Most coefficients match to ~1e-8.  VehBrandB12 (~4e-6) and VehBrandB6
    (~2e-7) are the worst — small-cell levels in a shallow curvature region
    where R's stats::glm and statsmodels converge to slightly different points.
    The 1e-4 tolerance is deliberately loose to survive these edge cases while
    still catching genuine regressions.
    """
    model, _ = anchor_results
    params = model.glm_model.params

    for name, r_val in _R_GLM_COEFFS.items():
        py_val = float(params[name])
        rel_diff = abs(py_val / r_val - 1) if r_val != 0 else abs(py_val)
        assert rel_diff < 1e-4, (
            f"GLM coeff '{name}' vs R: got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e} (threshold 1e-4)"
        )


# ---------------------------------------------------------------------------
# Test 3: Pinball scores — Python regression anchor (tight)
# ---------------------------------------------------------------------------


def test_pinball_scores_python_anchor(anchor_results):
    """Pinball scores must match the Python v0.1.0 baseline to 1e-10 absolute."""
    _, ps = anchor_results
    ps_dict = {row["model"]: row for _, row in ps.iterrows()}

    for model_name, expected in _PYTHON_PINBALL.items():
        row = ps_dict[model_name]
        for col, exp_val in expected.items():
            got = float(row[col])
            assert abs(got - exp_val) < 1e-10, (
                f"{model_name} {col}: got {got!r}, expected {exp_val!r}, "
                f"diff {abs(got - exp_val):.2e}"
            )


# ---------------------------------------------------------------------------
# Test 4: homog + glm pinball scores — cross-language correspondence with R
# ---------------------------------------------------------------------------


def test_pinball_scores_homog_glm_vs_r_anchor(anchor_results):
    """homog and glm deviances must match R v1.0.3 anchor to 1e-8 relative.

    These depend only on the statsmodels GLM (not XGBoost) so agreement is
    near-machine-precision.  iblm is excluded because divergent XGBoost tree
    paths (see module docstring) give R=0.2563 vs Python=0.2607.
    """
    _, ps = anchor_results
    ps_dict = {row["model"]: row for _, row in ps.iterrows()}

    for model_name in ("homog", "glm"):
        row = ps_dict[model_name]
        r_val = _R_PINBALL[model_name]["poisson_deviance"]
        py_val = float(row["poisson_deviance"])
        rel_diff = abs(py_val / r_val - 1)
        assert rel_diff < 1e-8, (
            f"{model_name} poisson_deviance vs R: got {py_val!r}, R={r_val!r}, "
            f"relative diff {rel_diff:.2e}"
        )
