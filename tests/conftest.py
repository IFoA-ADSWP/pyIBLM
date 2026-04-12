"""Shared fixtures for the IBLM test suite."""

import matplotlib
matplotlib.use("Agg")  # headless backend – must be set before any other plt import

import numpy as np
import pandas as pd
import pytest

from iblm import IBLM, load_freMTPLmini, split_into_train_validate_test

SEED = 42  # fixed Python RNG seed used throughout


# ---------------------------------------------------------------------------
# Base dataset and splits
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mini() -> pd.DataFrame:
    """Full freMTPLmini dataset with LogExposure derived from Exposure."""
    df = load_freMTPLmini()
    df["LogExposure"] = np.log(df["Exposure"])
    return df


@pytest.fixture(scope="session")
def splits(mini) -> dict:
    """Standard train/validate/test split of freMTPLmini (with LogExposure)."""
    df = mini.drop(columns=["Exposure"])
    return split_into_train_validate_test(df, seed=SEED)


@pytest.fixture(scope="session")
def splits_with_exposure(mini) -> dict:
    """Split where Exposure is kept (for weight tests)."""
    return split_into_train_validate_test(mini, seed=SEED)


# ---------------------------------------------------------------------------
# Fitted models (session-scoped so each is trained only once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_poisson(splits) -> IBLM:
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
    return m


@pytest.fixture(scope="session")
def model_gaussian(splits_with_exposure) -> IBLM:
    df = (
        splits_with_exposure["train"]
        .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
        .drop(columns=["ClaimNb", "LogExposure"])
    )
    # Rebuild dict with same structure
    s = {}
    for key in ("train", "validate", "test"):
        s[key] = (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
            .drop(columns=["ClaimNb", "LogExposure"])
        )
    m = IBLM()
    m.fit(
        s,
        response_var="ClaimRate",
        family="gaussian",
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
    )
    return m


@pytest.fixture(scope="session")
def model_gamma(splits_with_exposure) -> IBLM:
    s = {}
    for key in ("train", "validate", "test"):
        s[key] = (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: (x["ClaimNb"] / x["Exposure"]).clip(lower=0.01))
            .drop(columns=["ClaimNb", "LogExposure"])
        )
    m = IBLM()
    m.fit(
        s,
        response_var="ClaimRate",
        family="gamma",
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
    )
    return m


@pytest.fixture(scope="session")
def model_tweedie(splits_with_exposure) -> IBLM:
    s = {}
    for key in ("train", "validate", "test"):
        s[key] = (
            splits_with_exposure[key]
            .assign(ClaimRate=lambda x: x["ClaimNb"] / x["Exposure"])
            .drop(columns=["ClaimNb", "LogExposure"])
        )
    m = IBLM()
    m.fit(
        s,
        response_var="ClaimRate",
        family="tweedie",
        params={"seed": 0, "max_depth": 3},
        nrounds=200,
        early_stopping_rounds=20,
        verbose=0,
    )
    return m
