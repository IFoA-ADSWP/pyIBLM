"""Tests for input data validation."""

import pandas as pd
import pytest

from iblm import IBLM, load_freMTPLmini, split_into_train_validate_test
from iblm._utils import _check_data_variability

SEED = 42


def test_error_on_constant_field():
    """Fitting must raise when any predictor has only one unique value."""
    df = load_freMTPLmini().copy()
    df["DrivAge"] = 50  # make constant
    df["Area"] = df["Area"].cat.add_categories("A_only")
    df["Area"] = "A_only"
    df["Area"] = df["Area"].astype("category")

    splits = split_into_train_validate_test(df, seed=SEED)
    model = IBLM()
    with pytest.raises((ValueError, Exception)):
        model.fit(splits, response_var="ClaimNb", family="poisson")


def test_error_on_string_column():
    """Fitting must raise when data contains raw string (object) columns."""
    df = load_freMTPLmini().copy()
    # Convert a categorical back to plain string
    df["Area"] = df["Area"].astype(str)

    splits = split_into_train_validate_test(df, seed=SEED)
    model = IBLM()
    with pytest.raises((ValueError, TypeError, Exception)):
        model.fit(splits, response_var="ClaimNb", family="poisson")


def test_error_on_nan():
    """Fitting must raise when data contains NaN values."""
    df = load_freMTPLmini().copy()
    df.loc[0, "DrivAge"] = float("nan")

    splits = split_into_train_validate_test(df, seed=SEED)
    model = IBLM()
    with pytest.raises((ValueError, Exception)):
        model.fit(splits, response_var="ClaimNb", family="poisson")


def test_check_data_variability_raises():
    """_check_data_variability should raise for a constant column."""
    df = load_freMTPLmini().copy()
    df["DrivAge"] = 50
    with pytest.raises(ValueError):
        _check_data_variability(df, "ClaimNb")
