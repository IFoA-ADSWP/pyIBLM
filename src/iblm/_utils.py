"""Internal utility helpers shared across the IBLM package."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------


def split_into_train_validate_test(
    df: pd.DataFrame,
    train_prop: float = 0.7,
    validate_prop: float = 0.15,
    test_prop: float = 0.15,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Randomly split *df* into train, validate, and test subsets.

    Each row is assigned independently to one of the three subsets with the
    probabilities defined by *train_prop*, *validate_prop*, and *test_prop*.
    Because assignment is row-independent, the realised proportions may
    differ slightly from the requested values; this effect is negligible for
    the large datasets typical of IBLM workflows.

    Parameters
    ----------
    df:
        Input DataFrame to split.
    train_prop:
        Proportion of rows allocated to the training subset.
    validate_prop:
        Proportion of rows allocated to the validation subset (used for
        XGBoost early stopping during :meth:`~iblm.IBLM.fit`).
    test_prop:
        Proportion of rows allocated to the held-out test subset.
    seed:
        Optional integer random seed for reproducibility.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dict with keys ``"train"``, ``"validate"``, and ``"test"``, each
        containing the corresponding subset as a reset-index DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not math.isclose(train_prop + validate_prop + test_prop, 1.0, abs_tol=1e-9):
        raise ValueError("train_prop + validate_prop + test_prop must equal 1.")

    rng = np.random.default_rng(seed)
    labels = rng.choice(
        ["train", "validate", "test"],
        size=len(df),
        replace=True,
        p=[train_prop, validate_prop, test_prop],
    )

    return {
        "train": df.iloc[labels == "train"].reset_index(drop=True),
        "validate": df.iloc[labels == "validate"].reset_index(drop=True),
        "test": df.iloc[labels == "test"].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Internal helpers (prefixed with _)
# ---------------------------------------------------------------------------


def _check_required_keys(d: dict, required_keys: list[str], label: str = "input") -> None:
    """Raise a ``KeyError`` if any *required_keys* are absent from *d*."""
    missing = [k for k in required_keys if k not in d]
    if missing:
        raise KeyError(f"{label} is missing required keys: {missing}")


def _assign_variable_type(
    var: str,
    vars_continuous: list[str],
    vars_categorical: list[str],
) -> str:
    """Return ``'numerical'`` or ``'categorical'`` for *var*."""
    if var in vars_continuous:
        return "numerical"
    if var in vars_categorical:
        return "categorical"
    valid = sorted(set(vars_continuous) | set(vars_categorical))
    raise ValueError(
        f"'{var}' is not a recognised predictor variable. "
        f"Valid variables: {valid}"
    )


def _detect_outliers(x: np.ndarray | pd.Series, q: float = 0.01) -> np.ndarray:
    """Return a boolean mask where ``True`` means *keep* (not an outlier).

    Parameters
    ----------
    x:
        Numeric array.
    q:
        Quantile threshold; must satisfy ``0 < q < 0.5``.
    """
    x = np.asarray(x, dtype=float)
    if q <= 0 or q >= 0.5:
        raise ValueError("q must be between 0 (exclusive) and 0.5 (exclusive).")
    lower = np.nanquantile(x, q)
    upper = np.nanquantile(x, 1 - q)
    return (x >= lower) & (x <= upper)


def _check_data_variability(data: pd.DataFrame, response_var: str) -> None:
    """Raise if the response or any predictor has only one unique value."""
    if data[response_var].nunique() <= 1:
        raise ValueError(
            f"Response variable '{response_var}' must have more than one unique value."
        )
    unvaried = [c for c in data.columns if data[c].nunique() <= 1]
    if unvaried:
        raise ValueError(
            f"All predictor variables must have more than one unique value. "
            f"Constant columns: {unvaried}"
        )


def _is_categorical(series: pd.Series) -> bool:
    """Return True if *series* should be treated as categorical."""
    return isinstance(series.dtype, pd.CategoricalDtype)


def _get_cat_info(
    df: pd.DataFrame,
    categorical_vars: list[str],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Return (cat_levels_all, cat_levels_reference) dicts.

    Levels are sorted; the first sorted level is the reference, matching
    R's default ``contr.treatment`` behaviour.
    """
    cat_levels_all: dict[str, list] = {}
    cat_levels_reference: dict[str, Any] = {}

    for col in categorical_vars:
        s = df[col]
        if hasattr(s, "cat"):
            levels = list(s.cat.categories)
        else:
            levels = sorted(s.unique().tolist())
        cat_levels_all[col] = levels
        cat_levels_reference[col] = levels[0]

    return cat_levels_all, cat_levels_reference
