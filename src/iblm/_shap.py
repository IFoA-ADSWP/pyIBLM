"""SHAP extraction – extensible dispatch for fitted booster models."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd


@singledispatch
def extract_booster_shap(booster_model: Any, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Extract SHAP (SHapley Additive exPlanations) values from a fitted booster.

    This is a :func:`functools.singledispatch` generic that dispatches on the
    type of *booster_model*, making it straightforward to support additional
    booster libraries without modifying this module::

        from iblm import extract_booster_shap
        import lightgbm as lgb

        @extract_booster_shap.register(lgb.Booster)
        def _(booster_model, data, **kwargs):
            ...

    ``xgb.Booster`` is supported out of the box.

    Parameters
    ----------
    booster_model:
        A fitted booster object.
    data:
        DataFrame containing the predictor variables.  Any additional columns
        (response, offset, weights, etc.) are silently dropped.
    **kwargs:
        Additional arguments forwarded to the registered implementation.

    Returns
    -------
    pd.DataFrame
        SHAP values with shape ``(n_rows, n_features + 1)``.  Feature columns
        match the booster's feature names; the final column is named
        ``"BIAS"`` and contains the booster's bias term.
    """
    raise NotImplementedError(
        f"extract_booster_shap is not implemented for booster type "
        f"'{type(booster_model).__name__}'. "
        f"Register a new implementation with "
        f"@extract_booster_shap.register({type(booster_model).__name__})."
    )


# ---------------------------------------------------------------------------
# xgb.Booster implementation
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb

    @extract_booster_shap.register(xgb.Booster)
    def _extract_shap_xgb(
        booster_model: xgb.Booster, data: pd.DataFrame, **kwargs: Any
    ) -> pd.DataFrame:
        """Extract SHAP values from an ``xgb.Booster``.

        Parameters
        ----------
        booster_model:
            Trained XGBoost booster (e.g. from :meth:`IBLM.fit`).
        data:
            DataFrame of predictor variables.  Columns not in the booster's
            feature list are dropped automatically.
        **kwargs:
            Unused; kept for API consistency.

        Returns
        -------
        DataFrame with one column per feature plus a ``"BIAS"`` column.
        """
        feature_names: list[str] = booster_model.feature_names

        missing = [f for f in feature_names if f not in data.columns]
        if missing:
            raise ValueError(
                f"Data is missing {len(missing)} feature(s) required by the booster: "
                f"{missing}. Ensure the same columns used during training are present."
            )

        X = data[feature_names].copy()

        dmat = xgb.DMatrix(
            X,
            base_margin=np.zeros(len(X)),
            enable_categorical=True,
        )

        # pred_contribs=True → shape (n_rows, n_features + 1); last col = BIAS
        shap_matrix = booster_model.predict(dmat, pred_contribs=True)

        col_names = feature_names + ["BIAS"]
        shap_df = pd.DataFrame(shap_matrix, columns=col_names, index=data.index)

        return shap_df

except ImportError:
    pass  # xgboost not installed; registration skipped
