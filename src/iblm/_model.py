"""Core IBLM model class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from statsmodels.genmod import families as smf
from statsmodels.genmod.families import links as sml

from ._utils import (
    _assign_variable_type,
    _check_data_variability,
    _check_required_keys,
    _get_cat_info,
    _is_categorical,
)

# ---------------------------------------------------------------------------
# Family helpers
# ---------------------------------------------------------------------------

_SUPPORTED_FAMILIES = ("poisson", "quasipoisson", "gamma", "tweedie", "gaussian")


def _get_glm_family(family: str) -> smf.Family:
    """Return the statsmodels GLM family object for *family*."""
    family = family.lower()
    if family in ("poisson", "quasipoisson"):
        # quasipoisson gives identical point estimates; only SEs differ.
        return smf.Poisson(link=sml.Log())
    if family == "gamma":
        return smf.Gamma(link=sml.Log())
    if family == "tweedie":
        return smf.Tweedie(var_power=1.5, link=sml.Log())
    if family == "gaussian":
        return smf.Gaussian(link=sml.Identity())
    raise ValueError(
        f"family '{family}' not supported. Choose from: {_SUPPORTED_FAMILIES}"
    )


def _get_xgb_objective(family: str) -> dict[str, Any]:
    """Return the default XGBoost param dict for *family*."""
    family = family.lower()
    if family in ("poisson", "quasipoisson"):
        return {"objective": "count:poisson"}
    if family == "gamma":
        return {"objective": "reg:gamma"}
    if family == "tweedie":
        return {"objective": "reg:tweedie", "tweedie_variance_power": 1.5}
    if family == "gaussian":
        return {"objective": "reg:squarederror"}
    raise ValueError(f"family '{family}' not supported.")


def _link_name(glm_result: sm.GLM) -> str:
    """Return the link function name: 'log' or 'identity'."""
    lnk = type(glm_result.family.link).__name__.lower()
    # statsmodels uses 'identity' or 'log' etc.
    if lnk == "log":
        return "log"
    if lnk in ("identity", "identitylink"):
        return "identity"
    return lnk


# ---------------------------------------------------------------------------
# Design-matrix helpers
# ---------------------------------------------------------------------------


def _build_design_matrix(
    data: pd.DataFrame,
    continuous_vars: list[str],
    categorical_vars: list[str],
    cat_levels_all: dict[str, list],
    cat_levels_reference: dict[str, Any],
) -> pd.DataFrame:
    """Build the GLM design matrix (R-style column names, no separator)."""
    n = len(data)
    cols: dict[str, np.ndarray] = {}

    cols["(Intercept)"] = np.ones(n)

    for col in continuous_vars:
        cols[col] = data[col].to_numpy(dtype=float)

    for col in categorical_vars:
        ref = cat_levels_reference[col]
        for lvl in cat_levels_all[col]:
            if lvl != ref:
                cols[f"{col}{lvl}"] = (data[col] == lvl).to_numpy(dtype=float)

    return pd.DataFrame(cols, index=data.index)


def _get_xgb_feature_df(
    data: pd.DataFrame,
    predictor_vars: dict[str, list[str]],
    cat_levels_all: dict[str, list],
    drop_extra: list[str] | None = None,
) -> pd.DataFrame:
    """Return a feature DataFrame suitable for xgb.DMatrix (enable_categorical=True).

    Categorical columns are cast to ``pd.Categorical`` with the training
    category order so XGBoost sees a consistent encoding.
    """
    drop_extra = drop_extra or []
    keep = [c for c in predictor_vars["all"] if c not in drop_extra]
    df = data[keep].copy()

    for col in predictor_vars["categorical"]:
        if col in df.columns:
            df[col] = pd.Categorical(
                df[col], categories=cat_levels_all[col]
            )

    return df


# ---------------------------------------------------------------------------
# GLM fitting helper
# ---------------------------------------------------------------------------


def _fit_glm_robust(glm_kwargs: dict, X_train: pd.DataFrame, y_train: np.ndarray, family: str):
    """Fit a statsmodels GLM with a fallback to robust starting values.

    For Gamma and Tweedie, the default IRLS starting values can overflow when
    the response contains very large values (e.g. ClaimNb / tiny Exposure).
    The fallback initialises from (log(mean(y)), 0, 0, …), which is always safe.
    """
    try:
        return sm.GLM(**glm_kwargs).fit(disp=0)
    except (ValueError, np.linalg.LinAlgError):
        if family.lower() not in ("gamma", "tweedie"):
            raise
        sp = np.zeros(X_train.shape[1])
        sp[0] = np.log(max(float(np.mean(y_train)), 1e-6))
        return sm.GLM(**glm_kwargs).fit(disp=0, start_params=sp)


# ---------------------------------------------------------------------------
# IBLM class
# ---------------------------------------------------------------------------


class IBLM:
    """Interpretable Boosted Linear Model.

    Combines a Generalized Linear Model (GLM) with an XGBoost booster
    trained on the GLM residuals::

        Prediction = GLM × Booster  (log-link families)
        Prediction = GLM + Booster  (identity-link families)

    Typical workflow::

        df_dict = split_into_train_validate_test(df, seed=42)
        model = IBLM()
        model.fit(df_dict, response_var="ClaimNb", offset_var="LogExposure",
                  family="poisson")
        preds = model.predict(df_dict["test"])
    """

    # ------------------------------------------------------------------ #
    # Attributes set by fit()                                              #
    # ------------------------------------------------------------------ #
    glm_model: sm.regression.linear_model.RegressionResultsWrapper | None
    booster_model: xgb.Booster | None
    relationship: str | None          # "multiplicative" | "additive"
    response_var: str | None
    weight_var: str | None
    offset_var: str | None
    predictor_vars: dict | None       # {all, categorical, continuous}
    cat_levels: dict | None           # {all, reference}
    coeff_names: dict | None          # {all, all_cat, reference_cat}
    data: dict | None                 # {train, validate}
    xgb_params: dict | None

    def __init__(self) -> None:
        self.glm_model = None
        self.booster_model = None
        self.relationship = None
        self.response_var = None
        self.weight_var = None
        self.offset_var = None
        self.predictor_vars = None
        self.cat_levels = None
        self.coeff_names = None
        self.data = None
        self.xgb_params = None

    # ------------------------------------------------------------------ #
    # fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        df_dict: dict[str, pd.DataFrame],
        response_var: str,
        weight_var: str | None = None,
        offset_var: str | None = None,
        family: str = "poisson",
        params: dict | None = None,
        nrounds: int = 1000,
        early_stopping_rounds: int = 25,
        verbose: bool | int = 0,
        strip_glm: bool = True,
        **xgb_kwargs: Any,
    ) -> "IBLM":
        """Train the IBLM model.

        Parameters
        ----------
        df_dict:
            Dict with ``"train"`` and ``"validate"`` DataFrames (e.g. from
            :func:`split_into_train_validate_test`).
        response_var:
            Name of the response column.
        weight_var:
            Optional weight column name.
        offset_var:
            Optional offset column (must already be on the link scale, e.g.
            ``log(Exposure)`` for a Poisson model).
        family:
            GLM / XGBoost distribution family. One of ``"poisson"``,
            ``"quasipoisson"``, ``"gamma"``, ``"tweedie"``, ``"gaussian"``.
        params:
            Additional XGBoost parameters (passed to ``xgb.train``).
        nrounds:
            Maximum number of boosting rounds.
        early_stopping_rounds:
            Stop if validation metric does not improve for this many rounds.
        verbose:
            Verbosity for XGBoost training (0 = silent).
        strip_glm:
            If ``True``, remove training data cached inside the GLM result to
            save memory.
        **xgb_kwargs:
            Additional keyword arguments forwarded to ``xgb.train``.

        Returns
        -------
        self
        """
        params = params or {}

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        _check_required_keys(df_dict, ["train", "validate"])
        train_df = df_dict["train"]
        val_df = df_dict["validate"]

        if train_df.isnull().any(axis=None) or val_df.isnull().any(axis=None):
            raise ValueError("df_dict cannot contain NaN values.")

        str_cols = [
            c for c in train_df.columns if train_df[c].dtype == object
        ]
        if str_cols:
            raise ValueError(
                f"df_dict cannot contain string/object columns. "
                f"Convert to pandas Categorical. Offending columns: {str_cols}"
            )

        _check_data_variability(train_df, response_var)

        # ------------------------------------------------------------------
        # Identify predictor variables
        # ------------------------------------------------------------------
        exclude = [c for c in [response_var, weight_var, offset_var] if c]
        feature_cols = [c for c in train_df.columns if c not in exclude]

        categorical_vars = [c for c in feature_cols if _is_categorical(train_df[c])]
        continuous_vars = [c for c in feature_cols if c not in categorical_vars]

        cat_levels_all, cat_levels_reference = _get_cat_info(train_df, categorical_vars)

        # ------------------------------------------------------------------
        # Build GLM design matrices
        # ------------------------------------------------------------------
        X_train = _build_design_matrix(
            train_df, continuous_vars, categorical_vars,
            cat_levels_all, cat_levels_reference,
        )
        X_val = _build_design_matrix(
            val_df, continuous_vars, categorical_vars,
            cat_levels_all, cat_levels_reference,
        )

        y_train = train_df[response_var].to_numpy(dtype=float)
        y_val = val_df[response_var].to_numpy(dtype=float)

        offset_train = (
            train_df[offset_var].to_numpy(dtype=float) if offset_var else None
        )
        offset_val = (
            val_df[offset_var].to_numpy(dtype=float) if offset_var else None
        )

        weights_train = (
            train_df[weight_var].to_numpy(dtype=float) if weight_var else None
        )

        # ------------------------------------------------------------------
        # Offset warning: check if offset looks like it hasn't been log-transformed
        # ------------------------------------------------------------------
        if offset_var and family not in ("gaussian",):
            _warn_offset_scale(train_df[offset_var].to_numpy(dtype=float), offset_var)

        # ------------------------------------------------------------------
        # Fit GLM
        # ------------------------------------------------------------------
        glm_family = _get_glm_family(family)

        glm_kwargs: dict[str, Any] = dict(
            endog=y_train,
            exog=X_train,
            family=glm_family,
        )
        if offset_train is not None:
            glm_kwargs["offset"] = offset_train
        if weights_train is not None:
            glm_kwargs["var_weights"] = weights_train

        glm_result = _fit_glm_robust(glm_kwargs, X_train, y_train, family)

        link = _link_name(glm_result)
        if link == "log":
            relationship = "multiplicative"
        elif link == "identity":
            relationship = "additive"
        else:
            raise ValueError(f"Unsupported link function '{link}'.")

        # ------------------------------------------------------------------
        # GLM predictions on link scale → XGBoost base_margin
        # ------------------------------------------------------------------
        glm_link_train = glm_result.predict(
            exog=X_train,
            offset=offset_train if offset_train is not None else np.zeros(len(X_train)),
            which="linear",
        )
        glm_link_val = glm_result.predict(
            exog=X_val,
            offset=offset_val if offset_val is not None else np.zeros(len(X_val)),
            which="linear",
        )

        # ------------------------------------------------------------------
        # Build XGBoost DMatrices
        # ------------------------------------------------------------------
        X_xgb_train = _get_xgb_feature_df(
            train_df, {"all": feature_cols, "categorical": categorical_vars},
            cat_levels_all,
        )
        X_xgb_val = _get_xgb_feature_df(
            val_df, {"all": feature_cols, "categorical": categorical_vars},
            cat_levels_all,
        )

        dtrain = xgb.DMatrix(
            X_xgb_train, label=y_train,
            weight=weights_train,
            enable_categorical=True,
        )
        dtrain.set_base_margin(glm_link_train)

        dval = xgb.DMatrix(
            X_xgb_val, label=y_val,
            enable_categorical=True,
        )
        dval.set_base_margin(glm_link_val)

        # ------------------------------------------------------------------
        # Configure XGBoost params
        # ------------------------------------------------------------------
        xgb_default = _get_xgb_objective(family)
        # User-supplied params override defaults
        for k, v in params.items():
            if k in xgb_default:
                import warnings
                warnings.warn(
                    f"XGBoost param '{k}' overrides default for family '{family}'.",
                    stacklevel=2,
                )
        final_params = {**xgb_default, **params}

        # ------------------------------------------------------------------
        # Train XGBoost
        # ------------------------------------------------------------------
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(
                xgb.callback.EarlyStopping(rounds=early_stopping_rounds)
            )

        booster = xgb.train(
            params=final_params,
            dtrain=dtrain,
            num_boost_round=nrounds,
            evals=[(dval, "validation")],
            verbose_eval=verbose,
            callbacks=callbacks,
            **xgb_kwargs,
        )

        # ------------------------------------------------------------------
        # Build metadata (coeff_names, cat_levels, predictor_vars)
        # ------------------------------------------------------------------
        glm_coef_names = list(glm_result.params.index)

        all_cat_coeff_names = []
        for col in categorical_vars:
            all_cat_coeff_names.extend(
                f"{col}{lvl}" for lvl in cat_levels_all[col]
            )

        coeff_names_all = ["(Intercept)"] + continuous_vars + all_cat_coeff_names
        reference_cat = [
            name for name in coeff_names_all if name not in glm_coef_names
            and name != "(Intercept)"
        ]

        predictor_vars = {
            "all": feature_cols,
            "categorical": categorical_vars,
            "continuous": continuous_vars,
        }
        cat_levels = {
            "all": cat_levels_all,
            "reference": cat_levels_reference,
        }
        coeff_names = {
            "all": coeff_names_all,
            "all_cat": all_cat_coeff_names,
            "reference_cat": reference_cat,
        }

        # ------------------------------------------------------------------
        # Optionally strip GLM training data from result object
        # ------------------------------------------------------------------
        if strip_glm:
            _strip_glm(glm_result)

        # ------------------------------------------------------------------
        # Store XGBoost params (without DMatrix data) for train_xgb_as_per_iblm
        # ------------------------------------------------------------------
        stored_xgb_params = {
            "params": final_params,
            "num_boost_round": nrounds,
            **xgb_kwargs,
        }
        if early_stopping_rounds:
            stored_xgb_params["early_stopping_rounds"] = early_stopping_rounds

        # ------------------------------------------------------------------
        # Assign to self
        # ------------------------------------------------------------------
        self.glm_model = glm_result
        self.booster_model = booster
        self.relationship = relationship
        self.response_var = response_var
        self.weight_var = weight_var
        self.offset_var = offset_var
        self.predictor_vars = predictor_vars
        self.cat_levels = cat_levels
        self.coeff_names = coeff_names
        self.data = {"train": train_df.copy(), "validate": val_df.copy()}
        self.xgb_params = stored_xgb_params

        return self

    # ------------------------------------------------------------------ #
    # predict                                                              #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        newdata: pd.DataFrame,
        trim: float | None = None,
        type: str = "response",
    ) -> np.ndarray:
        """Generate predictions from the fitted IBLM model.

        Parameters
        ----------
        newdata:
            DataFrame with the same structure as training data.
        trim:
            Post-hoc trimming of XGBoost predictions.  If ``None`` (default)
            no trimming is applied.  For example, ``trim=0.2`` constrains
            booster predictions to ``[0.8, 1.2]`` before re-normalising.
        type:
            ``"response"`` (default) or ``"link"``.

        Returns
        -------
        numpy.ndarray of predictions.
        """
        if self.glm_model is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")
        if type not in ("response", "link"):
            raise ValueError("type must be 'response' or 'link'.")

        offset_var = self.offset_var

        # Prepare offset for GLM prediction
        if offset_var and offset_var in newdata.columns:
            new_offset = newdata[offset_var].to_numpy(dtype=float)
        else:
            if offset_var:
                import warnings
                warnings.warn(
                    f"Model was fitted with offset '{offset_var}' but it was not "
                    "found in newdata. Offset assumed to be zero.",
                    stacklevel=2,
                )
            new_offset = np.zeros(len(newdata))

        # Build GLM design matrix
        X_new = _build_design_matrix(
            newdata,
            self.predictor_vars["continuous"],
            self.predictor_vars["categorical"],
            self.cat_levels["all"],
            self.cat_levels["reference"],
        )

        # GLM prediction
        glm_pred = self.glm_model.predict(
            exog=X_new,
            offset=new_offset,
            which="linear" if type == "link" else "mean",
        )
        glm_pred = np.asarray(glm_pred)

        # XGBoost feature matrix (no offset column)
        X_xgb = _get_xgb_feature_df(
            newdata,
            self.predictor_vars,
            self.cat_levels["all"],
        )
        dtest = xgb.DMatrix(
            X_xgb,
            base_margin=np.zeros(len(newdata)),
            enable_categorical=True,
        )
        booster_pred = self.booster_model.predict(dtest)

        # Optional trimming (multiplicative only)
        if trim is not None:
            booster_pred = np.clip(booster_pred, max(1 - trim, 0.0), 1 + trim)
            booster_pred = booster_pred / booster_pred.mean()

        # Combine
        if self.relationship == "multiplicative" and type == "response":
            return glm_pred * booster_pred
        if self.relationship == "additive":
            return glm_pred + booster_pred
        if self.relationship == "multiplicative" and type == "link":
            return glm_pred + np.log(booster_pred)

        raise ValueError(
            f"Cannot combine relationship='{self.relationship}' with type='{type}'."
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        fitted = self.relationship is not None
        if fitted:
            n_cat = len(self.predictor_vars["categorical"])
            n_cont = len(self.predictor_vars["continuous"])
            return (
                f"IBLM(fitted=True, relationship='{self.relationship}', "
                f"response='{self.response_var}', "
                f"n_continuous={n_cont}, n_categorical={n_cat})"
            )
        return "IBLM(fitted=False)"


# ---------------------------------------------------------------------------
# Standalone function: train_xgb_as_per_iblm
# ---------------------------------------------------------------------------


def train_xgb_as_per_iblm(iblm_model: IBLM, **xgb_kwargs: Any) -> xgb.Booster:
    """Retrain a standalone XGBoost model using the IBLM's stored parameters.

    This is useful for direct comparison: the XGBoost model is trained on the
    raw response (with offset as base_margin if applicable), not on GLM
    residuals.

    Parameters
    ----------
    iblm_model:
        A fitted :class:`IBLM` instance.
    **xgb_kwargs:
        Override any stored XGBoost parameters.

    Returns
    -------
    Trained ``xgb.Booster``.
    """
    if iblm_model.glm_model is None:
        raise RuntimeError("iblm_model has not been fitted.")
    if not isinstance(iblm_model.booster_model, xgb.Booster):
        raise TypeError("booster_model must be an xgb.Booster.")

    train_df = iblm_model.data["train"]
    val_df = iblm_model.data["validate"]
    response_var = iblm_model.response_var
    weight_var = iblm_model.weight_var
    offset_var = iblm_model.offset_var

    y_train = train_df[response_var].to_numpy(dtype=float)
    y_val = val_df[response_var].to_numpy(dtype=float)

    weights_train = (
        train_df[weight_var].to_numpy(dtype=float) if weight_var else None
    )

    X_xgb_train = _get_xgb_feature_df(
        train_df, iblm_model.predictor_vars, iblm_model.cat_levels["all"]
    )
    X_xgb_val = _get_xgb_feature_df(
        val_df, iblm_model.predictor_vars, iblm_model.cat_levels["all"]
    )

    dtrain = xgb.DMatrix(
        X_xgb_train, label=y_train, weight=weights_train, enable_categorical=True
    )
    dval = xgb.DMatrix(X_xgb_val, label=y_val, enable_categorical=True)

    if offset_var and offset_var in train_df.columns:
        dtrain.set_base_margin(train_df[offset_var].to_numpy(dtype=float))
        dval.set_base_margin(val_df[offset_var].to_numpy(dtype=float))

    stored = {**iblm_model.xgb_params, **xgb_kwargs}
    params = stored.pop("params", {})
    num_boost_round = stored.pop("num_boost_round", 1000)
    early_stopping_rounds = stored.pop("early_stopping_rounds", None)

    callbacks = []
    if early_stopping_rounds:
        callbacks.append(xgb.callback.EarlyStopping(rounds=early_stopping_rounds))

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        callbacks=callbacks,
        **stored,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _strip_glm(glm_result: Any) -> None:
    """Remove cached training data from a statsmodels GLM result."""
    try:
        glm_result.model.data.endog = None
        glm_result.model.data.exog = None
        glm_result.model.data.weights = None
    except AttributeError:
        pass


def _warn_offset_scale(offset: np.ndarray, offset_var: str) -> None:
    """Warn if the offset looks like it has not been log-transformed."""
    import warnings
    if len(offset) == 0:
        return
    # Heuristic 1: values span many orders of magnitude → probably raw counts
    pos = offset[offset > 0]
    if len(pos) > 1 and pos.max() / pos.min() > 1000:
        warnings.warn(
            f"'{offset_var}' spans several orders of magnitude. "
            "For log-link models the offset must be on the log scale. "
            "Did you forget to apply log()?",
            stacklevel=3,
        )
    # Heuristic 2: majority of values in (0, 1] → probably raw proportions
    elif np.mean((offset > 0) & (offset <= 1)) > 0.5:
        warnings.warn(
            f"'{offset_var}' has a range of [0, 1]. "
            "For log-link models the offset must be on the log scale. "
            "Did you forget to apply log()?",
            stacklevel=3,
        )
