"""ExplainIBLM class and associated data-transformation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._shap import extract_booster_shap

if TYPE_CHECKING:
    from ._model import IBLM
    import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Public API: explain_iblm convenience function
# ---------------------------------------------------------------------------


def explain_iblm(
    iblm_model: "IBLM",
    data: pd.DataFrame,
    migrate_reference_to_bias: bool = True,
) -> "ExplainIBLM":
    """Create an :class:`ExplainIBLM` explainer for *iblm_model*.

    Convenience constructor equivalent to
    ``ExplainIBLM(iblm_model, data, migrate_reference_to_bias)``.

    Parameters
    ----------
    iblm_model:
        A fitted :class:`~iblm.IBLM` model.
    data:
        DataFrame to explain.  Typically the test split returned by
        :func:`~iblm.split_into_train_validate_test`.
    migrate_reference_to_bias:
        If ``True`` (default), the SHAP contributions of categorical reference
        levels are absorbed into the bias term rather than being shown as
        per-variable corrections.  It is recommended to leave this as ``True``.

    Returns
    -------
    :class:`ExplainIBLM`
    """
    return ExplainIBLM(iblm_model, data, migrate_reference_to_bias)


# ---------------------------------------------------------------------------
# ExplainIBLM class
# ---------------------------------------------------------------------------


class ExplainIBLM:
    """SHAP-based explainer for a fitted :class:`~iblm.IBLM` model.

    Explains the beta values of the IBLM ensemble and the booster corrections
    applied to them, using SHAP (SHapley Additive exPlanations) values
    extracted from the XGBoost component.

    The key data attributes (:attr:`shap`, :attr:`beta_corrections`,
    :attr:`data_beta_coeff`) are computed once at construction time.  The
    plotting methods (:meth:`beta_corrected_scatter`, etc.) are bound to those
    results, so only display settings (variable names, colours, quantile
    bounds) need to be supplied when calling them.

    Attributes
    ----------
    shap : pd.DataFrame
        Raw SHAP values extracted from the XGBoost booster, one row per
        observation and one column per feature, plus a ``"BIAS"`` column.
    beta_corrections : pd.DataFrame
        Per-row booster beta corrections in wide (one-hot) format, with a
        leading ``"bias"`` column.
    data_beta_coeff : pd.DataFrame
        Combined GLM + booster beta coefficients, one row per observation
        and one column per predictor plus a leading ``"bias"`` column.

    Parameters
    ----------
    iblm_model:
        A fitted :class:`~iblm.IBLM` model.
    data:
        DataFrame to explain.  Typically the test split returned by
        :func:`~iblm.split_into_train_validate_test`.
    migrate_reference_to_bias:
        If ``True`` (default), the SHAP contributions of categorical reference
        levels are absorbed into the bias term rather than being shown as
        per-variable corrections.  It is recommended to leave this as ``True``.
    """

    def __init__(
        self,
        iblm_model: "IBLM",
        data: pd.DataFrame,
        migrate_reference_to_bias: bool = True,
    ) -> None:
        _check_iblm_fitted(iblm_model)

        # Raw SHAP values
        shap = extract_booster_shap(iblm_model.booster_model, data)

        # Wide one-hot versions
        wide_input_frame = data_to_onehot(data, iblm_model)
        shap_wide = shap_to_onehot(shap, wide_input_frame, iblm_model)
        beta_corrections = beta_corrections_derive(
            shap_wide, wide_input_frame, iblm_model, migrate_reference_to_bias
        )

        # Beta coefficients (GLM + booster)
        data_glm = data_beta_coeff_glm(data, iblm_model)
        data_booster = data_beta_coeff_booster(data, beta_corrections, iblm_model)
        data_beta_coeff = data_glm + data_booster

        self.shap = shap
        self.beta_corrections = beta_corrections
        self.data_beta_coeff = data_beta_coeff

        # Bind plotting methods (closures capture the computed data)
        self._iblm_model = iblm_model
        self._data = data
        self._wide_input_frame = wide_input_frame
        self._migrate_reference_to_bias = migrate_reference_to_bias

    # ------------------------------------------------------------------ #
    # Plotting methods                                                     #
    # ------------------------------------------------------------------ #

    def beta_corrected_scatter(
        self,
        varname: str,
        q: float = 0,
        color: str | None = None,
    ) -> "plt.Figure":
        """Scatter plot or boxplot of beta coefficients after SHAP corrections.

        For **continuous** variables, produces a scatter plot of variable
        values vs. corrected beta coefficients, with a LOWESS smooth overlaid
        and horizontal lines showing the GLM coefficient (solid) and ±1
        standard error (dashed).

        For **categorical** variables, produces a boxplot of corrected beta
        coefficients for each non-reference level, with the GLM coefficient
        overlaid as a point.

        Parameters
        ----------
        varname:
            Name of the predictor variable to plot.  Must be present in the
            fitted model.
        q:
            Quantile threshold for outlier removal in continuous plots.
            ``0`` (default) retains all points.
        color:
            Optional name of a variable in the data to colour scatter points
            by.  Supported for continuous variables only.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._plots import _beta_corrected_scatter_plot

        return _beta_corrected_scatter_plot(
            varname=varname,
            q=q,
            color=color,
            data_beta_coeff=self.data_beta_coeff,
            data=self._data,
            iblm_model=self._iblm_model,
        )

    def beta_corrected_density(
        self,
        varname: str,
        q: float = 0.05,
        type: str = "kde",
    ) -> "plt.Figure | dict[str, plt.Figure]":
        """Density plot of corrected beta coefficients for a variable.

        Displays the distribution of SHAP-corrected beta values alongside the
        original GLM coefficient (solid vertical line) and ±1 standard error
        bounds (dashed vertical lines).

        Parameters
        ----------
        varname:
            Predictor name **or** a specific coefficient name (e.g.
            ``"AreaB"``).  Passing a categorical variable name returns one
            plot per non-reference level.
        q:
            Quantile bound for x-axis trimming; must satisfy ``0 <= q < 0.5``.
            A value of ``0.05`` shows only the central 90 % of the
            distribution.  Set to ``0`` for no trimming.
        type:
            Plot style: ``"kde"`` (kernel density estimate, default) or
            ``"hist"`` (histogram).

        Returns
        -------
        matplotlib.figure.Figure or dict[str, matplotlib.figure.Figure]
            A single ``Figure`` for a continuous variable or a specific
            coefficient level; a dict of ``Figure`` objects keyed by
            coefficient name for a categorical variable.
        """
        from ._plots import _beta_corrected_density_plot

        return _beta_corrected_density_plot(
            varname=varname,
            q=q,
            plot_type=type,
            wide_input_frame=self._wide_input_frame,
            beta_corrections=self.beta_corrections,
            data=self._data,
            iblm_model=self._iblm_model,
        )

    def bias_density(
        self,
        q: float = 0,
        type: str = "hist",
    ) -> "dict[str, plt.Figure | None]":
        """Density plots for SHAP corrections that have been migrated to bias.

        Visualises the distribution of booster corrections that are absorbed
        into the bias term: for continuous variables, rows where the feature
        value is zero; for categorical variables (when
        ``migrate_reference_to_bias=True``), rows at the reference level.
        Variables with no records contributing to bias are omitted.

        Parameters
        ----------
        q:
            Quantile bound for x-axis trimming; must satisfy ``0 <= q < 0.5``.
            Set to ``0`` (default) for no trimming.
        type:
            Plot style: ``"kde"`` (kernel density estimate) or ``"hist"``
            (histogram, default).

        Returns
        -------
        dict[str, matplotlib.figure.Figure or None]
            A dict with two keys:

            * ``"bias_correction_var"`` – faceted plot showing the bias
              correction distribution separately for each contributing variable.
            * ``"bias_correction_total"`` – plot of the total corrected bias
              distribution across all observations.

            Either value may be ``None`` if no bias migration occurred.
        """
        from ._plots import _bias_density_plot

        return _bias_density_plot(
            q=q,
            plot_type=type,
            migrate_reference_to_bias=self._migrate_reference_to_bias,
            shap=self.shap,
            data=self._data,
            iblm_model=self._iblm_model,
        )

    def overall_correction(
        self,
        transform_x_scale_by_link: bool = True,
    ) -> "plt.Figure":
        """Distribution of the booster's overall correction for each observation.

        Shows the total booster component (multiplicative factor for log-link
        models; additive adjustment for identity-link models) across all
        observations, as a kernel density plot.

        Parameters
        ----------
        transform_x_scale_by_link:
            If ``True`` (default), the x-axis is transformed by the inverse
            link function (e.g. exponentiated for log-link models so the axis
            shows multiplicative factors rather than log-scale values).

        Returns
        -------
        matplotlib.figure.Figure
        """
        from ._plots import _overall_correction_plot

        return _overall_correction_plot(
            transform_x_scale_by_link=transform_x_scale_by_link,
            shap=self.shap,
            iblm_model=self._iblm_model,
        )

    def __repr__(self) -> str:
        return (
            f"ExplainIBLM("
            f"n_rows={len(self.shap)}, "
            f"n_features={len(self.shap.columns) - 1})"
        )


# ---------------------------------------------------------------------------
# Data helpers (also exported for advanced use)
# ---------------------------------------------------------------------------


def data_to_onehot(
    data: pd.DataFrame,
    iblm_model: "IBLM",
    remove_target: bool = True,
) -> pd.DataFrame:
    """Convert *data* to wide one-hot format aligned with *iblm_model*.

    All categorical variables are expanded to one binary column per level,
    including the reference level (which will be all-zeros for any row at
    that level).  Continuous variables are preserved as numeric.  A leading
    ``(Intercept)`` column of ones is prepended.

    Parameters
    ----------
    data:
        Input DataFrame.  Typically the test split returned by
        :func:`~iblm.split_into_train_validate_test`.
    iblm_model:
        Fitted :class:`~iblm.IBLM` model used to determine the expected
        columns and their ordering.
    remove_target:
        If ``True`` (default), drop the response variable column if present
        in *data*.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns ordered as in
        ``iblm_model.coeff_names["all"]``.
    """
    _check_iblm_fitted(iblm_model)

    cat_vars = iblm_model.predictor_vars["categorical"]
    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_levels_all = iblm_model.cat_levels["all"]
    coeff_names_all = iblm_model.coeff_names["all"]
    response_var = iblm_model.response_var

    n = len(data)
    result: dict[str, Any] = {}

    result["(Intercept)"] = np.ones(n)

    for col in cont_vars:
        result[col] = data[col].to_numpy(dtype=float)

    # One-hot encode ALL levels (including reference) → reference col = 0 unless
    # that row has the reference level
    for col in cat_vars:
        for lvl in cat_levels_all[col]:
            result[f"{col}{lvl}"] = (data[col] == lvl).to_numpy(dtype=float)

    # Build DataFrame and reorder to match coeff_names_all
    # (some columns may not yet exist if a training level is absent in data →
    #  they will be all-zero, already handled since we iterate cat_levels_all)
    frame = pd.DataFrame(result, index=data.index)

    # Ensure all expected columns exist (fill missing with 0)
    for col in coeff_names_all:
        if col not in frame.columns:
            frame[col] = 0.0

    frame = frame[coeff_names_all]

    if remove_target and response_var in frame.columns:
        frame = frame.drop(columns=[response_var])

    return frame


def shap_to_onehot(
    shap: pd.DataFrame,
    wide_input_frame: pd.DataFrame,
    iblm_model: "IBLM",
) -> pd.DataFrame:
    """Distribute categorical SHAP values across their one-hot columns.

    XGBoost produces a single SHAP value per categorical variable per row.
    This function distributes that value onto the active level's one-hot
    column (multiplying by the one-hot mask), so that each level has its own
    SHAP column and inactive levels are zero.

    Parameters
    ----------
    shap:
        Raw SHAP DataFrame from :func:`~iblm.extract_booster_shap`.
    wide_input_frame:
        One-hot encoded input data from :func:`data_to_onehot`.
    iblm_model:
        Fitted :class:`~iblm.IBLM` model.

    Returns
    -------
    pd.DataFrame
        Wide-format SHAP DataFrame with one column per one-hot level and a
        leading ``"bias"`` column (renamed from ``"BIAS"``).
    """
    _check_iblm_fitted(iblm_model)

    cat_vars = iblm_model.predictor_vars["categorical"]
    cat_levels_all = iblm_model.cat_levels["all"]
    response_var = iblm_model.response_var

    if not cat_vars:
        shap_wide = shap.copy()
        shap_wide.insert(0, "bias", shap["BIAS"].values)
        return shap_wide

    # Columns in wide_input_frame that we need as the mask
    wif = wide_input_frame.drop(
        columns=[c for c in ["(Intercept)", response_var] if c in wide_input_frame.columns],
        errors="ignore",
    )

    # Build one-hot distributed SHAP frames for each categorical
    cat_blocks: list[pd.DataFrame] = []
    for col in cat_vars:
        levels = cat_levels_all[col]
        onehot_cols = [f"{col}{lvl}" for lvl in levels]
        mask = wif[onehot_cols].to_numpy(dtype=float)                  # (n, n_levels)
        shap_vals = shap[col].to_numpy(dtype=float)[:, np.newaxis]     # (n, 1)
        distributed = shap_vals * mask                                  # (n, n_levels)
        cat_blocks.append(
            pd.DataFrame(distributed, columns=onehot_cols, index=shap.index)
        )

    cat_frame = pd.concat(cat_blocks, axis=1)

    # Drop original categorical columns from shap, add the distributed ones
    shap_wide = pd.concat(
        [shap.drop(columns=[c for c in cat_vars if c in shap.columns]), cat_frame],
        axis=1,
    )

    # Reorder to match wide_input_frame columns
    target_cols = [c for c in wif.columns if c in shap_wide.columns]
    shap_wide = shap_wide[target_cols]

    # Prepend "bias" column (= XGBoost BIAS)
    shap_wide.insert(0, "bias", shap["BIAS"].values)

    return shap_wide


def beta_corrections_derive(
    shap_wide: pd.DataFrame,
    wide_input_frame: pd.DataFrame,
    iblm_model: "IBLM",
    migrate_reference_to_bias: bool = True,
) -> pd.DataFrame:
    """Compute per-row booster beta corrections from wide-format SHAP values.

    Processes SHAP values to produce beta coefficient corrections.  Three
    transformations are applied:

    1. **Continuous variables** – each SHAP value is divided by the
       corresponding feature value to express the correction as a beta
       addend: ``beta_correction = SHAP / feature_value``.
    2. **Continuous variables at zero** – rows where the feature value is
       zero cannot be scaled; their SHAP contributions are migrated into
       the bias column instead.
    3. **Categorical reference levels** – if *migrate_reference_to_bias* is
       ``True``, SHAP contributions for rows at the reference level of each
       categorical variable are absorbed into the bias column.

    Parameters
    ----------
    shap_wide:
        Wide-format SHAP DataFrame from :func:`shap_to_onehot`.
    wide_input_frame:
        One-hot encoded input data from :func:`data_to_onehot`.
    iblm_model:
        Fitted :class:`~iblm.IBLM` model.
    migrate_reference_to_bias:
        If ``True`` (default), absorb reference-level SHAP contributions
        into the bias column.  It is recommended to leave this as ``True``.

    Returns
    -------
    pd.DataFrame
        Beta corrections in one-hot (wide) format, with a leading
        ``"bias"`` column.
    """
    _check_iblm_fitted(iblm_model)

    cont_vars = iblm_model.predictor_vars["continuous"]
    ref_cat_cols = iblm_model.coeff_names["reference_cat"]

    beta_corrections = shap_wide.copy()

    # ---- Continuous: migrate SHAP for zero-valued rows to bias ----
    if cont_vars:
        cont_wif = wide_input_frame[cont_vars].to_numpy(dtype=float)
        is_zero = (cont_wif == 0).astype(float)
        shap_cont = shap_wide[cont_vars].to_numpy(dtype=float)
        shap_for_zeros = (is_zero * shap_cont).sum(axis=1)
    else:
        shap_for_zeros = np.zeros(len(beta_corrections))

    # ---- Categorical reference levels: migrate to bias ----
    if migrate_reference_to_bias and ref_cat_cols:
        ref_cols_present = [c for c in ref_cat_cols if c in shap_wide.columns]
        shap_for_cat_ref = shap_wide[ref_cols_present].to_numpy(dtype=float).sum(axis=1)
        beta_corrections[ref_cols_present] = 0.0
    else:
        shap_for_cat_ref = np.zeros(len(beta_corrections))

    beta_corrections["bias"] = (
        beta_corrections["bias"].to_numpy()
        + shap_for_zeros
        + shap_for_cat_ref
    )

    # ---- Scale continuous SHAP by feature value ----
    for col in cont_vars:
        feat_vals = wide_input_frame[col].to_numpy(dtype=float)
        shap_vals = beta_corrections[col].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(feat_vals != 0, shap_vals / feat_vals, 0.0)
        beta_corrections[col] = scaled

    return beta_corrections


def data_beta_coeff_glm(
    data: pd.DataFrame,
    iblm_model: "IBLM",
) -> pd.DataFrame:
    """Return a DataFrame of GLM beta coefficients expanded row-wise.

    Creates a tabular view where each row contains the GLM coefficients that
    apply to that observation:

    * **Continuous variables** – each column contains the constant GLM
      coefficient (the same value for every row).
    * **Categorical variables** – each column contains the GLM coefficient
      for the level observed in that row (``0`` for the reference level).
    * **bias** – the intercept coefficient (constant across all rows).

    Parameters
    ----------
    data:
        Predictor DataFrame.  Response, weight, and offset columns are
        silently ignored.
    iblm_model:
        Fitted :class:`~iblm.IBLM` model.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["bias", *predictor_vars["all"]]``, one
        row per observation.
    """
    _check_iblm_fitted(iblm_model)

    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_vars = iblm_model.predictor_vars["categorical"]
    cat_levels_all = iblm_model.cat_levels["all"]
    cat_levels_ref = iblm_model.cat_levels["reference"]
    params = iblm_model.glm_model.params  # pd.Series of GLM coefficients
    exclude = [iblm_model.response_var, iblm_model.weight_var, iblm_model.offset_var]
    feature_cols = iblm_model.predictor_vars["all"]

    n = len(data)
    result: dict[str, np.ndarray] = {}

    # Bias = intercept
    result["bias"] = np.full(n, params.get("(Intercept)", 0.0))

    # Continuous: constant beta per variable
    for col in cont_vars:
        result[col] = np.full(n, params.get(col, 0.0))

    # Categorical: look up coefficient for each row's level
    for col in cat_vars:
        ref = cat_levels_ref[col]
        level_coeffs = {}
        for lvl in cat_levels_all[col]:
            coeff_name = f"{col}{lvl}"
            level_coeffs[lvl] = params.get(coeff_name, 0.0) if lvl != ref else 0.0
        result[col] = data[col].map(level_coeffs).to_numpy(dtype=float)

    # Build in the same column order as the predictor_vars
    cols = ["bias"] + feature_cols
    return pd.DataFrame({c: result[c] for c in cols if c in result}, index=data.index)


def data_beta_coeff_booster(
    data: pd.DataFrame,
    beta_corrections: pd.DataFrame,
    iblm_model: "IBLM",
) -> pd.DataFrame:
    """Return a DataFrame of booster beta corrections expanded row-wise.

    Mirrors the structure of :func:`data_beta_coeff_glm`: one column per
    predictor plus a leading ``"bias"`` column.  For categorical variables,
    the one-hot columns in *beta_corrections* are collapsed back to a single
    column per variable by summing across levels (only the active level's
    column is non-zero, so no information is lost).

    Adding the output of this function to that of :func:`data_beta_coeff_glm`
    gives the combined GLM + booster beta coefficients per observation.

    Parameters
    ----------
    data:
        Predictor DataFrame.
    beta_corrections:
        Per-row booster beta corrections from :func:`beta_corrections_derive`.
    iblm_model:
        Fitted :class:`~iblm.IBLM` model.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same column structure as
        :func:`data_beta_coeff_glm`.
    """
    _check_iblm_fitted(iblm_model)

    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_vars = iblm_model.predictor_vars["categorical"]
    cat_levels_all = iblm_model.cat_levels["all"]
    feature_cols = iblm_model.predictor_vars["all"]

    n = len(data)
    result: dict[str, np.ndarray] = {}

    result["bias"] = beta_corrections["bias"].to_numpy(dtype=float)

    for col in cont_vars:
        result[col] = beta_corrections[col].to_numpy(dtype=float)

    for col in cat_vars:
        level_cols = [
            f"{col}{lvl}"
            for lvl in cat_levels_all[col]
            if f"{col}{lvl}" in beta_corrections.columns
        ]
        result[col] = beta_corrections[level_cols].sum(axis=1).to_numpy(dtype=float)

    cols = ["bias"] + feature_cols
    return pd.DataFrame({c: result[c] for c in cols if c in result}, index=data.index)


# ---------------------------------------------------------------------------
# Internal guard
# ---------------------------------------------------------------------------


def _check_iblm_fitted(iblm_model: "IBLM") -> None:
    if iblm_model.glm_model is None:
        raise RuntimeError("iblm_model has not been fitted yet.")
