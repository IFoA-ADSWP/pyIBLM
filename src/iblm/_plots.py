"""Plotting implementations for IBLM explainability visualisations.

All functions return ``matplotlib.figure.Figure`` objects (or dicts of them).
They do NOT call ``plt.show()`` – the caller decides when/how to display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._theme import IBLM_COLORS, _apply_theme
from ._utils import _assign_variable_type, _detect_outliers

if TYPE_CHECKING:
    from ._model import IBLM


# ---------------------------------------------------------------------------
# beta_corrected_scatter
# ---------------------------------------------------------------------------


def _beta_corrected_scatter_plot(
    varname: str,
    q: float,
    color: str | None,
    data_beta_coeff: pd.DataFrame,
    data: pd.DataFrame,
    iblm_model: "IBLM",
) -> plt.Figure:
    """Scatter / boxplot of beta coefficients after SHAP corrections."""
    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_vars = iblm_model.predictor_vars["categorical"]
    glm_params = iblm_model.glm_model.params
    glm_bse = iblm_model.glm_model.bse

    vartype = _assign_variable_type(varname, cont_vars, cat_vars)

    plot_df = data.copy()
    plot_df["_beta_coeff"] = data_beta_coeff[varname].values

    fig, ax = plt.subplots(figsize=(8, 5))

    if vartype == "categorical":
        # ---- Boxplot per level (reference level excluded) ----
        cat_levels = iblm_model.cat_levels["all"][varname]
        ref_level = iblm_model.cat_levels["reference"][varname]
        non_ref = [lvl for lvl in cat_levels if lvl != ref_level]

        plot_df_filtered = plot_df[plot_df[varname] != ref_level].copy()
        plot_df_filtered[varname] = plot_df_filtered[varname].astype(str)

        sns.boxplot(
            data=plot_df_filtered,
            x=varname,
            y="_beta_coeff",
            order=[str(l) for l in non_ref],
            color=IBLM_COLORS[3],
            linecolor=IBLM_COLORS[0],
            ax=ax,
        )

        # Overlay GLM coefficients as blue dots
        glm_coeff_rows = []
        for lvl in non_ref:
            coeff_name = f"{varname}{lvl}"
            if coeff_name in glm_params.index:
                glm_coeff_rows.append({"level": str(lvl), "coeff": glm_params[coeff_name]})
        if glm_coeff_rows:
            coeff_df = pd.DataFrame(glm_coeff_rows)
            x_positions = [str(l) for l in non_ref]
            for _, row in coeff_df.iterrows():
                xi = x_positions.index(row["level"])
                ax.scatter(xi, row["coeff"], color=IBLM_COLORS[2], zorder=5, s=60)

        ax.set_xlabel(varname)
        ax.set_ylabel("Beta Coefficients")
        ax.set_title(f"Beta Coefficients after SHAP corrections for {varname}")

    else:
        # ---- Scatter + lowess smooth (continuous) ----
        if q > 0:
            keep = _detect_outliers(plot_df["_beta_coeff"].to_numpy(), q=q)
            plot_df = plot_df[keep]

        x_vals = plot_df[varname].to_numpy(dtype=float)
        y_vals = plot_df["_beta_coeff"].to_numpy(dtype=float)

        glm_coeff_name = varname
        beta = glm_params.get(glm_coeff_name, np.nan)
        show_se = iblm_model.family == "poisson"
        try:
            stderror = glm_bse[glm_coeff_name] if show_se else np.nan
        except KeyError:
            stderror = np.nan

        # Determine colour mapping
        if color is not None and color in data.columns:
            color_vals = plot_df[color].to_numpy()
            color_vartype = _assign_variable_type(color, cont_vars, cat_vars)
            if color_vartype == "numerical":
                sc = ax.scatter(
                    x_vals, y_vals, c=color_vals, cmap="Blues_r", alpha=0.4, s=8, zorder=3
                )
                plt.colorbar(sc, ax=ax, label=color)
            else:
                unique_cats = sorted(set(color_vals.tolist()))
                palette = sns.color_palette("tab10", len(unique_cats))
                cat_map = {c: palette[i] for i, c in enumerate(unique_cats)}
                c_colors = [cat_map[v] for v in color_vals]
                ax.scatter(x_vals, y_vals, c=c_colors, alpha=0.4, s=8, zorder=3)
                handles = [
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=palette[i], markersize=8, label=str(c))
                    for i, c in enumerate(unique_cats)
                ]
                ax.legend(handles=handles, title=color, fontsize=8)
        else:
            ax.scatter(x_vals, y_vals, color=IBLM_COLORS[0], alpha=0.4, s=8, zorder=3)

        # LOWESS smooth
        if len(x_vals) >= 10:
            sort_idx = np.argsort(x_vals)
            smoothed = lowess(y_vals[sort_idx], x_vals[sort_idx], frac=0.3)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=IBLM_COLORS[2], linewidth=1.5, zorder=4)

        # GLM coefficient line
        if not np.isnan(beta):
            ax.axhline(beta, color="black", linewidth=0.8, zorder=5)
            if not np.isnan(stderror):
                ax.axhline(beta - stderror, color="black", linewidth=0.5,
                           linestyle="--", zorder=5)
                ax.axhline(beta + stderror, color="black", linewidth=0.5,
                           linestyle="--", zorder=5)

        subtitle_parts = [f"{varname} beta: {round(beta, 3)}"]
        if not np.isnan(stderror):
            subtitle_parts.append(f"SE: +/-{round(stderror, 4)}")
        ax.set_title(
            f"Beta Coefficients after SHAP corrections for {varname}"
            f"\n{', '.join(subtitle_parts)}",
            fontsize=11,
        )

        ax.set_xlabel(varname)
        ax.set_ylabel("Beta Coefficients")

    _apply_theme(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# beta_corrected_density
# ---------------------------------------------------------------------------


def _beta_corrected_density_plot(
    varname: str,
    q: float,
    plot_type: str,
    wide_input_frame: pd.DataFrame,
    beta_corrections: pd.DataFrame,
    data: pd.DataFrame,
    iblm_model: "IBLM",
) -> "plt.Figure | dict[str, plt.Figure]":
    """Density / histogram of corrected beta values for a variable."""
    if not (0 <= q < 0.5):
        raise ValueError("q must satisfy 0 <= q < 0.5.")
    if plot_type not in ("kde", "hist"):
        raise ValueError("type must be 'kde' or 'hist'.")

    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_vars = iblm_model.predictor_vars["categorical"]
    cat_levels_all = iblm_model.cat_levels["all"]
    ref_cat = iblm_model.coeff_names["reference_cat"]
    glm_params = iblm_model.glm_model.params
    glm_bse = iblm_model.glm_model.bse

    # Determine vartype
    if varname in cont_vars:
        vartype = "numerical"
    elif varname in cat_vars:
        vartype = "categorical"
    elif varname in ref_cat:
        raise ValueError(
            f"'{varname}' is a reference level – no beta coefficient exists for it."
        )
    elif varname in glm_params.index:
        vartype = "categorical_level"
    else:
        valid = sorted(set(cont_vars) | set(cat_vars))
        raise ValueError(
            f"'{varname}' not found in model. Valid predictors: {valid}"
        )

    # For categorical: recurse to produce a dict of plots
    if vartype == "categorical":
        levels_to_plot = [
            f"{varname}{lvl}"
            for lvl in cat_levels_all[varname]
            if f"{varname}{lvl}" in glm_params.index
        ]
        return {
            lvl_name: _beta_corrected_density_plot(
                varname=lvl_name,
                q=q,
                plot_type=plot_type,
                wide_input_frame=wide_input_frame,
                beta_corrections=beta_corrections,
                data=data,
                iblm_model=iblm_model,
            )
            for lvl_name in levels_to_plot
        }

    # Single plot
    beta = glm_params.get(varname, np.nan)
    show_se = iblm_model.family == "poisson"
    try:
        stderror = glm_bse[varname] if show_se else np.nan
    except KeyError:
        stderror = np.nan

    shap_deviations = beta_corrections[varname].to_numpy(dtype=float)

    # For a specific categorical level: keep only rows where that level is active
    if vartype == "categorical_level" and varname in wide_input_frame.columns:
        mask = wide_input_frame[varname].to_numpy(dtype=float) == 1
        shap_deviations = shap_deviations[mask]

    corrected_values = beta + shap_deviations

    # Axis bounds – only extend to beta ± SE when SE is shown
    if q > 0:
        lo_q, hi_q = np.quantile(corrected_values, [q, 1 - q])
    else:
        lo_q, hi_q = corrected_values.min(), corrected_values.max()

    stderror_val = 0 if np.isnan(stderror) else stderror
    lower_bound = min(lo_q, beta - stderror_val)
    upper_bound = max(hi_q, beta + stderror_val)

    fig, ax = plt.subplots(figsize=(7, 4))

    plot_data = corrected_values[
        (corrected_values >= lower_bound) & (corrected_values <= upper_bound)
    ]

    if plot_type == "kde":
        sns.kdeplot(
            plot_data, ax=ax,
            color=IBLM_COLORS[0], fill=True, alpha=0.3,
        )
    else:
        ax.hist(
            plot_data, bins=100,
            color=IBLM_COLORS[3], edgecolor=IBLM_COLORS[0], alpha=0.6,
        )

    # Vertical lines
    if not np.isnan(beta):
        ax.axvline(beta, color=IBLM_COLORS[1], linewidth=1.0)
    if not np.isnan(stderror):
        ax.axvline(beta - stderror, color=IBLM_COLORS[1], linewidth=0.8, linestyle="--")
        ax.axvline(beta + stderror, color=IBLM_COLORS[1], linewidth=0.8, linestyle="--")

    ax.set_xlim(lower_bound, upper_bound)
    ax.set_xlabel("Beta Coefficients")
    subtitle = f"{varname} beta: {round(beta, 3)}"
    if not np.isnan(stderror):
        subtitle += f", SE: +/-{round(stderror, 4)}"
    ax.set_title(
        f"Beta density after SHAP corrections for {varname}\n{subtitle}",
        fontsize=11,
    )

    _apply_theme(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# bias_density
# ---------------------------------------------------------------------------


def _bias_density_plot(
    q: float,
    plot_type: str,
    migrate_reference_to_bias: bool,
    shap: pd.DataFrame,
    data: pd.DataFrame,
    iblm_model: "IBLM",
) -> dict[str, plt.Figure | None]:
    """Density plots for SHAP corrections migrated to bias."""
    if not (0 <= q < 0.5):
        raise ValueError("q must satisfy 0 <= q < 0.5.")
    if plot_type not in ("kde", "hist"):
        raise ValueError("type must be 'kde' or 'hist'.")

    cont_vars = iblm_model.predictor_vars["continuous"]
    cat_vars = iblm_model.predictor_vars["categorical"]
    ref_levels = iblm_model.cat_levels["reference"]
    glm_params = iblm_model.glm_model.params
    glm_bse = iblm_model.glm_model.bse
    show_se = iblm_model.family == "poisson"

    rows: list[dict] = []

    # Continuous variables: rows where feature == 0
    for var in cont_vars:
        if var not in data.columns:
            continue
        is_zero = data[var].to_numpy(dtype=float) == 0
        row_ids = np.where(is_zero)[0]
        corrections = shap.loc[data.index[is_zero], var].to_numpy(dtype=float)
        for rid, corr in zip(row_ids, corrections):
            rows.append({"row_id": rid, "var": var, "bias_correction": corr})

    # Categorical variables: rows at reference level
    if migrate_reference_to_bias:
        for var in cat_vars:
            ref = ref_levels[var]
            is_ref = data[var] == ref
            row_ids = np.where(is_ref.to_numpy())[0]
            corrections = shap.loc[data.index[is_ref], var].to_numpy(dtype=float)
            for rid, corr in zip(row_ids, corrections):
                rows.append({"row_id": rid, "var": var, "bias_correction": corr})

    if not rows:
        import warnings
        warnings.warn("No bias migration within dataset when calling bias_density().")
        return {"bias_correction_var": None, "bias_correction_total": None}

    bias_df = pd.DataFrame(rows)
    remaining_vars = bias_df["var"].unique().tolist()

    # Bounds
    all_corr = bias_df["bias_correction"].to_numpy(dtype=float)
    if q > 0:
        lo_q, hi_q = np.quantile(all_corr, [q, 1 - q])
    else:
        lo_q, hi_q = all_corr.min(), all_corr.max()

    # Standard errors for continuous variables (for reference lines, Poisson only)
    se_vals: list[float] = []
    if show_se:
        for var in cont_vars:
            if var in glm_bse.index:
                se = glm_bse[var]
                if not np.isnan(se):
                    se_vals.extend([se, -se])

    if se_vals:
        lower_bound = min(lo_q, min(se_vals))
        upper_bound = max(hi_q, max(se_vals))
    else:
        lower_bound, upper_bound = lo_q, hi_q

    # ---- Plot 1: faceted by variable ----
    unique_vars = bias_df["var"].unique()
    n_facets = len(unique_vars)
    ncols = min(3, n_facets)
    nrows = (n_facets + ncols - 1) // ncols

    fig_var, axes_var = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                     squeeze=False)
    axes_flat = axes_var.flatten()

    for i, var in enumerate(unique_vars):
        ax = axes_flat[i]
        var_data = bias_df[bias_df["var"] == var]["bias_correction"].to_numpy(dtype=float)
        var_data_clipped = var_data[
            (var_data >= lower_bound) & (var_data <= upper_bound)
        ]

        if plot_type == "kde":
            sns.kdeplot(var_data_clipped, ax=ax, color="grey", fill=True, alpha=0.3)
        else:
            ax.hist(var_data_clipped, bins=100, color="grey", alpha=0.4,
                    edgecolor="grey")

        # SE lines for continuous vars (Poisson only)
        if show_se and var in glm_bse.index:
            se = glm_bse[var]
            if not np.isnan(se):
                ax.axvline(se, color=IBLM_COLORS[1], linestyle="--", linewidth=0.8)
                ax.axvline(-se, color=IBLM_COLORS[1], linestyle="--", linewidth=0.8)

        ax.set_xlim(lower_bound, upper_bound)
        ax.set_title(var, fontsize=10)
        ax.set_xlabel("Bias Value Corrections")
        ax.set_ylabel("Count")
        _apply_theme(ax)

    # Hide unused subplots
    for j in range(n_facets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig_var.suptitle("Density for SHAP corrections that are migrated to bias",
                     fontsize=12, fontweight="bold", color=IBLM_COLORS[4])
    fig_var.tight_layout()

    # ---- Plot 2: total bias by row ----
    intercept = glm_params.get("(Intercept)", 0.0)
    try:
        se_intercept = glm_bse["(Intercept)"] if show_se else np.nan
    except KeyError:
        se_intercept = np.nan

    total_df = (
        bias_df.groupby("row_id")["bias_correction"].sum().reset_index()
    )
    total_df["bias_correction"] = total_df["bias_correction"] + intercept

    total_vals = total_df["bias_correction"].to_numpy(dtype=float)
    if q > 0:
        lo_t, hi_t = np.quantile(total_vals, [q, 1 - q])
    else:
        lo_t, hi_t = total_vals.min(), total_vals.max()

    se_i = 0 if np.isnan(se_intercept) else se_intercept
    lower_total = min(lo_t, intercept - se_i)
    upper_total = max(hi_t, intercept + se_i)

    total_clipped = total_vals[
        (total_vals >= lower_total) & (total_vals <= upper_total)
    ]

    fig_total, ax_total = plt.subplots(figsize=(7, 4))
    if plot_type == "kde":
        sns.kdeplot(total_clipped, ax=ax_total, color="grey", fill=True, alpha=0.3)
    else:
        ax_total.hist(total_clipped, bins=100, color="grey", alpha=0.4, edgecolor="grey")

    ax_total.axvline(intercept, color=IBLM_COLORS[2], linewidth=0.8)
    if not np.isnan(se_intercept):
        ax_total.axvline(intercept + se_intercept, color=IBLM_COLORS[1],
                         linestyle="--", linewidth=0.8)
        ax_total.axvline(intercept - se_intercept, color=IBLM_COLORS[1],
                         linestyle="--", linewidth=0.8)

    ax_total.set_xlim(lower_total, upper_total)
    ax_total.set_xlabel("Bias Values")
    ax_total.set_ylabel("Count")
    subtitle_total = f"bias: {round(intercept, 3)}"
    if not np.isnan(se_intercept):
        subtitle_total += f", SE: +/-{round(se_intercept, 4)}"
    ax_total.set_title(
        f"Density for corrected bias values\n{subtitle_total}",
        fontsize=11,
    )

    _apply_theme(ax_total)
    fig_total.tight_layout()

    return {
        "bias_correction_var": fig_var,
        "bias_correction_total": fig_total,
    }


# ---------------------------------------------------------------------------
# overall_correction
# ---------------------------------------------------------------------------


def _overall_correction_plot(
    transform_x_scale_by_link: bool,
    shap: pd.DataFrame,
    iblm_model: "IBLM",
) -> plt.Figure:
    """Distribution of the booster's overall multiplicative / additive correction."""
    family = iblm_model.glm_model.family
    relationship = iblm_model.relationship
    link_name = type(family.link).__name__.lower()

    # Sum all SHAP contributions → total booster output on link scale
    total_link = shap.sum(axis=1).to_numpy(dtype=float)

    # Inverse-link transform
    try:
        total_invlink = family.link.inverse(total_link)
    except Exception:
        # Fallback: apply numpy-based inverse manually
        if link_name == "log":
            total_invlink = np.exp(total_link)
        else:
            total_invlink = total_link

    baseline = family.link.inverse(np.array([0.0]))[0]  # linkinv(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(total_invlink, ax=ax, color=IBLM_COLORS[0], fill=True, alpha=0.3)
    ax.axvline(baseline, color="black", linewidth=0.8)

    # Optionally transform the x-axis
    if transform_x_scale_by_link and link_name == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xlabel(f"{relationship.title()} correction (log scale)")
        ax.annotate(
            f"** x-axis is on log scale",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8,
        )
    else:
        xlabel = f"{relationship.title()} correction"
        ax.set_xlabel(xlabel)

    mean_corr = round(float(np.mean(total_invlink)), 3)
    ax.set_title(
        f"Distribution of {relationship} corrections to GLM prediction"
        f"\nmean correction: {mean_corr}",
        fontsize=11,
    )

    _apply_theme(ax)
    fig.tight_layout()
    return fig
