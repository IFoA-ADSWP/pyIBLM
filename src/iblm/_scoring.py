"""Model evaluation: get_pinball_scores and correction_corridor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from ._theme import IBLM_COLORS, _apply_theme

if TYPE_CHECKING:
    from ._model import IBLM


# ---------------------------------------------------------------------------
# get_pinball_scores
# ---------------------------------------------------------------------------


def get_pinball_scores(
    data: pd.DataFrame,
    iblm_model: "IBLM",
    trim: float | None = None,
    additional_models: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute deviance and pinball scores for IBLM and comparison models.

    Scores are computed for three built-in models and any additional models
    supplied:

    * **homog** – intercept-only (homogeneous) baseline fitted on the
      training data; all predictions equal the weighted training mean.
    * **glm** – the GLM component of *iblm_model*.
    * **iblm** – the full IBLM ensemble.

    Pinball scores are calculated relative to the homogeneous baseline.
    Higher (more positive) scores indicate better predictive performance; a
    negative score indicates worse than the homogeneous baseline.

    Parameters
    ----------
    data:
        Test DataFrame.  Typically ``df_dict["test"]`` from
        :func:`~iblm.split_into_train_validate_test`.
    iblm_model:
        A fitted :class:`~iblm.IBLM` model.
    trim:
        Optional trim value forwarded to :meth:`~iblm.IBLM.predict`.
    additional_models:
        Optional dict of ``{name: model}`` for additional comparisons.
        Each model must expose a ``.predict(X)`` method or be an
        ``xgb.Booster``.  Models must have been fitted on the same training
        data as *iblm_model* for sensible results.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per model and the following columns:

        * ``"model"`` – model label (``"homog"``, ``"glm"``, ``"iblm"``,
          plus any keys from *additional_models*).
        * ``"<family>_deviance"`` – mean deviance on the test data using the
          loss function for the fitted family.
        * ``"pinball_score"`` – ``1 - deviance / homog_deviance``.
    """
    if iblm_model.glm_model is None:
        raise RuntimeError("iblm_model has not been fitted.")

    additional_models = additional_models or {}

    response_var = iblm_model.response_var
    weight_var = iblm_model.weight_var
    offset_var = iblm_model.offset_var
    glm_family = iblm_model.glm_model.family
    linkinv = glm_family.link.inverse

    actual = data[response_var].to_numpy(dtype=float)
    weights = (
        data[weight_var].to_numpy(dtype=float) if weight_var and weight_var in data.columns
        else None
    )

    # Offset for test data
    if offset_var and offset_var in data.columns:
        test_offset = data[offset_var].to_numpy(dtype=float)
    else:
        if offset_var and offset_var not in data.columns:
            import warnings
            warnings.warn(
                f"Column '{offset_var}' not found in data. Offset of 0 assumed.",
                stacklevel=2,
            )
        test_offset = np.zeros(len(data))

    # ------------------------------------------------------------------
    # Homogeneous baseline: intercept-only GLM on training data
    # ------------------------------------------------------------------
    train_df = iblm_model.data["train"]
    y_train = train_df[response_var].to_numpy(dtype=float)

    glm_kw: dict[str, Any] = dict(
        endog=y_train,
        exog=np.ones((len(y_train), 1)),
        family=glm_family,
    )
    if offset_var and offset_var in train_df.columns:
        glm_kw["offset"] = train_df[offset_var].to_numpy(dtype=float)
    if weight_var and weight_var in train_df.columns:
        glm_kw["var_weights"] = train_df[weight_var].to_numpy(dtype=float)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        homog_glm = sm.GLM(**glm_kw).fit(disp=0)

    beta0 = float(homog_glm.params[0])
    homog_preds = linkinv(beta0 + test_offset)

    # ------------------------------------------------------------------
    # GLM predictions on test data
    # ------------------------------------------------------------------
    from ._model import _build_design_matrix, _get_xgb_feature_df

    X_test = _build_design_matrix(
        data,
        iblm_model.predictor_vars["continuous"],
        iblm_model.predictor_vars["categorical"],
        iblm_model.cat_levels["all"],
        iblm_model.cat_levels["reference"],
    )
    glm_preds = iblm_model.glm_model.predict(
        exog=X_test, offset=test_offset
    )
    glm_preds = np.asarray(glm_preds)

    # IBLM predictions
    iblm_preds = iblm_model.predict(data, trim=trim, type="response")

    model_predictions: dict[str, np.ndarray] = {
        "homog": np.asarray(homog_preds),
        "glm": glm_preds,
        "iblm": iblm_preds,
    }

    # ------------------------------------------------------------------
    # Additional models
    # ------------------------------------------------------------------
    for name, model in additional_models.items():
        if isinstance(model, xgb.Booster):
            X_xgb = _get_xgb_feature_df(
                data, iblm_model.predictor_vars, iblm_model.cat_levels["all"]
            )
            dmat = xgb.DMatrix(X_xgb, enable_categorical=True)
            if offset_var and offset_var in data.columns:
                dmat.set_base_margin(test_offset)
            model_predictions[name] = model.predict(dmat)
        elif hasattr(model, "predict"):
            # scikit-learn / statsmodels style
            try:
                model_predictions[name] = np.asarray(
                    model.predict(X_test, offset=test_offset)
                )
            except TypeError:
                model_predictions[name] = np.asarray(model.predict(X_test))
        else:
            raise TypeError(
                f"Additional model '{name}' must have a .predict() method or be an xgb.Booster."
            )

    # ------------------------------------------------------------------
    # Compute deviances and pinball scores
    # ------------------------------------------------------------------
    family_name = type(iblm_model.glm_model.family).__name__.lower()
    # Normalise: statsmodels Tweedie → "tweedie"
    if "tweedie" in family_name:
        family_name = "tweedie"
    elif "poisson" in family_name:
        family_name = "poisson"
    elif "gamma" in family_name:
        family_name = "gamma"
    elif "gaussian" in family_name:
        family_name = "gaussian"

    devcol = f"{family_name}_deviance"

    records = []
    for model_name, preds in model_predictions.items():
        dev = _calculate_deviance(actual, preds, family_name, weights)
        records.append({"model": model_name, devcol: dev})

    result_df = pd.DataFrame(records)
    homog_dev = result_df.loc[result_df["model"] == "homog", devcol].iloc[0]
    result_df["pinball_score"] = 1 - result_df[devcol] / homog_dev

    return result_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# correction_corridor
# ---------------------------------------------------------------------------


def correction_corridor(
    iblm_model: "IBLM",
    data: pd.DataFrame,
    trim_vals: list[float | None] | None = None,
    sample_perc: float = 0.2,
    color: str | None = None,
    seed: int | None = None,
    **scatter_kwargs: Any,
) -> plt.Figure:
    """Faceted scatter plot of GLM vs IBLM predictions across trim values.

    Creates a faceted scatter plot comparing the GLM predictions to the full
    IBLM ensemble predictions across different trim values.  The diagonal line
    (y = x) represents perfect agreement between the two components; deviation
    from the diagonal shows the magnitude of the booster's correction.

    One facet is produced per entry in *trim_vals*.

    Parameters
    ----------
    iblm_model:
        A fitted :class:`~iblm.IBLM` model.
    data:
        DataFrame to plot.  Typically the test split returned by
        :func:`~iblm.split_into_train_validate_test`.
    trim_vals:
        Trim values to display, one facet each.  ``None`` in the list means
        no trimming.  Defaults to ``[None, 4, 1, 0.2, 0.1, 0]``.
    sample_perc:
        Fraction of *data* to sample before plotting.  Default ``0.2``
        improves performance with large datasets.
    color:
        Optional name of a variable in *data* to colour scatter points by.
    seed:
        Random seed for the sample.
    **scatter_kwargs:
        Additional keyword arguments forwarded to the scatter plot call.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if iblm_model.glm_model is None:
        raise RuntimeError("iblm_model has not been fitted.")

    if trim_vals is None:
        trim_vals = [None, 4, 1, 0.2, 0.1, 0]

    df = data.sample(frac=sample_perc, random_state=seed).reset_index(drop=True)

    from ._model import _build_design_matrix

    offset_var = iblm_model.offset_var
    if offset_var and offset_var in df.columns:
        offset = df[offset_var].to_numpy(dtype=float)
    else:
        offset = np.zeros(len(df))

    X_glm = _build_design_matrix(
        df,
        iblm_model.predictor_vars["continuous"],
        iblm_model.predictor_vars["categorical"],
        iblm_model.cat_levels["all"],
        iblm_model.cat_levels["reference"],
    )
    glm_pred = np.asarray(
        iblm_model.glm_model.predict(exog=X_glm, offset=offset)
    )

    # Collect predictions per trim value
    rows_list: list[pd.DataFrame] = []
    for trim_val in trim_vals:
        iblm_pred = iblm_model.predict(df, trim=trim_val, type="response")
        label = "None" if trim_val is None else str(trim_val)
        chunk = pd.DataFrame({
            "glm": glm_pred,
            "iblm": iblm_pred,
            "trim": label,
        })
        if color and color in df.columns:
            chunk[color] = df[color].values
        rows_list.append(chunk)

    df_all = pd.concat(rows_list, ignore_index=True)

    trim_order = [
        "None" if t is None else str(t) for t in trim_vals
    ]
    df_all["trim"] = pd.Categorical(df_all["trim"], categories=trim_order, ordered=True)

    n_facets = len(trim_vals)
    ncols = min(3, n_facets)
    nrows = (n_facets + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, trim_label in enumerate(trim_order):
        ax = axes_flat[i]
        sub = df_all[df_all["trim"] == trim_label]

        sc_kwargs = {"s": 8, "alpha": 0.4, **scatter_kwargs}

        if color and color in sub.columns:
            c_vals = sub[color].to_numpy()
            if pd.api.types.is_numeric_dtype(sub[color]):
                sc = ax.scatter(sub["glm"], sub["iblm"], c=c_vals,
                                cmap="Blues_r", **sc_kwargs)
                plt.colorbar(sc, ax=ax)
            else:
                cats = sorted(sub[color].unique())
                pal = sns.color_palette("tab10", len(cats))
                col_map = {c: pal[j] for j, c in enumerate(cats)}
                c_colors = [col_map[v] for v in c_vals]
                ax.scatter(sub["glm"], sub["iblm"], c=c_colors, **sc_kwargs)
        else:
            ax.scatter(sub["glm"], sub["iblm"], color=IBLM_COLORS[0], **sc_kwargs)

        # y = x diagonal
        lo = min(sub["glm"].min(), sub["iblm"].min())
        hi = max(sub["glm"].max(), sub["iblm"].max())
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("GLM Prediction")
        ax.set_ylabel("IBLM Prediction")
        ax.set_title(f"trim = {trim_label}")
        _apply_theme(ax)

    # Hide unused subplots
    for j in range(n_facets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Correction Corridor by Trim Value",
                 fontsize=13, fontweight="bold", color=IBLM_COLORS[4])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deviance calculation
# ---------------------------------------------------------------------------


def _calculate_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    family: str,
    weight: np.ndarray | None = None,
    correction: float = 1e-10,
) -> float:
    """Compute the weighted mean deviance for a given GLM family."""
    family = family.lower()
    y_true = np.asarray(y_true, dtype=float) + correction
    y_pred = np.asarray(y_pred, dtype=float) + correction

    if weight is None:
        weight = np.ones_like(y_true)
    weight = np.asarray(weight, dtype=float)
    w_sum = weight.sum()

    if family == "gaussian":
        return float(np.sum(weight * (y_true - y_pred) ** 2) / w_sum)

    if family == "poisson":
        return float(
            2 * np.sum(weight * (y_pred - y_true - y_true * np.log(y_pred / y_true))) / w_sum
        )

    if family == "gamma":
        return float(
            2 * np.sum(weight * (-np.log(y_true / y_pred) + (y_true - y_pred) / y_pred)) / w_sum
        )

    if family == "tweedie":
        p = 1.5
        return float(
            2 * np.sum(
                weight * (
                    (y_true ** (2 - p)) / ((1 - p) * (2 - p))
                    - (y_true * y_pred ** (1 - p)) / (1 - p)
                    + (y_pred ** (2 - p)) / (2 - p)
                )
            ) / w_sum
        )

    raise ValueError(f"family must be one of: gaussian, poisson, gamma, tweedie. Got '{family}'.")
