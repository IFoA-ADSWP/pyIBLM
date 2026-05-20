"""IBLM – Interpretable Boosted Linear Models.

An ensemble modelling framework that combines a Generalized Linear Model
(GLM) with an XGBoost booster trained on the GLM residuals to produce
predictions that are both accurate and interpretable via SHAP values.

Depending on the link function used, the two components are combined as:

* **Multiplicative** (log-link families such as Poisson, Gamma, Tweedie):
  ``prediction = GLM prediction × Booster prediction``
* **Additive** (identity-link families such as Gaussian):
  ``prediction = GLM prediction + Booster prediction``

Typical workflow::

    import pandas as pd
    from iblm import (
        load_freMTPLmini,
        split_into_train_validate_test,
        IBLM,
        ExplainIBLM,
        get_pinball_scores,
        correction_corridor,
    )

    import numpy as np

    df = load_freMTPLmini()
    df["LogExposure"] = np.log(df["Exposure"])
    df = df.drop(columns=["Exposure"])
    df_dict = split_into_train_validate_test(df, seed=9000)

    model = IBLM()
    model.fit(
        df_dict,
        response_var="ClaimNb",
        offset_var="LogExposure",
        family="poisson",
    )

    preds = model.predict(df_dict["test"])

    ex = ExplainIBLM(model, df_dict["test"])
    fig = ex.beta_corrected_scatter("DrivAge")
    fig.show()

    scores = get_pinball_scores(df_dict["test"], model)
"""

from ._data import load_freMTPL2freq, load_freMTPLmini
from ._explain import (
    ExplainIBLM,
    beta_corrections_derive,
    data_beta_coeff_booster,
    data_beta_coeff_glm,
    data_to_onehot,
    explain_iblm,
    shap_to_onehot,
)
from ._model import IBLM, train_xgb_as_per_iblm
from ._scoring import correction_corridor, get_pinball_scores
from ._shap import extract_booster_shap
from ._theme import IBLM_COLORS, theme_iblm
from ._utils import split_into_train_validate_test

__all__ = [
    # Core model
    "IBLM",
    # Explainer
    "ExplainIBLM",
    "explain_iblm",
    # SHAP extraction (extensible via @extract_booster_shap.register)
    "extract_booster_shap",
    # Data helpers
    "data_to_onehot",
    "shap_to_onehot",
    "beta_corrections_derive",
    "data_beta_coeff_glm",
    "data_beta_coeff_booster",
    # Evaluation
    "get_pinball_scores",
    "correction_corridor",
    # Utilities
    "split_into_train_validate_test",
    "train_xgb_as_per_iblm",
    # Theme
    "theme_iblm",
    "IBLM_COLORS",
    # Datasets
    "load_freMTPLmini",
    "load_freMTPL2freq",
    # Package metadata
    "__version__",
    "__authors__",
    "__contributors__",
]

__version__ = "2.0.2"
__authors__ = [
    "Paul Beard <paul.beard.actuarial@gmail.com>",
    "Karol Gawlowski <kg.actuarial@gmail.com>",
]
__contributors__ = [
    "Zhouwen Zhou <zhou.zhuowen@hotmail.com>",
]
