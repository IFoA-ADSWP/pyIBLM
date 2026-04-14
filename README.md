# iblm

## Interpretable Boosted Linear Models

[![PyPI version](https://img.shields.io/pypi/v/iblm.svg)](https://pypi.org/project/iblm/)
[![Python versions](https://img.shields.io/pypi/pyversions/iblm.svg)](https://pypi.org/project/iblm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### Overview

**iblm** implements *Interpretable Boosted Linear Models* — a hybrid modelling approach that combines the transparency of Generalized Linear Models (GLMs) with the predictive power of gradient boosting.

The model is a two-stage ensemble:

1. A **GLM** is fitted on the training data, producing interpretable coefficient estimates.
2. An **XGBoost booster** is trained on the GLM residuals, using the GLM's linear predictor as its base margin — learning only what the GLM could not capture.

Depending on the link function, the two components are combined as:

- **Multiplicative** (log-link families: Poisson, Gamma, Tweedie):
  `prediction = GLM prediction × Booster correction`
- **Additive** (identity-link families: Gaussian):
  `prediction = GLM prediction + Booster correction`

SHAP values decompose the booster correction back onto the original GLM feature scale, making the full model auditable and interpretable at the individual prediction level.

The package provides:

- Fitting of IBLM models across Poisson, Quasi-Poisson, Gamma, Tweedie, and Gaussian families
- SHAP-based explainability tools with beta coefficient visualisations
- Model comparison via pinball scores and correction corridor plots
- Bundled insurance pricing datasets (`freMTPLmini`, `freMTPL2freq`)

An equivalent **R package** is available on CRAN:
🔗 [https://CRAN.R-project.org/package=IBLM](https://CRAN.R-project.org/package=IBLM)

---

### Installation

Install the released version from PyPI:

```bash
pip install iblm
```

To use `load_freMTPL2freq()` (downloads the full French MTPL dataset), install with the optional `data` dependency:

```bash
pip install "iblm[data]"
```

---

### Quick start

```python
import numpy as np
from iblm import (
    load_freMTPLmini,
    split_into_train_validate_test,
    IBLM,
    ExplainIBLM,
    get_pinball_scores,
)

# Load and prepare data
df = load_freMTPLmini()
df["LogExposure"] = np.log(df["Exposure"])
df = df.drop(columns=["Exposure"])

df_dict = split_into_train_validate_test(df, seed=9000)

# Fit model
model = IBLM()
model.fit(
    df_dict,
    response_var="ClaimNb",
    offset_var="LogExposure",
    family="poisson",
)

# Evaluate
scores = get_pinball_scores(df_dict["test"], model)
print(scores)

# Explain
ex = ExplainIBLM(model, df_dict["test"])
fig = ex.beta_corrected_scatter("DrivAge", color="VehPower")
fig.show()
```

---

### Documentation

For full documentation on the R implementation (functions, methods and theoretical background):

🔗 [https://ifoa-adswp.github.io/IBLM/](https://ifoa-adswp.github.io/IBLM/)

---

### Contributing

Contributions are welcome. To report a bug or suggest a feature, please open an issue on GitHub:

🔗 [https://github.com/paulbeardactuarial/vibeeLM/issues](https://github.com/paulbeardactuarial/vibeeLM/issues)

---

### Citation

If you use **iblm** in research or teaching, please cite it as:

> Gawlowski, K., Beard, P. and Zhou, Z. (2025). *iblm: Interpretable Boosted Linear Models.* Python package version 2.0.1.

---

### Authors

- **Paul Beard** — [paul.beard.actuarial@gmail.com](mailto:paul.beard.actuarial@gmail.com)
- **Karol Gawlowski** — [kg.actuarial@gmail.com](mailto:kg.actuarial@gmail.com)

Additional contributions by **Zhouwen Zhou**.

---

### License

This package is licensed under the **MIT License**.
See the [`LICENSE`](LICENSE) file for full details.
