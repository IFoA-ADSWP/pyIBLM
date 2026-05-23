# pyIBLM

## Interpretable Boosted Linear Models

[![CI](https://github.com/IFoA-ADSWP/pyIBLM/actions/workflows/ci.yml/badge.svg)](https://github.com/IFoA-ADSWP/pyIBLM/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyiblm.svg)](https://pypi.org/project/pyiblm/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyiblm.svg)](https://pypi.org/project/pyiblm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/pyiblm.svg)](https://pypi.org/project/pyiblm/)

---

### Overview

**pyIBLM** implements *Interpretable Boosted Linear Models* — a hybrid modelling approach that combines the transparency of Generalized Linear Models (GLMs) with the predictive power of gradient boosting.

The package provides:

Functions for fitting interpretable boosted linear models
Tools to analyze and visualize model results
Support for model comparison and diagnostics

An equivalent **R package** is available:
🔗 [https://ifoa-adswp.github.io/IBLM](https://ifoa-adswp.github.io/IBLM)

---

### Installation

Install the released version from PyPI:

```bash
pip install pyiblm
```

To use `load_freMTPL2freq()` (downloads the full French MTPL dataset), install with the optional `data` dependency:

```bash
pip install "pyiblm[data]"
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

### Documentation

For full documentation on the R implementation (functions, methods and theoretical background):

🔗 [https://ifoa-adswp.github.io/IBLM/](https://ifoa-adswp.github.io/IBLM/)

---

### Contributing

Contributions are welcome. To report a bug or suggest a feature, please open an issue on GitHub:

🔗 [https://github.com/IFoA-ADSWP/pyIBLM/issues](https://github.com/IFoA-ADSWP/pyIBLM/issues)

---

### Citation

If you use **pyIBLM** in research or teaching, please cite it as:

> Gawlowski, K. and Beard, P. (2026). *pyIBLM: Interpretable Boosted Linear Models.* Python package version 2.0.1.

---

### Authors

- **Karol Gawlowski** — [kg.actuarial@gmail.com](mailto:kg.actuarial@gmail.com)
- **Paul Beard** — [paul.beard.actuarial@gmail.com](mailto:paul.beard.actuarial@gmail.com)

Additional contributions by **Zhouwen Zhou**.

---

### License

This package is licensed under the **MIT License**.
See the [`LICENSE`](LICENSE) file for full details.
