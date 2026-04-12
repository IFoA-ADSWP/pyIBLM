"""Dataset loading utilities."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pandas as pd


def load_freMTPLmini() -> pd.DataFrame:
    """Load the bundled ``freMTPLmini`` dataset.

    This is a 25,000-row sample of the French motor third-party liability
    dataset, pre-processed to be ready for use with :class:`~iblm.IBLM`.

    The dataset includes a ``LogExposure`` column (``log(Exposure)``) and
    categorical columns encoded as ``pandas.Categorical``.

    Returns
    -------
    pd.DataFrame
    """
    data_path = importlib.resources.files("iblm") / "data" / "freMTPLmini.csv"
    df = pd.read_csv(data_path)

    # Convert string/object columns to categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    return df


def load_freMTPL2freq() -> pd.DataFrame:
    """Load ``freMTPL2freq`` by downloading it from the CASdatasets repository.

    Requires the optional ``rdata`` package::

        pip install iblm[data]

    Applies the same pre-processing as the R ``IBLM`` package:

    * ``ClaimNb`` cast to float and divided by ``Exposure`` (claim rate).
    * ``ClaimNb`` winsorised at the 99.9th percentile.
    * ``VehAge`` clipped at 50.
    * ``IDpol`` and ``Exposure`` dropped.
    * String columns converted to ``pandas.Categorical``.

    Returns
    -------
    pd.DataFrame
    """
    try:
        import rdata
    except ImportError as exc:
        raise ImportError(
            "The 'rdata' package is required to use load_freMTPL2freq(). "
            "Install it with:  pip install iblm[data]"
        ) from exc

    import tempfile
    import urllib.request

    _COMMIT = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
    url = (
        f"https://github.com/dutangc/CASdatasets/raw/{_COMMIT}/data/freMTPL2freq.rda"
    )

    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        result = rdata.read_rda(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    df = result["freMTPL2freq"]

    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"]) / df["Exposure"]
    cap = df["ClaimNb"].quantile(0.999)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=cap)
    df["VehAge"] = df["VehAge"].clip(upper=50)
    df = df.drop(columns=["IDpol", "Exposure"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    return df
