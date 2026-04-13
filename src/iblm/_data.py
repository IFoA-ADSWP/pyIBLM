"""Dataset loading utilities."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import pandas as pd


def load_freMTPLmini() -> pd.DataFrame:
    """Load the bundled ``freMTPLmini`` dataset.

    A 25,000-row sample of the French motor third-party liability dataset,
    pre-processed and ready for use with :class:`~iblm.IBLM`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 25,000 rows and the following columns:

        * **Area** – Area classification of the policy holder's community,
          from ``"A"`` (rural) to ``"F"`` (urban centre).
        * **BonusMalus** – Bonus-malus coefficient; values below 100
          indicate a bonus (discount), values above 100 a malus (surcharge).
        * **DrivAge** – Age of the driver in years.
        * **VehAge** – Age of the vehicle in years (capped at 50).
        * **VehBrand** – Vehicle brand code (e.g. ``"B6"``, ``"B12"``).
        * **VehPower** – Vehicle power rating.
        * **ClaimNb** – Annualised claim rate (claims divided by exposure,
          winsorised at the 99.9th percentile).
        * **LogExposure** – Natural log of the exposure period in years.

        Categorical columns are encoded as ``pandas.Categorical``.
    """
    data_path = importlib.resources.files("iblm") / "data" / "freMTPLmini.csv"
    df = pd.read_csv(data_path)

    # Convert string/object columns to categorical
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category")

    return df


def load_freMTPL2freq() -> pd.DataFrame:
    """Download and return the French Motor Third-Party Liability frequency dataset.

    Downloads ``freMTPL2freq`` from the CASdatasets repository and applies
    minor pre-processing steps.

    Requires the optional ``rdata`` package::

        pip install iblm[data]

    .. note::
        This function requires an active internet connection.

    The following pre-processing steps are applied:

    * ``ClaimNb`` cast to integer (raw claim counts; **not** divided by Exposure).
    * ``VehAge`` clipped at 50 years.
    * ``IDpol`` (policy identifier) dropped; all other columns are retained.
    * Character columns converted to ``pandas.Categorical``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:

        * **ClaimNb** – Number of claims during the exposure period.
        * **Exposure** – Length of the exposure period in years.
        * **VehPower** – Power of the car (ordered categorical).
        * **VehAge** – Vehicle age in years, capped at 50.
        * **DrivAge** – Driver age in years (minimum 18, the legal driving
          age in France).
        * **BonusMalus** – Bonus-malus coefficient (50–350); values below
          100 indicate a bonus, values above 100 a malus.
        * **VehBrand** – Car brand (categorical, with proprietary labels).
        * **VehGas** – Fuel type: ``"Diesel"`` or ``"Regular"``.
        * **Area** – Density classification of the driver's community,
          from ``"A"`` (rural) to ``"F"`` (urban centre).
        * **Density** – Population density (inhabitants per km²) of the
          driver's city.
        * **Region** – Policy region in France, based on the 1970–2015
          administrative classification.

    References
    ----------
    Dutang, C. CASdatasets: Insurance datasets.
    https://github.com/dutangc/CASdatasets
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

    df["ClaimNb"] = pd.to_numeric(df["ClaimNb"])
    df["VehAge"] = df["VehAge"].clip(upper=50)
    df = df.drop(columns=["IDpol"])

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")

    return df
