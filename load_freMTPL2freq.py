import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import rdata


def load_freMTPL2freq() -> pd.DataFrame:
    """Load freMTPL2freq directly from the CASdatasets GitHub repository.

    Downloads the .rda file from a pinned commit. Applies the
    preprocessing:
      - ClaimNb cast to float then divided by Exposure (converted to a claim rate)
      - ClaimNb winsorized at the 99.9th percentile
      - VehAge clipped at 50
      - IDpol and Exposure dropped
      - Character columns converted to pandas Categorical (equivalent to R factors)
    """
    _COMMIT = "c49cbbb37235fc49616cac8ccac32e1491cdc619"
    url = f"https://github.com/dutangc/CASdatasets/raw/{_COMMIT}/data/freMTPL2freq.rda"

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
