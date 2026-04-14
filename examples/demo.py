"""IBLM demonstration script.

Loads the French MTPL frequency dataset, fits an IBLM model, evaluates
performance and saves explainability plots to an ``artifacts/`` directory.

Usage
-----
    pip install -e ".[data]"
    python demo.py

Notes
-----
``load_freMTPL2freq()`` downloads the dataset from GitHub on first call and
requires an internet connection.  Subsequent calls use a local cache.
"""

from pathlib import Path

import numpy as np

from iblm import (
    IBLM,
    ExplainIBLM,
    correction_corridor,
    get_pinball_scores,
    load_freMTPL2freq,
    split_into_train_validate_test,
)

OUT_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    df = load_freMTPL2freq()

    # Create log-scale offset for exposure and drop the raw column
    df["LogExposure"] = np.log(df["Exposure"])
    df = df.drop(columns=["Exposure"])

    df_dict = split_into_train_validate_test(df, seed=10)

    print("CHECKPOINT: Data loaded and split")
    print(f"  train:    {len(df_dict['train']):>7,} rows")
    print(f"  validate: {len(df_dict['validate']):>7,} rows")
    print(f"  test:     {len(df_dict['test']):>7,} rows")

    # ------------------------------------------------------------------
    # Fit IBLM
    # ------------------------------------------------------------------
    model = IBLM()
    model.fit(
        df_dict,
        response_var="ClaimNb",
        offset_var="LogExposure",
        family="poisson",
        nrounds=500,
        early_stopping_rounds=20,
        verbose=50,
        params={"max_depth": 3, "eta": 0.025},
    )

    print(f"\nCHECKPOINT: Model fitted — {model}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    scores = get_pinball_scores(df_dict["test"], model)
    print("\nPinball scores:")
    print(scores.to_string(index=False))

    # ------------------------------------------------------------------
    # Explainability plots
    # ------------------------------------------------------------------
    ex = ExplainIBLM(model, df_dict["test"])

    scatter  = ex.beta_corrected_scatter("DrivAge", color="VehPower")
    density  = ex.beta_corrected_density("VehAge")
    overall  = ex.overall_correction(transform_x_scale_by_link=False)
    corridor = correction_corridor(model, df_dict["test"], color="DrivAge", seed=42)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)

    scatter.savefig(OUT_DIR / "beta_corrected_scatter.png",  dpi=160, bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / 'beta_corrected_scatter.png'}")

    # beta_corrected_density returns a Figure for continuous variables,
    # or a dict[str, Figure] for categorical variables.
    if isinstance(density, dict):
        for level_name, fig in density.items():
            path = OUT_DIR / f"beta_corrected_density_{level_name}.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            print(f"Saved: {path}")
    else:
        density.savefig(OUT_DIR / "beta_corrected_density.png", dpi=160, bbox_inches="tight")
        print(f"Saved: {OUT_DIR / 'beta_corrected_density.png'}")

    overall.savefig(OUT_DIR / "overall_correction.png",      dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'overall_correction.png'}")

    corridor.savefig(OUT_DIR / "correction_corridor.png",    dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'correction_corridor.png'}")

    print(f"\nAll artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
