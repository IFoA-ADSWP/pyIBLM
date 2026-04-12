"""IBLM colour palette and matplotlib/seaborn theme."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Mirrors iblm_colors in theme_iblm.R
IBLM_COLORS: list[str] = [
    "#113458",  # [0] dark navy  – primary line colour
    "#D9AB16",  # [1] amber      – secondary / highlight colour
    "#4096C0",  # [2] sky blue   – tertiary colour
    "#DCDCD9",  # [3] light grey – grid lines / fills
    "#113458",  # [4] dark navy  – title text (same as [0])
    "#2166AC",  # [5] mid blue
    "#FFFFFF",  # [6] white
    "#B2182B",  # [7] red
]


def theme_iblm(ax: plt.Axes | None = None) -> dict:
    """Apply the IBLM visual style to *ax* (or the current axes).

    Returns the dict of ``rcParams`` updates so callers can inspect what
    was applied.  Typical usage::

        fig, ax = plt.subplots()
        theme_iblm(ax)
    """
    rc: dict = {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.color": IBLM_COLORS[3],
        "grid.linewidth": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": IBLM_COLORS[4],
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }

    plt.rcParams.update(rc)

    if ax is not None:
        ax.set_facecolor("white")
        ax.grid(True, color=IBLM_COLORS[3], linewidth=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return rc


def _apply_theme(ax: plt.Axes) -> None:
    """Apply ``theme_iblm`` to a single Axes object.

    This is the internal helper called by every plot function in the package.
    It delegates to the public :func:`theme_iblm` so the two are always
    consistent.
    """
    theme_iblm(ax)
