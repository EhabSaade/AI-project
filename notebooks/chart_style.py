"""Shared styling for the report figures.

Colours follow the dataviz reference palette's light theme. The series slots
listed here were run through its palette validator together on this surface
(all checks pass); validate again before adding a slot.
"""

from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

CHART = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834"],
}

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)


def style_axes(ax):
    ax.set_facecolor(CHART["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHART["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, axis="y", color=CHART["grid"], linewidth=0.8, linestyle="-", alpha=1)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelcolor=CHART["ink_muted"])
    ax.xaxis.label.set_color(CHART["ink_secondary"])
    ax.yaxis.label.set_color(CHART["ink_secondary"])
    ax.title.set_color(CHART["ink"])


def add_legend(ax, **kwargs):
    """Legend in text ink; drops the marker ring so keys do not read as dashed lines."""
    legend = ax.legend(frameon=False, labelcolor=CHART["ink_secondary"], fontsize=9, **kwargs)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(0)
    return legend


def plot_series(ax, x, y, color, label=None, markevery=None):
    """A 2px line with ringed markers; pass markevery=[-1] for an end-dot only."""
    ax.plot(
        x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
        marker="o", markersize=7, markeredgecolor=CHART["surface"], markeredgewidth=1.5,
        markevery=markevery, label=label,
    )
