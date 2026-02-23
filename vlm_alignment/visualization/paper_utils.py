"""Utilities for composing multi-panel publication figures."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, get_data_type_color,
    style_axis, COLORS,
)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.10):
    """Add a panel label like '(a)' to the top-left corner of an axis."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=16, fontweight="bold",
        va="top", ha="left",
    )


def bar_chart_on_axis(
    ax: plt.Axes,
    names: List[str],
    values: List[float],
    title: str = "",
    ylabel: str = "Score",
    show_values: bool = True,
    ylim_top: Optional[float] = None,
    fmt: str = ".4f",
):
    """Draw a standard bar chart on a given axis."""
    colors = [get_model_color(n) for n in names]
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.5)
    if show_values:
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:{fmt}}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    else:
        ax.set_ylim(0, max(values) * 1.18 if values else 1.0)
    style_axis(ax, title=title, ylabel=ylabel)


def grouped_bar_on_axis(
    ax: plt.Axes,
    group_names: List[str],
    series: dict,
    title: str = "",
    ylabel: str = "Score",
    show_values: bool = True,
    fmt: str = ".3f",
):
    """Draw a grouped bar chart. series = {series_name: [values]}."""
    n_groups = len(group_names)
    n_series = len(series)
    width = 0.8 / n_series
    x = np.arange(n_groups)

    for i, (name, values) in enumerate(series.items()):
        offset = (i - n_series / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width, label=name.upper(),
            color=get_model_color(name), edgecolor="black", linewidth=1,
        )
        if show_values:
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(max(v) for v in series.values()) * 0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in group_names])
    ax.legend(fontsize=10)
    style_axis(ax, title=title, ylabel=ylabel)


def scatter_with_labels(
    ax: plt.Axes,
    x_vals: List[float],
    y_vals: List[float],
    names: List[str],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    show_diagonal: bool = False,
    show_trend: bool = False,
    annotation_text: Optional[str] = None,
):
    """Draw a labeled scatter plot."""
    colors = [get_model_color(n) for n in names]
    ax.scatter(x_vals, y_vals, c=colors, s=200, edgecolors="black",
               linewidth=2, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name.upper(), (x_vals[i], y_vals[i]),
            textcoords="offset points", xytext=(10, 5),
            fontsize=11, fontweight="bold",
        )
    if show_diagonal:
        lims = [
            min(min(x_vals), min(y_vals)) - 0.05,
            max(max(x_vals), max(y_vals)) + 0.05,
        ]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
        ax.legend(fontsize=9)
    if show_trend:
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_vals) - 0.01, max(x_vals) + 0.01, 50)
        ax.plot(x_line, p(x_line), "k--", alpha=0.3, linewidth=1.5)
    if annotation_text:
        ax.text(
            0.05, 0.95, annotation_text,
            transform=ax.transAxes, fontsize=10,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"),
        )
    style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
