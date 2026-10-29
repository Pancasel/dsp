"""Matplotlib style and helpers for the report figures (evaluation/results/figures/*.png)."""

import math

from evaluation.shared.system_loading import FIGURES_FOLDER


# One fixed colour per method in every figure.
METHOD_COLORS = {
    "graph": "#2a78d6",
    "popularity": "#eb6834",
    "decagon": "#1baf7a",
    "random": "#a3a29b",
    "extra": "#6250d6",
}


def import_pyplot_with_report_style():
    """matplotlib.pyplot with the report style; imported only when a figure is drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#52514e",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.dpi": 150,
        }
    )
    return pyplot


def save_figure(figure, file_name: str) -> None:
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_FOLDER / file_name, bbox_inches="tight", facecolor="white")
    print(f"  -> {FIGURES_FOLDER / file_name}")


def draw_grouped_bars(
    title: str,
    metric_names: list[str],
    values_by_series: dict[str, dict[str, float]],
    colors_by_series: dict[str, str],
    file_name: str,
    y_axis_label: str = "Giá trị",
) -> None:
    """One group of bars per metric, one bar per series, each labelled with its value."""
    pyplot = import_pyplot_with_report_style()
    figure, axes = pyplot.subplots(figsize=(7.2, 3.8))
    bar_width = 0.8 / len(values_by_series)
    for series_index, (series_label, values) in enumerate(values_by_series.items()):
        offset = (series_index - (len(values_by_series) - 1) / 2) * bar_width
        bar_positions = [metric_index + offset for metric_index in range(len(metric_names))]
        heights = [values.get(metric, float("nan")) for metric in metric_names]
        bars = axes.bar(
            bar_positions, heights, bar_width * 0.92, label=series_label, color=colors_by_series[series_label]
        )
        for bar, height in zip(bars, heights):
            if not math.isnan(height):
                axes.annotate(
                    f"{height:.3f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="#52514e",
                    xytext=(0, 2),
                    textcoords="offset points",
                )
    axes.set_xticks(range(len(metric_names)), metric_names)
    axes.set_ylim(0, 1.08)
    axes.set_ylabel(y_axis_label)
    axes.set_title(title, loc="left", fontsize=11)
    axes.legend(
        frameon=False,
        fontsize=8,
        ncols=min(len(values_by_series), 4),
        loc="upper left",
        bbox_to_anchor=(0, -0.1),
    )
    axes.grid(axis="x", visible=False)
    save_figure(figure, file_name)
