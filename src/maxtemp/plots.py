"""Figures for the report. Every function saves one PNG and returns its path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"

# One fixed colour per model, used identically in every chart.
MODEL_COLORS = {"mlp": "#2a78d6", "ridge": "#eb6834", "persistence": "#1baf7a"}
MODEL_LABELS = {
    "mlp": "Neural network (MLP)",
    "ridge": "Ridge regression",
    "persistence": "Persistence",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 10,
    }
)


def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning_curve(history: Mapping[str, list[float]], city: str, path: Path) -> Path:
    """Training vs. validation RMSE per epoch, with the restored best epoch marked."""
    train, val = np.asarray(history["rmse"]), np.asarray(history["val_rmse"])
    epochs = np.arange(1, len(train) + 1)
    best = int(np.argmin(history["val_loss"])) + 1

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, color=MODEL_COLORS["mlp"], linewidth=2, label="Training (dropout on)")
    ax.plot(epochs, val, color=MODEL_COLORS["ridge"], linewidth=2, label="Validation")
    ax.axvline(best, color=INK_MUTED, linestyle=":", linewidth=1.2)
    # Skip the first epochs (huge initial error) so the interesting range is visible.
    skip = min(3, len(train) - 1)
    ax.set_ylim(0, float(np.max(np.r_[train[skip:], val[skip:]])) * 1.15)
    ax.annotate(
        f"best epoch {best} (weights restored)",
        xy=(best, 1),
        xycoords=("data", "axes fraction"),
        xytext=(-6 if best > 0.6 * len(train) else 6, -14),
        textcoords="offset points",
        ha="right" if best > 0.6 * len(train) else "left",
        color=INK_SECONDARY,
        fontsize=9,
    )
    ax.set(xlabel="Epoch", ylabel="RMSE, °C", title=f"{city.title()} — learning curve")
    ax.legend(loc="lower left")
    _style(ax)
    return _save(fig, path)


def plot_predicted_vs_true(
    y_true: np.ndarray, y_pred: np.ndarray, city: str, rmse: float, path: Path
) -> Path:
    """Scatter of forecasts against observations on the test set."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo = float(min(np.min(y_true), np.min(y_pred))) - 2
    hi = float(max(np.max(y_true), np.max(y_pred))) + 2
    ax.plot(
        [lo, hi], [lo, hi], color=INK_MUTED, linestyle="--", linewidth=1, label="Perfect forecast"
    )
    ax.scatter(y_pred, y_true, s=12, alpha=0.45, color=MODEL_COLORS["mlp"], linewidths=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal")
    ax.set(
        xlabel="Predicted max temperature, °C",
        ylabel="Observed max temperature, °C",
        title=f"{city.title()} — test set (RMSE {rmse:.2f} °C)",
    )
    ax.legend(loc="upper left")
    _style(ax, grid_axis="both")
    return _save(fig, path)


def plot_forecast_window(
    dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    city: str,
    path: Path,
    last_n_days: int = 120,
) -> Path:
    """Observed vs. forecast for the last days of the test period with a 95 % band."""
    sl = slice(-last_n_days, None)
    d, t, m, s = dates[sl], np.asarray(y_true)[sl], np.asarray(mean)[sl], np.asarray(std)[sl]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(
        d,
        m - 1.96 * s,
        m + 1.96 * s,
        color=MODEL_COLORS["mlp"],
        alpha=0.18,
        linewidth=0,
        label="Model uncertainty ±1.96 std (MC dropout)",
    )
    ax.plot(d, t, color=INK, linewidth=1.6, label="Observed")
    ax.plot(d, m, color=MODEL_COLORS["mlp"], linewidth=2, label="Forecast")
    ax.set(
        ylabel="Max temperature, °C",
        title=f"{city.title()} — next-day forecast, last {last_n_days} test days",
    )
    lo, hi = (
        float(np.min([t.min(), (m - 1.96 * s).min()])),
        float(np.max([t.max(), (m + 1.96 * s).max()])),
    )
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))  # headroom for the legend
    ax.legend(loc="upper left", ncols=3)
    _style(ax)
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_model_comparison(
    results: Mapping[str, Mapping[str, Mapping[str, float]]], path: Path
) -> Path:
    """Grouped bars: test RMSE for every city and model (lower is better)."""
    cities = list(results)
    models = [m for m in MODEL_COLORS if all(m in results[c] for c in cities)]
    x = np.arange(len(cities))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model in enumerate(models):
        values = [results[c][model]["rmse"] for c in cities]
        offsets = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, color=INK_SECONDARY, fontsize=8.5)
    ax.set_xticks(x, [c.title() for c in cities])
    ax.set(
        ylabel="Test RMSE, °C (lower is better)",
        title="Next-day max temperature — model comparison",
    )
    ax.margins(y=0.22)
    ax.legend(loc="upper left", ncols=len(models))
    _style(ax)
    ax.tick_params(axis="x", length=0)
    return _save(fig, path)
