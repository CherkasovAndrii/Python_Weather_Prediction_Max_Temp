"""Evaluation metrics for regression forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE and MAE in °C, plus the coefficient of determination R²."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def interval_coverage(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, z: float = 1.96
) -> float:
    """Share of observations that fall inside ``mean ± z·std``.

    For a well-calibrated 95 % interval (z = 1.96) this should be close to 0.95.
    """
    y_true, mean, std = (np.asarray(a, dtype=float) for a in (y_true, mean, std))
    inside = np.abs(y_true - mean) <= z * std
    return float(inside.mean())
