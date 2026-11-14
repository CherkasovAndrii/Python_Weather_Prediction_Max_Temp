"""End-to-end experiment: data -> baselines + neural network -> metrics and figures."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maxtemp import __version__
from maxtemp.baselines import PersistenceForecast, build_ridge
from maxtemp.config import ExperimentConfig
from maxtemp.data import (
    chronological_split,
    clean_dataset,
    load_dataset,
    make_supervised,
    target_column,
)
from maxtemp.metrics import interval_coverage, regression_metrics
from maxtemp.plots import (
    plot_forecast_window,
    plot_learning_curve,
    plot_model_comparison,
    plot_predicted_vs_true,
)

logger = logging.getLogger(__name__)


def run_city(
    df: pd.DataFrame,
    city: str,
    cfg: ExperimentConfig,
    reports_dir: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate all models for one city; return its metrics."""
    from maxtemp import nn  # imported lazily: TensorFlow is slow to load

    city = city.upper()
    figures_dir = reports_dir / "figures"
    X, y = make_supervised(df, city, cfg.horizon_days)
    split = chronological_split(X, y, cfg.split.train_end, cfg.split.val_end)
    logger.info("%s: %d features, split sizes %s", city, X.shape[1], split.sizes())

    results: dict[str, Any] = {"split_sizes": split.sizes(), "n_features": int(X.shape[1])}

    persistence = PersistenceForecast(target_column(df, city)).fit(split.X_train)
    results["persistence"] = regression_metrics(split.y_test, persistence.predict(split.X_test))

    ridge = build_ridge().fit(split.X_train, split.y_train)
    results["ridge"] = regression_metrics(split.y_test, ridge.predict(split.X_test))

    nn.set_seed(cfg.seed)
    model = nn.build_mlp(split.X_train, cfg.model)
    history = nn.train_mlp(model, split.X_train, split.y_train, split.X_val, split.y_val, cfg.model)
    mean, std = nn.mc_dropout_predict(model, split.X_test, cfg.model.mc_samples)
    results["mlp"] = regression_metrics(split.y_test, mean) | {
        "interval_95_coverage": interval_coverage(split.y_test, mean, std),
        "mean_uncertainty_std": float(np.mean(std)),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "epochs_trained": len(history["loss"]),
    }
    logger.info(
        "%s test RMSE — persistence %.2f | ridge %.2f | mlp %.2f",
        city,
        results["persistence"]["rmse"],
        results["ridge"]["rmse"],
        results["mlp"]["rmse"],
    )

    slug = city.lower()
    plot_learning_curve(history, city, figures_dir / f"{slug}_learning_curve.png")
    plot_predicted_vs_true(
        split.y_test.to_numpy(),
        mean,
        city,
        results["mlp"]["rmse"],
        figures_dir / f"{slug}_test_scatter.png",
    )
    plot_forecast_window(
        split.y_test.index,
        split.y_test.to_numpy(),
        mean,
        std,
        city,
        figures_dir / f"{slug}_forecast.png",
    )

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        model.save(models_dir / f"{slug}_mlp.keras")

    return results


def results_table(results: dict[str, dict[str, Any]]) -> str:
    """Render test metrics as a Markdown table."""
    lines = [
        "| City | Model | RMSE, °C | MAE, °C | R² |",
        "|------|-------|---------:|--------:|---:|",
    ]
    names = {"persistence": "Persistence", "ridge": "Ridge regression", "mlp": "Neural network"}
    for city, res in results.items():
        for key, name in names.items():
            m = res[key]
            lines.append(
                f"| {city.title()} | {name} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['r2']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def run_experiment(
    cfg: ExperimentConfig,
    data_path: Path,
    reports_dir: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Run every configured city and write ``metrics.json``, ``results.md`` and figures."""
    df, fixed = clean_dataset(load_dataset(data_path))
    if fixed:
        logger.info("Replaced %d invalid values: %s", sum(fixed.values()), fixed)
    per_city = {
        city.upper(): run_city(df, city, cfg, reports_dir, models_dir) for city in cfg.cities
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(per_city, reports_dir / "figures" / "model_comparison.png")

    report = {
        "version": __version__,
        "config": asdict(cfg),
        "data_cleaning": {"invalid_values_replaced": fixed},
        "results": per_city,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports_dir / "results.md").write_text(results_table(per_city), encoding="utf-8")
    return report
