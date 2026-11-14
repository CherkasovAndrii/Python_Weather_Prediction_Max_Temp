"""Experiment configuration.

All tunable values live here so that an experiment is fully described by one object
and can be reproduced from the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_PATH = Path("data/weather_prediction_dataset.csv")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_CITIES = ("ROMA", "BUDAPEST", "STOCKHOLM")


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split boundaries (inclusive end dates of each period).

    The dataset covers 2000-01-01 .. 2010-01-01. Splitting by time instead of at
    random prevents the model from "seeing the future" during training.
    """

    train_end: str = "2006-12-31"
    val_end: str = "2007-12-31"


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters of the neural network and its training loop."""

    hidden_units: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.1
    l2: float = 1e-3
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 30
    mc_samples: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce one run."""

    cities: tuple[str, ...] = DEFAULT_CITIES
    horizon_days: int = 1
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
