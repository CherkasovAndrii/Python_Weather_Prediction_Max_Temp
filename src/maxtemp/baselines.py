"""Simple reference models. A neural network is only useful if it beats these."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class PersistenceForecast:
    """Naive forecast: tomorrow's maximum temperature equals today's."""

    def __init__(self, today_column: str) -> None:
        self.today_column = today_column

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> PersistenceForecast:
        if self.today_column not in X.columns:
            raise KeyError(f"Column '{self.today_column}' not found in features")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.today_column].to_numpy(dtype=float)


def build_ridge() -> Pipeline:
    """Standardised linear regression with the L2 penalty chosen by cross-validation."""
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
