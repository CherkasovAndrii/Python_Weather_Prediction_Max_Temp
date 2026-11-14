"""Feed-forward neural network with Monte Carlo dropout for uncertainty estimates."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow C++ info logs

import keras
import numpy as np
import pandas as pd

from maxtemp.config import ModelConfig


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible runs."""
    keras.utils.set_random_seed(seed)


def build_mlp(X_train: pd.DataFrame | np.ndarray, cfg: ModelConfig) -> keras.Model:
    """Create and compile the network.

    Feature scaling is part of the model (a ``Normalization`` layer adapted on the
    training data only), so the saved model can be applied to raw observations.
    """
    X_train = np.asarray(X_train, dtype="float32")
    normalizer = keras.layers.Normalization(name="normalize")
    normalizer.adapt(X_train)

    inputs = keras.Input(shape=(X_train.shape[1],), name="observations")
    x = normalizer(inputs)
    regularizer = keras.regularizers.L2(cfg.l2) if cfg.l2 > 0 else None
    for i, units in enumerate(cfg.hidden_units, start=1):
        x = keras.layers.Dense(
            units, activation="relu", kernel_regularizer=regularizer, name=f"dense_{i}"
        )(x)
        if cfg.dropout_rate > 0 and i < len(cfg.hidden_units):
            x = keras.layers.Dropout(cfg.dropout_rate, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(1, name="temp_max")(x)

    model = keras.Model(inputs, outputs, name="max_temp_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def train_mlp(
    model: keras.Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: ModelConfig,
    verbose: int = 0,
) -> dict[str, list[float]]:
    """Fit with early stopping on the validation loss; restores the best weights."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    history = model.fit(
        np.asarray(X_train, dtype="float32"),
        np.asarray(y_train, dtype="float32"),
        validation_data=(np.asarray(X_val, dtype="float32"), np.asarray(y_val, dtype="float32")),
        batch_size=cfg.batch_size,
        epochs=cfg.max_epochs,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}


def predict(model: keras.Model, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Deterministic prediction (dropout disabled)."""
    return model.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()


def mc_dropout_predict(
    model: keras.Model, X: pd.DataFrame | np.ndarray, n_samples: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo dropout: run the network ``n_samples`` times with dropout active.

    Returns the mean prediction and its standard deviation, which serves as an
    estimate of the model's uncertainty for every single forecast.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    X = np.asarray(X, dtype="float32")
    samples = np.stack(
        [keras.ops.convert_to_numpy(model(X, training=True)).ravel() for _ in range(n_samples)]
    )
    return samples.mean(axis=0), samples.std(axis=0)
