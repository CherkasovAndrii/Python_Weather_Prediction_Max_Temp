import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

from maxtemp import nn
from maxtemp.config import ModelConfig


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(256, 5)), columns=list("abcde"))
    y = pd.Series(3 * X["a"] - X["b"] + 0.1 * rng.normal(size=256))
    return X, y


def test_model_output_shape(toy_data):
    X, _ = toy_data
    model = nn.build_mlp(X, ModelConfig(hidden_units=(8, 4)))
    assert model.input_shape == (None, 5)
    assert nn.predict(model, X).shape == (len(X),)


def test_deterministic_prediction_is_repeatable(toy_data):
    X, _ = toy_data
    model = nn.build_mlp(X, ModelConfig(hidden_units=(8, 4), dropout_rate=0.5))
    np.testing.assert_allclose(nn.predict(model, X), nn.predict(model, X))


def test_mc_dropout_gives_uncertainty(toy_data):
    X, _ = toy_data
    model = nn.build_mlp(X, ModelConfig(hidden_units=(16, 8), dropout_rate=0.3))
    mean, std = nn.mc_dropout_predict(model, X, n_samples=20)
    assert mean.shape == std.shape == (len(X),)
    assert (std > 0).all()


def test_no_dropout_means_no_uncertainty(toy_data):
    X, _ = toy_data
    model = nn.build_mlp(X, ModelConfig(hidden_units=(8, 4), dropout_rate=0.0))
    _, std = nn.mc_dropout_predict(model, X, n_samples=5)
    np.testing.assert_allclose(std, 0.0, atol=1e-6)


def test_mc_dropout_requires_several_samples(toy_data):
    X, _ = toy_data
    model = nn.build_mlp(X, ModelConfig(hidden_units=(4,)))
    with pytest.raises(ValueError):
        nn.mc_dropout_predict(model, X, n_samples=1)


@pytest.mark.slow
def test_training_reduces_loss_and_is_reproducible(toy_data):
    X, y = toy_data
    cfg = ModelConfig(hidden_units=(16, 8), max_epochs=30, patience=30)

    def fit_once():
        nn.set_seed(123)
        model = nn.build_mlp(X, cfg)
        return nn.train_mlp(model, X[:200], y[:200], X[200:], y[200:], cfg)

    first, second = fit_once(), fit_once()
    assert first["val_loss"][-1] < first["val_loss"][0]
    np.testing.assert_allclose(first["val_loss"], second["val_loss"], rtol=1e-5)
