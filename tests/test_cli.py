import json

import pytest

from maxtemp.cli import main


def test_cities_command(real_csv, capsys):
    assert main(["--data", str(real_csv), "cities"]) == 0
    out = capsys.readouterr().out.split()
    assert "ROMA" in out
    assert "DE_BILT" in out
    assert len(out) == 18


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["fly"])


@pytest.mark.slow
def test_train_end_to_end(synthetic_csv, tmp_path):
    pytest.importorskip("tensorflow")
    reports, models = tmp_path / "reports", tmp_path / "models"
    code = main(
        [
            "--data", str(synthetic_csv),
            "train", "--cities", "roma", "oslo",
            "--epochs", "3", "--mc-samples", "5",
            "--reports-dir", str(reports), "--models-dir", str(models),
        ]
    )  # fmt: skip
    assert code == 0

    report = json.loads((reports / "metrics.json").read_text())
    assert set(report["results"]) == {"ROMA", "OSLO"}
    for res in report["results"].values():
        assert {"persistence", "ridge", "mlp"} <= set(res)
        assert res["mlp"]["epochs_trained"] <= 3

    figures = {p.name for p in (reports / "figures").iterdir()}
    assert "model_comparison.png" in figures
    assert {"roma_learning_curve.png", "roma_test_scatter.png", "roma_forecast.png"} <= figures
    assert (models / "roma_mlp.keras").is_file()
    assert "| Roma | Neural network |" in (reports / "results.md").read_text()
