"""Command-line interface: ``maxtemp train`` and ``maxtemp cities``."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from maxtemp import __version__
from maxtemp.config import (
    DEFAULT_CITIES,
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_REPORTS_DIR,
    ExperimentConfig,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maxtemp",
        description="Forecast tomorrow's maximum temperature for European cities.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA_PATH, help="path to the dataset CSV"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cities", help="list the cities available in the dataset")

    train = sub.add_parser("train", help="train and evaluate models, write reports")
    train.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
        metavar="CITY",
        help="cities to model (default: %(default)s)",
    )
    train.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    train.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    train.add_argument("--no-save-models", action="store_true", help="do not save .keras files")
    train.add_argument("--epochs", type=int, help="maximum number of training epochs")
    train.add_argument("--seed", type=int, help="random seed")
    train.add_argument("--mc-samples", type=int, help="Monte Carlo dropout passes")
    return parser


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(cities=tuple(c.upper() for c in args.cities))
    model_overrides = {
        k: v
        for k, v in {"max_epochs": args.epochs, "mc_samples": args.mc_samples}.items()
        if v is not None
    }
    if model_overrides:
        cfg = replace(cfg, model=replace(cfg.model, **model_overrides))
    if args.seed is not None:
        cfg = replace(cfg, seed=args.seed)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "cities":
        from maxtemp.data import available_cities, load_dataset

        print("\n".join(available_cities(load_dataset(args.data))))
        return 0

    from maxtemp.pipeline import results_table, run_experiment

    report = run_experiment(
        _config_from_args(args),
        data_path=args.data,
        reports_dir=args.reports_dir,
        models_dir=None if args.no_save_models else args.models_dir,
    )
    print("\n" + results_table(report["results"]))
    print(f"Reports written to {args.reports_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
