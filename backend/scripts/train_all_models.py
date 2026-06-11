"""Train multiclass DR-grade and binary referable screening models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from train import MIN_OPTUNA_TRIALS, train_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multiclass DR grading and binary referable screening models.",
    )
    parser.add_argument("--features-csv", type=Path, default=config.FEATURES_CSV)
    parser.add_argument("--trials", type=int, default=MIN_OPTUNA_TRIALS)
    parser.add_argument(
        "--skip-multiclass",
        action="store_true",
        help="Train only the binary referable screening model.",
    )
    parser.add_argument(
        "--skip-binary",
        action="store_true",
        help="Train only the multiclass stage model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed Optuna trial CSVs if a previous training run was interrupted.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a short validation run with fewer than 50 Optuna trials.",
    )
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Skip slow SHAP/permutation artifacts while still exporting models and metrics.",
    )
    args = parser.parse_args()

    if not args.skip_multiclass:
        print("Training multiclass DR grade model (classes 0-4)...", flush=True)
        train_models(
            features_csv=args.features_csv,
            results_dir=config.RESULTS_DIR,
            n_trials=args.trials,
            binary_referable=False,
            resume_completed=args.resume,
            smoke=args.smoke,
            skip_interpretability=args.skip_interpretability,
        )

    if not args.skip_binary:
        print("Training binary referable screening model...", flush=True)
        train_models(
            features_csv=args.features_csv,
            results_dir=config.RESULTS_DIR / "binary",
            n_trials=args.trials,
            binary_referable=True,
            resume_completed=args.resume,
            smoke=args.smoke,
            skip_interpretability=args.skip_interpretability,
        )


if __name__ == "__main__":
    main()
