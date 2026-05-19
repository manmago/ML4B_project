from __future__ import annotations

import argparse
from pathlib import Path

from ml4b.config import DEFAULT_STEP_SECONDS, DEFAULT_WINDOW_SECONDS, MODELS_DIR, PROCESSED_DIR, RAW_DIR, SLEEPDATA_DIR
from ml4b.io import discover_night_dirs
from ml4b.model import load_model_bundle
from ml4b.pipeline import build_workspace_dataset, load_or_build_night_frame, predict_hypnogram_for_night, preprocess_workspace, train_workspace_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML4B sleep classification utilities")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build the labeled dataset and feature table")
    build_parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    build_parser.add_argument("--sleepdata-dir", type=Path, default=SLEEPDATA_DIR)
    build_parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "feature_dataset.csv")
    build_parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    build_parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)

    train_parser = subparsers.add_parser("train", help="Train and save a sleep model")
    train_parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    train_parser.add_argument("--sleepdata-dir", type=Path, default=SLEEPDATA_DIR)
    train_parser.add_argument("--model-path", type=Path, default=MODELS_DIR / f"sleep_model_w{DEFAULT_WINDOW_SECONDS}_s{DEFAULT_STEP_SECONDS}.joblib")
    train_parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    train_parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)
    train_parser.add_argument("--use-cache", action="store_true", default=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Build and cache labeled night frames and features")
    preprocess_parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    preprocess_parser.add_argument("--sleepdata-dir", type=Path, default=SLEEPDATA_DIR)
    preprocess_parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    preprocess_parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)
    preprocess_parser.add_argument("--force-rebuild", action="store_true", default=False)

    predict_parser = subparsers.add_parser("predict", help="Predict a hypnogram for a single night")
    predict_parser.add_argument("--night-id", type=str, default=None)
    predict_parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    predict_parser.add_argument("--sleepdata-dir", type=Path, default=SLEEPDATA_DIR)
    predict_parser.add_argument("--model-path", type=Path, default=MODELS_DIR / f"sleep_model_w{DEFAULT_WINDOW_SECONDS}_s{DEFAULT_STEP_SECONDS}.joblib")
    predict_parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    predict_parser.add_argument("--step-seconds", type=int, default=DEFAULT_STEP_SECONDS)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        result = build_workspace_dataset(
            raw_dir=args.raw_dir,
            sleepdata_dir=args.sleepdata_dir,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.feature_frame.to_csv(args.output, index=False)
        print(f"Built dataset with {len(result.night_bundles)} nights and {len(result.feature_frame)} windows")
        print(f"Saved to {args.output}")
        return

    if args.command == "train":
        result, model_bundle = train_workspace_model(
            raw_dir=args.raw_dir,
            sleepdata_dir=args.sleepdata_dir,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            model_path=args.model_path,
            use_cache=args.use_cache,
        )
        print("Model trained")
        print(model_bundle.metrics)
        print(f"Windows: {len(result.feature_frame)}")
        print(f"Model saved to {args.model_path}")
        return

    if args.command == "preprocess":
        result = preprocess_workspace(
            raw_dir=args.raw_dir,
            sleepdata_dir=args.sleepdata_dir,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            force_rebuild=args.force_rebuild,
        )
        print(f"Preprocessed nights: {len(result.night_bundles)}")
        print(f"Feature windows cached: {len(result.feature_frame)}")
        return

    if args.command == "predict":
        if not args.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {args.model_path}")

        model_bundle = load_model_bundle(args.model_path)
        if args.night_id is None:
            night_dirs = discover_night_dirs(args.raw_dir)
            if not night_dirs:
                raise ValueError("No night folders were found.")
            night_bundle = load_or_build_night_frame(night_dirs[0].name, raw_dir=args.raw_dir, sleepdata_dir=args.sleepdata_dir)
        else:
            available = {path.name for path in discover_night_dirs(args.raw_dir)}
            if args.night_id not in available:
                raise ValueError(f"Night not found: {args.night_id}. Available: {', '.join(sorted(available))}")
            night_bundle = load_or_build_night_frame(args.night_id, raw_dir=args.raw_dir, sleepdata_dir=args.sleepdata_dir)

        prediction = predict_hypnogram_for_night(night_bundle.frame, model_bundle, window_seconds=args.window_seconds, step_seconds=args.step_seconds)
        print(prediction[["window_start", "window_end", "predicted_label", "sleep_probability"]].head(20).to_string(index=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
