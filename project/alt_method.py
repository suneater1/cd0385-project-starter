"""End to end workflow for the Kaggle Bike Sharing competition.

The script installs required packages, downloads the dataset using the Kaggle
API and trains an AutoGluon model with extensive feature engineering.  The
number of hyperparameter optimisation trials and the training time limit can be
configured via command line arguments.
"""

import argparse
import os
import subprocess
import sys
import zipfile

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor


def ensure_packages() -> None:
    """Install required Python packages if they are missing."""
    required = ["pandas", "numpy", "kaggle", "autogluon.tabular"]
    for pkg in required:
        module = pkg.split(".")[0]
        try:
            __import__(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def ensure_kaggle_dataset(dataset_dir: str = "project") -> None:
    """Download and extract dataset using the Kaggle API if not present."""
    train_path = os.path.join(dataset_dir, "train.csv")
    if os.path.exists(train_path):
        return

    kaggle_dir = os.path.expanduser("~/.kaggle")
    cred_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(cred_path):
        token = os.getenv("KAGGLE_JSON")
        if token:
            os.makedirs(kaggle_dir, exist_ok=True)
            with open(cred_path, "w") as f:
                f.write(token)
            os.chmod(cred_path, 0o600)
        else:
            raise RuntimeError(
                "Kaggle credentials not found. Set KAGGLE_JSON env variable or place kaggle.json in ~/.kaggle"
            )

    subprocess.check_call(
        ["kaggle", "competitions", "download", "-c", "bike-sharing-demand", "-p", dataset_dir]
    )

    for file in os.listdir(dataset_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(dataset_dir, file)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dataset_dir)
            os.remove(zip_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional time and weather based features."""
    df["hour"] = df["datetime"].dt.hour.astype("category")
    df["dayofweek"] = df["datetime"].dt.dayofweek.astype("category")
    df["month"] = df["datetime"].dt.month.astype("category")
    df["year"] = df["datetime"].dt.year.astype("category")

    def hour_bucket(h: int) -> str:
        if 7 <= h <= 9:
            return "morning_rush"
        elif 11 <= h <= 13:
            return "lunch"
        elif 17 <= h <= 19:
            return "evening_rush"
        elif 22 <= h <= 23 or 0 <= h <= 5:
            return "night"
        return "off_peak"

    df["hour_bucket"] = df["hour"].astype(int).apply(hour_bucket).astype("category")

    def temp_bin(t: float) -> str:
        if t < 10:
            return "cold"
        if t < 25:
            return "mild"
        return "hot"

    df["temp_bin"] = df["temp"].apply(temp_bin).astype("category")
    df["wind_type"] = pd.cut(df["windspeed"], bins=[-1, 1, 15, 100], labels=["calm", "breezy", "windy"]).astype("category")
    df["humidity_level"] = pd.cut(df["humidity"], bins=[-1, 30, 70, 100], labels=["low", "normal", "high"]).astype("category")
    df["is_weekend"] = df["dayofweek"].astype(int).isin([5, 6]).astype("int")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AutoGluon model for Bike Sharing")
    parser.add_argument("--num_trials", type=int, default=25, help="Number of HPO trials")
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit for training in seconds (default: 1 hour)",
    )
    args = parser.parse_args()

    ensure_packages()
    ensure_kaggle_dataset()

    train = pd.read_csv("project/train.csv", parse_dates=["datetime"])
    test = pd.read_csv("project/test.csv", parse_dates=["datetime"])

    train = engineer_features(train)
    test = engineer_features(test)

    train["count_log"] = np.log1p(train["count"])

    train_sorted = train.sort_values("datetime")
    cutoff = int(len(train_sorted) * 0.8)
    train_data = train_sorted.iloc[:cutoff]
    val_data = train_sorted.iloc[cutoff:]

    drop_cols = ["casual", "registered", "count", "count_log"]
    features = [c for c in train.columns if c not in drop_cols]

    predictor = TabularPredictor(label="count_log", eval_metric="root_mean_squared_error").fit(
        train_data[features + ["count_log"]],
        tuning_data=val_data[features + ["count_log"]],
        presets="best_quality",
        num_trials=args.num_trials,
        time_limit=args.time_limit,
        dynamic_stacking=True,
    )

    val_pred = np.expm1(predictor.predict(val_data[features])).clip(lower=0)
    rmsle = np.sqrt(np.mean(np.square(np.log1p(val_data["count"]) - np.log1p(val_pred))))
    print(f"Validation RMSLE: {rmsle:.5f}")

    test_pred = np.expm1(predictor.predict(test[features])).clip(lower=0)
    submission = pd.read_csv("project/sampleSubmission.csv")
    submission["count"] = test_pred
    submission.to_csv("project/alt_submission.csv", index=False)


if __name__ == "__main__":
    main()
