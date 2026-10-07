"""TFT ensemble inference used by the final submission entrypoint."""

from __future__ import annotations

import json
import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

os.environ.setdefault("LIGHTNING_DISABLE_APP_TIPS", "1")

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.exceptions import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*Attribute 'loss' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=".*Attribute 'logging_metrics' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", message=".*predict_dataloader.*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*`isinstance\\(treespec, LeafSpec\\)` is deprecated.*")

for logger_name in ("lightning", "lightning.pytorch", "lightning_fabric", "pytorch_lightning"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


DOMAIN_FEAT_COLS = [
    "workload_intensity",
    "demand_forecast",
    "upstream_quality_forecast",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
]
VELOCITY_COLS = ["queue_pressure_forecast", "network_pressure_forecast", "shock_risk"]
TIME_REAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
TIME_CAT_COLS = ["hour_cat", "dow_cat"]

EVAL_HORIZON = 336
MIN_ENCODER = 168
MAX_ENCODER = 336
RANDOMIZE_LENGTH = (0.2, 0.05)
STRIDE = 24
TRAIN_FILE = "train.csv"
TEST_INPUT_FILE = "test_input.csv"
TEST_INDEX_FILE = "forecast_index_test.csv"
VAL_INPUT_FILE = "validation_input.csv"
VAL_INDEX_FILE = "forecast_index_validation.csv"

BATCH_SIZE = 32
EVAL_BATCH = 32
N_WORKERS = 0


@contextmanager
def quiet_library_output():
    if os.environ.get("SUBMISSION_VERBOSE"):
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def resolve_input_files(input_dir: Path) -> tuple[Path, Path, Path]:
    train_path = input_dir / TRAIN_FILE
    if not train_path.exists():
        raise FileNotFoundError(f"Expected {TRAIN_FILE} in {input_dir}")

    test_input = input_dir / TEST_INPUT_FILE
    test_index = input_dir / TEST_INDEX_FILE
    val_input = input_dir / VAL_INPUT_FILE
    val_index = input_dir / VAL_INDEX_FILE

    if test_input.exists() and test_index.exists():
        return train_path, test_input, test_index
    if val_input.exists() and val_index.exists():
        return train_path, val_input, val_index

    raise FileNotFoundError(
        "Expected either test_input.csv + forecast_index_test.csv "
        "or validation_input.csv + forecast_index_validation.csv in the input directory."
    )


def impute_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["series_id", "timestamp"]).copy()
    df[DOMAIN_FEAT_COLS] = df.groupby("series_id", group_keys=False)[DOMAIN_FEAT_COLS].transform(
        lambda x: x.ffill().bfill()
    )
    return df


def add_time_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    t = pd.to_datetime(df[time_col])
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(float)
    df["hour_cat"] = t.dt.hour.astype(str)
    df["dow_cat"] = t.dt.dayofweek.astype(str)
    return df


def validate_columns(name: str, frame: pd.DataFrame, *, target: bool, covariates: bool) -> None:
    required = {"series_id", "timestamp"}
    if target:
        required.add("target")
    if covariates:
        required.update(DOMAIN_FEAT_COLS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_panel(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    train_path, future_path, index_path = resolve_input_files(input_dir)
    train_df = pd.read_csv(train_path, parse_dates=["timestamp"])
    future_df = pd.read_csv(future_path, parse_dates=["timestamp"])
    forecast_index = pd.read_csv(index_path, parse_dates=["timestamp"])

    validate_columns(train_path.name, train_df, target=True, covariates=True)
    validate_columns(future_path.name, future_df, target=False, covariates=True)
    validate_columns(index_path.name, forecast_index, target=False, covariates=False)
    if "target" in future_df.columns:
        raise ValueError(f"{future_path.name} must not contain target values.")

    train_df = add_time_features(impute_domain_features(train_df))
    future_df = add_time_features(impute_domain_features(future_df))

    train_df["is_future"] = False
    future_df["is_future"] = True
    future_df["target"] = 0.0

    panel = pd.concat([train_df, future_df], ignore_index=True)
    panel = panel.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    t0 = panel["timestamp"].min()
    panel["time_idx"] = ((panel["timestamp"] - t0).dt.total_seconds() // 3600).astype(int)

    for col in VELOCITY_COLS:
        panel[f"{col}_vel"] = panel.groupby("series_id")[col].diff().fillna(0.0)

    panel["series_id"] = panel["series_id"].astype(str)
    panel["target"] = panel["target"].astype(np.float32)
    forecast_index["series_id"] = forecast_index["series_id"].astype(str)

    t_train = int(panel.loc[~panel.is_future, "time_idx"].max()) + 1
    n_series = panel["series_id"].nunique()

    counts = panel.groupby("series_id").size()
    if counts.nunique() != 1:
        raise ValueError(f"Expected a rectangular hourly panel; got ragged series lengths:\n{counts.describe()}")
    gaps = panel.groupby("series_id")["time_idx"].apply(lambda s: s.diff().dropna().ne(1).sum())
    if int(gaps.sum()) != 0:
        raise ValueError("Input panel is not a contiguous hourly grid.")
    if panel[DOMAIN_FEAT_COLS].isna().sum().sum() != 0:
        raise ValueError("NaNs survived domain feature imputation.")

    return panel, forecast_index, t_train, n_series


def build_reference_ds(panel: pd.DataFrame, t_train: int) -> TimeSeriesDataSet:
    known_reals = ["time_idx"] + DOMAIN_FEAT_COLS + [f"{c}_vel" for c in VELOCITY_COLS] + TIME_REAL_COLS
    ds = TimeSeriesDataSet(
        panel[panel.time_idx < t_train],
        time_idx="time_idx",
        target="target",
        group_ids=["series_id"],
        min_encoder_length=MIN_ENCODER,
        max_encoder_length=MAX_ENCODER,
        min_prediction_length=EVAL_HORIZON,
        max_prediction_length=EVAL_HORIZON,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=TIME_CAT_COLS,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        randomize_length=RANDOMIZE_LENGTH,
        allow_missing_timesteps=False,
    )

    if STRIDE and STRIDE > 1:
        decoded = ds.decoded_index.reset_index(drop=True)
        sequence_length = ds.index["sequence_length"].to_numpy()
        max_start = int(decoded.time_idx_first_prediction.max())
        keyed = pd.DataFrame(
            {
                "g": decoded["series_id"].astype(str),
                "t": decoded["time_idx_first_prediction"].to_numpy(),
                "s": sequence_length,
            }
        )
        longest = keyed.groupby(["g", "t"])["s"].transform("max").to_numpy() == sequence_length
        on_grid = ((max_start - keyed["t"].to_numpy()) % STRIDE) == 0
        selected = keyed.loc[longest & on_grid]
        keep = np.zeros(len(keyed), dtype=bool)
        keep[selected.index[~selected.duplicated(["g", "t"])]] = True
        ds = ds.filter(lambda _idx: keep)

    last_target = int((ds.decoded_index.time_idx_first_prediction + EVAL_HORIZON - 1).max())
    if last_target != t_train - 1:
        raise ValueError(f"Reference windows stop at target {last_target}, expected {t_train - 1}.")
    return ds


def build_prediction_ds(reference_ds: TimeSeriesDataSet, panel: pd.DataFrame, t_train: int, n_series: int) -> TimeSeriesDataSet:
    prediction_ds = TimeSeriesDataSet.from_dataset(
        reference_ds,
        panel[panel.time_idx < t_train + EVAL_HORIZON],
        predict=True,
        stop_randomization=True,
    )
    starts = prediction_ds.decoded_index.time_idx_first_prediction.unique()
    if len(starts) != 1 or int(starts[0]) != t_train:
        raise ValueError(f"Prediction window is misaligned: starts={starts}, expected={t_train}")
    if len(prediction_ds) != n_series:
        raise ValueError(f"Expected {n_series} prediction windows, got {len(prediction_ds)}")
    return prediction_ds


def to_dataloader(ds: TimeSeriesDataSet, train: bool = False):
    return ds.to_dataloader(
        train=train,
        batch_size=BATCH_SIZE if train else EVAL_BATCH,
        num_workers=N_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def predictions_to_frame(preds: np.ndarray, index: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    horizon = preds.shape[1]
    frames = []
    for i in range(len(index)):
        start = int(index["time_idx"].iloc[i])
        frames.append(
            pd.DataFrame(
                {
                    "series_id": str(index["series_id"].iloc[i]),
                    "time_idx": np.arange(start, start + horizon),
                    "prediction": preds[i].astype(np.float64),
                }
            )
        )
    long = pd.concat(frames, ignore_index=True)
    return long.merge(
        panel[["series_id", "time_idx", "timestamp"]],
        on=["series_id", "time_idx"],
        how="left",
    )[["series_id", "timestamp", "prediction"]]


def load_checkpoint_bundle(checkpoint_path: Path) -> list[Path]:
    try:
        bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        bundle = json.loads(checkpoint_path.read_text())
    if not isinstance(bundle, dict) or "checkpoint_files" not in bundle:
        raise ValueError("checkpoint.pt must contain a dict with a 'checkpoint_files' list.")
    base = checkpoint_path.parent
    paths = [(base / rel).resolve() for rel in bundle["checkpoint_files"]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model checkpoint files referenced by checkpoint.pt: {missing}")
    return paths


def predict(input_dir: Path, output_file: Path, checkpoint_file: Path) -> None:
    seed_everything(42, workers=True, verbose=False)
    panel, forecast_index, t_train, n_series = prepare_panel(input_dir)
    reference_ds = build_reference_ds(panel, t_train)
    prediction_ds = build_prediction_ds(reference_ds, panel, t_train, n_series)
    loader = to_dataloader(prediction_ds, train=False)
    checkpoint_paths = load_checkpoint_bundle(checkpoint_file)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    seed_predictions = []
    prediction_index = None
    for ckpt in checkpoint_paths:
        model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt))
        with quiet_library_output():
            out = model.predict(
                loader,
                mode="prediction",
                return_index=True,
                trainer_kwargs={
                    "accelerator": accelerator,
                    "devices": 1,
                    "precision": precision,
                    "logger": False,
                    "enable_progress_bar": False,
                },
            )
        seed_predictions.append(out.output.cpu().float().numpy())
        prediction_index = out.index

    averaged = np.mean(seed_predictions, axis=0)
    predictions = predictions_to_frame(averaged, prediction_index, panel)
    predictions = forecast_index[["series_id", "timestamp"]].merge(
        predictions,
        on=["series_id", "timestamp"],
        how="left",
    )

    if predictions["prediction"].isna().any():
        missing = int(predictions["prediction"].isna().sum())
        raise ValueError(f"{missing} forecast-index rows did not receive predictions.")
    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError("Predictions contain non-finite values.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions[["series_id", "timestamp", "prediction"]].to_csv(output_file, index=False)
