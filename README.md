# Deep Learning for Multivariate Time Series Forecasting

University group project on **multivariate time-series forecasting with known future covariates**. We compare traditional machine-learning and deep-learning approaches and develop a **Temporal Fusion Transformer (TFT)** for long-horizon forecasting across multiple related time series.

The main benchmark contains **96 hourly time series**, **11 known future covariates**, and a **336-hour forecast horizon**. We evaluate the TFT against naive baselines, LightGBM, and a gated MLP, and then test whether the same modeling approach transfers to a second dataset based on cow accelerometer measurements.

## Key Results

| Model | Benchmark WAPE | MAE | RMSE |
|---|---:|---:|---:|
| Naive last value | 48.10 | 5.29 | 6.97 |
| Seasonal mean | 34.41 | 3.79 | 5.21 |
| LightGBM | 14.68 | 1.62 | 3.06 |
| Gated MLP | 14.75 | 1.62 | 3.06 |
| **Temporal Fusion Transformer** | **14.48** | **1.59** | 3.08 |

The TFT achieved the best validation WAPE, although the small gap between the learned models suggests that the available known future covariates impose a strong **feature-set / information ceiling**.

On the transfer experiment using the Japanese Black Beef Cow Behavior dataset, the TFT reached a **WAPE of 43.0**, compared with **63.8** for the naive last-value baseline. Removing the known behaviour covariate increased WAPE to 44.5.

## Approach

The forecasting pipeline combines:

- exploratory data analysis and multivariate feature engineering
- known future domain covariates
- cyclical time features for hour, weekday, and day of year
- first-order differences for selected forecast variables
- per-series target normalization
- chronological rolling-origin validation
- Temporal Fusion Transformers with variable selection, gating, recurrent sequence processing, and attention
- a five-seed ensemble for the final benchmark predictions

The final TFT uses a hidden size of 96, four attention heads, one LSTM layer, and a dropout rate of 0.15. Models were trained with an encoder window of up to 336 hours and a fixed 336-hour prediction horizon.

## Repository Structure

```text
.
├── code/
│   ├── tft_final_model.ipynb          # Benchmark training, validation and final TFT ensemble
│   ├── tft_cow_model.ipynb            # Transfer experiment on cow accelerometer data
│   ├── predict.py                     # Standalone inference entry point
│   ├── checkpoint.pt                  # Ensemble checkpoint manifest
│   ├── checkpoints/                   # Five trained TFT model checkpoints
│   ├── src/
│   │   └── tft_inference.py           # Preprocessing and inference pipeline
│   ├── validation_tft_final_model.csv # Benchmark validation predictions
│   └── validation_tft_cow_model.csv   # Cow-dataset validation predictions
├── report/
│   ├── DLAM_Report.pdf                # Final project report
│   └── report.tex                     # LaTeX source of the report
├── requirements.txt
└── README.md
```

## Notebooks

### `tft_final_model.ipynb`

Contains the main benchmark workflow, including data preparation, feature engineering, model configuration, training, chronological validation, five-seed ensembling, and prediction generation.

The final ensemble uses seeds `42`, `7`, `13`, `99`, and `1234`.

### `tft_cow_model.ipynb`

Applies the same overall forecasting idea to the Japanese Black Beef Cow Behavior dataset. Tri-axial accelerometer data are converted into movement intensity and resampled to one-second intervals. The experiment predicts the next 30 seconds from 60–120 seconds of history and includes an ablation of the known behaviour covariate.

## Running the Inference Code

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run the inference script from the `code` directory:

```bash
cd code
python predict.py \
  --input_dir /path/to/input \
  --output_file /path/to/predictions.csv \
  --checkpoint checkpoint.pt
```

Expected benchmark input files are:

```text
train.csv
test_input.csv
forecast_index_test.csv
metadata.json
```

The generated output follows the schema:

```text
series_id,timestamp,prediction
```

## Reproducibility

The benchmark experiments were developed in Python using PyTorch, Lightning, PyTorch Forecasting, pandas, NumPy, and scikit-learn. The final benchmark training used two NVIDIA Tesla T4 GPUs.

The original course dataset is **not included in this repository**. The notebooks therefore require the corresponding input data to reproduce the benchmark training run. The cow dataset can be obtained from the public Japanese Black Beef Cow Behavior dataset referenced in the report.

## Report

The complete project report is available in [`report/DLAM_Report.pdf`](report/DLAM_Report.pdf), with the corresponding LaTeX source in [`report/report.tex`](report/report.tex).

It covers the research questions, related work, feature engineering, TFT architecture, training setup, model comparisons, augmentation experiments, transfer experiment, limitations, and future work.

## Project Context

This repository is a cleaned-up version of the original university submission and is intended to make the implementation, experiments, model artifacts, and report easier to explore on GitHub.
