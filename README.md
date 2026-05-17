# QINN vs ANN Stock Forecasting Comparison

This repository compares two neural-network approaches for stock return forecasting:

- an advanced artificial neural network, or ANN
- a quantum-inspired neural network, or QINN

The project is written as a research experiment, not as a trading system. The goal is to test whether the QINN-style architecture behaves differently from a larger classical ANN when both are trained on the same stock-market features.

## What This Project Tests

The experiment uses historical daily stock data and engineered market features to predict future returns over several horizons:

- 1 trading day
- 15 trading days
- 21 trading days, roughly one month
- 63 trading days, roughly one quarter

The main comparison is between:

- **ANN:** a larger deep-learning model with dense, CNN, LSTM, attention, and residual components
- **QINN:** a smaller hybrid model using quantum-inspired feature encoding and variational-circuit style layers through PennyLane

The project tracks both regression metrics, such as RMSE and MAE, and directional accuracy, meaning whether the model predicted the correct up/down direction.

## Current Result Snapshot

The saved result file at `results/RealTest1.json` contains 240 completed experiment runs:

- 15 stocks
- 4 horizons
- 2 models
- 2 training-window setups

Aggregate results from that file:

| Metric | ANN | QINN |
| --- | ---: | ---: |
| Runs | 120 | 120 |
| Average directional accuracy | 50.58% | 51.53% |
| Average RMSE | 0.1528 | 0.1426 |
| Average training time | 338.31 sec | 515.58 sec |
| Model parameters | 2,069,691 | 56,199 |

The QINN model was slightly better on average in this saved run, but the difference should be treated carefully. A paired t-test on directional accuracy gives `p = 0.144`, so the current saved results do not prove a statistically significant accuracy advantage. The RMSE comparison is closer, with `p = 0.070`, but still should not be overstated.

The honest reading is: QINN appears more compact and somewhat competitive in these experiments, but this repository needs stronger validation before making a strong claim of quantum-inspired advantage.

## Stocks Used

The saved experiment covers:

`AVAV`, `BAC`, `BRK-B`, `C`, `CVX`, `LULU`, `MS`, `NKTR`, `NVDA`, `PFE`, `TEAM`, `TRIP`, `TSLA`, `UNH`, `XOM`

## Repository Layout

```text
.
├── models/
│   ├── ann_model.py              # ANN architecture and trainer
│   ├── qinn_model.py             # QINN architecture and trainer
│   └── Ensemble/                 # Ensemble model experiments
├── preprocessing/
│   ├── data_processor.py         # Data loading, splits, scaling, labels
│   ├── feature_engineering.py    # Technical and statistical features
│   └── transformers.py           # Additional preprocessing utilities
├── evaluation/
│   ├── metrics.py                # Metrics and risk/statistical utilities
│   └── statistical_analysis.py   # Statistical comparison helpers
├── scripts/
│   ├── run_hyperparam_tune.py
│   ├── run_walkforward_eval.py
│   ├── run_walkforward_eval_safe.py
│   └── run_ensemble_smoke.py
├── tools/
│   └── small helper scripts
├── results/
│   ├── RealTest1.json
│   ├── hyperparam_tune_results.json
│   └── RESULTS_REPORT.html
├── tests/
├── main_experiment.py
├── adjusted_experiment.py
└── README.md
```

## Setup

Create a virtual environment first:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TA-Lib is optional. The current feature code does not require it during import, because installing TA-Lib on Windows can be annoying. If you later add TA-Lib-based features, install it separately.

## Running Checks

Basic code checks:

```bash
python -m compileall .
python -m pytest
```

The tests are intentionally light right now. They verify that the core model and preprocessing modules import correctly. More research-grade tests should be added before relying on the experiment results.

## Running Experiments

The full experiment can take a long time because QINN training is slower than ANN training:

```bash
python main_experiment.py
```

For a smaller or safer run, start with:

```bash
python scripts/run_ensemble_smoke.py
python scripts/run_walkforward_eval_safe.py
```

The safe walk-forward script uses an XGBoost surrogate for QINN by default. That is useful for quick testing, but it should not be described as a true QINN result unless `USE_QINN_SURROGATE` is turned off.

## Important Research Notes

This project has several limitations that matter:

- The reported results are close to random directional accuracy, which is normal for stock forecasting.
- The saved results do not currently show a statistically significant directional-accuracy advantage for QINN.
- Some older project notes claimed stronger results than the saved data supports. This README uses the more conservative interpretation.
- The experiment uses today’s selected stock universe, so survivorship and selection bias are possible.
- Stock prediction is noisy, and good backtest numbers can disappear with small changes in dates, features, or transaction assumptions.
- This is not financial advice and should not be used as a live trading strategy.

## Suggested Next Improvements

The next useful steps are:

- add tests for leakage-free target generation
- add tests proving scalers fit only on training data
- separate true QINN runs from surrogate runs in result files
- add a clean script that reproduces the summary table from `RealTest1.json`
- run multiple walk-forward folds instead of relying on one saved result file
- report confidence intervals, paired tests, and multiple-testing caveats directly in the generated report

## Bottom Line

This is a solid experimental starting point for comparing a compact quantum-inspired model with a larger ANN on financial time-series data. The current results are interesting but not conclusive. The best version of this project is one that is careful, reproducible, and honest about uncertainty.
