# Volatility Forecasting (crude-oil realized volatility, 2024)

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-biLSTM-EE4C2C?logo=pytorch&logoColor=white)](src/model.py)
[![GARCH](https://img.shields.io/badge/arch-GARCH(1%2C1)-4c6ef5)](src/garch.py)
[![Data](https://img.shields.io/badge/data-Brent%20%26%20WTI%20weekly-FFD21E)](data)

One-week-ahead forecasting of the **realized volatility of crude oil**. The target is weekly realized
volatility built from daily log returns — the square root of the 20-day rolling sum of squared returns,
averaged to Friday weeks — for two series: **Brent futures** (`BZ=F`, extended before 2007 with a
committed price file) and **WTI spot** (Alpha Vantage `WTI`). The forecaster is a two-layer
**bidirectional LSTM** that reads a 20-week window and predicts the next week.

The question the repository answers is whether a backbone **pretrained on one oil series transfers to the
other**, and whether **exogenous market data** (S&P 500) and **news text** (a sentiment index and 384-d
sentence embeddings of oil news) help. A backbone is pretrained up to 2005-01-01, then transferred to the
Brent target: the LSTM is frozen, the output layer is re-fit, and the model walks forward one week at a
time to the end of 2023 (969 weekly predictions, 2005-05-27 to 2023-12-29). A rolling-window GARCH(1,1)
(`arch`) is implemented in `src/garch.py`, and a GARCH conditional-volatility series is one column of
the exogenous feature table. All numbers
below are read out of `notebooks/3_evaluation.ipynb`; nothing is reproduced here that the notebook does
not print.

## Pipeline

```
 daily prices                                      weekly oil news (2002-2023, parquet)
   WTI spot        (Alpha Vantage WTI)                headlines + 384-d sentence embeddings
   Brent futures   (BZ=F, + data/exterior/BZ_future.csv before 2007)
   │                                                 │
   ▼  log returns                                    ▼  sentiment index, UMAP-75, UMAP-100, IE
 realized volatility                               data/nlp/*.parquet -> data/nlp/SP_vol_*.csv
   sqrt( 20-day rolling sum of r^2 ),
   resampled to weekly (W-FRI mean)
   │
   │  exogenous table (data/exterior/exterior.csv), weekly, forward-filled:
   │    S&P 500 level and log return, Dow Jones, GDP, fuel CPI, 1-year expected inflation,
   │    BDI level/return/vol, 3m-7y-30y Treasury yields, GARCH(1,1) conditional volatility
   ▼
 StandardScaler per target and feature set  ->  data/scaler/*.joblib
   │
   ▼  input: 20 weeks x features          target: next week's volatility
 2-layer biLSTM (hidden 64, float64) + linear head
   RMSE loss, Adam 1e-3, ReduceLROnPlateau, early stopping (patience 10), batch 32, seed 0
   │
   ├── stage 1  pretrain on one series up to 2005-01-01, 80/20 chronological train/val
   │            -> weights/backbone/BiLSTM_{SP,BZ}_{featureset}.pth
   └── stage 2  transfer to Brent: freeze the backbone, re-fit the output layer at every step,
                predict one week, advance, repeat until 2024-01-01
   │
   ▼
 results/{BZBZ,SPBZ}_{featureset}_re.csv   (Predictions, Ground Truths, standardized)
   │
   ▼  inverse-transform with the matching scaler
 notebooks/3_evaluation.ipynb  ->  MAE, RMSE, MAPE, QLIKE, R2; regime split; Diebold-Mariano
```

Model names in the tables read `{pretraining series}-{target series}-{features}`: `BZ-BZ` is pretrained on
Brent, `WTI-BZ` on WTI spot, and both are evaluated on Brent. `-E` adds the S&P 500 index level, `-S` the
news sentiment index, `-ES` the S&P 500 log return together with the sentiment index.

## Results

### Target series (weekly realized volatility, `data/{BZ,SP}_vol_only_scaledX.csv`)

| Series | n | Mean | Std | Min | Median | Max | Kurtosis | Skew |
|---|---|---|---|---|---|---|---|---|
| Brent (BZ) | 1266 | 0.0937 | 0.0480 | 0.0226 | 0.0833 | 0.4482 | 14.36 | 2.94 |
| WTI (SP) | 1995 | 0.1010 | 0.0584 | 0.0307 | 0.0877 | 0.7325 | 27.53 | 3.86 |

Jarque–Bera rejects normality for both (p = 0.0); the augmented Dickey–Fuller test rejects a unit root
(WTI p = 3.5e-09, Brent p = 1.6e-04).

### Whole test period (969 weeks, 2005-05-27 to 2023-12-29, volatility units)

| Model | R2 | MAE | RMSE | MAPE | QLIKE |
|---|---|---|---|---|---|
| BZ-BZ | 0.8690 | 0.009561 | 0.018931 | 0.096309 | 0.009829 |
| WTI-BZ | 0.8875 | 0.009103 | 0.017548 | 0.091686 | 0.009538 |
| BZ-BZ-E | 0.9063 | 0.009363 | 0.016010 | 0.108551 | 0.010958 |
| WTI-BZ-E | 0.9175 | 0.009028 | 0.015027 | 0.106084 | 0.010618 |
| BZ-BZ-S | 0.8979 | 0.009129 | 0.016713 | 0.097939 | 0.009809 |
| WTI-BZ-S | 0.9210 | 0.008538 | 0.014704 | 0.093222 | 0.008795 |
| BZ-BZ-ES | 0.8958 | 0.009000 | 0.016888 | **0.0916** | 0.009142 |
| **WTI-BZ-ES** | **0.9219** | **0.008424** | **0.014618** | 0.092407 | **0.008744** |

### By regime

Calm period, up to 2008-09-01:

| Model | MAE | RMSE | MAPE | QLIKE |
|---|---|---|---|---|
| BZ-BZ | 0.005105 | 0.006810 | 0.065757 | 0.003580 |
| **WTI-BZ** | **0.004883** | **0.006520** | **0.062951** | **0.003322** |
| BZ-BZ-E | 0.005506 | 0.007095 | 0.073008 | 0.004085 |
| WTI-BZ-E | 0.005124 | 0.006879 | 0.065865 | 0.003675 |
| BZ-BZ-S | 0.005263 | 0.006963 | 0.068153 | 0.003783 |
| WTI-BZ-S | 0.005298 | 0.007025 | 0.068821 | 0.003826 |
| BZ-BZ-ES | 0.005156 | 0.006919 | 0.065981 | 0.003691 |
| WTI-BZ-ES | 0.005320 | 0.007008 | 0.069187 | 0.003841 |

2008 crisis, 2008-09-01 to 2009-06-01:

| Model | MAE | RMSE | MAPE | QLIKE |
|---|---|---|---|---|
| BZ-BZ | 0.028146 | 0.034110 | 0.131126 | 0.014987 |
| WTI-BZ | 0.030451 | 0.035644 | 0.156281 | 0.017474 |
| BZ-BZ-E | 0.020952 | 0.025312 | 0.100144 | 0.008301 |
| **WTI-BZ-E** | **0.013588** | **0.017557** | **0.071305** | **0.004735** |
| BZ-BZ-S | 0.022117 | 0.026150 | 0.107175 | 0.009737 |
| WTI-BZ-S | 0.020950 | 0.025951 | 0.100861 | 0.008730 |
| BZ-BZ-ES | 0.024280 | 0.028621 | 0.115968 | 0.011422 |
| WTI-BZ-ES | 0.020454 | 0.024806 | 0.099073 | 0.007987 |

COVID period, 2019-12-01 to 2022-09-01:

| Model | MAE | RMSE | MAPE | QLIKE |
|---|---|---|---|---|
| BZ-BZ | 0.019218 | 0.039316 | 0.122090 | 0.021406 |
| WTI-BZ | 0.017561 | 0.035022 | 0.117700 | 0.023277 |
| BZ-BZ-E | 0.016541 | 0.030807 | 0.122749 | 0.019424 |
| WTI-BZ-E | 0.015945 | 0.028506 | 0.122120 | 0.019479 |
| BZ-BZ-S | 0.017020 | 0.033368 | 0.117851 | 0.019306 |
| WTI-BZ-S | 0.014517 | 0.027105 | 0.108355 | 0.016717 |
| BZ-BZ-ES | 0.017325 | 0.033690 | 0.116538 | 0.018519 |
| **WTI-BZ-ES** | **0.013694** | **0.026914** | **0.104196** | **0.016314** |

Reading of these tables: the WTI-pretrained backbone beats the Brent-pretrained one on the whole period in
every metric; the extra features help in the two stressed windows (in 2008 the best model, `WTI-BZ-E`,
halves the MAE of the volatility-only models) and not in the calm period, where the plain `WTI-BZ` is best.

**Diebold–Mariano.** All 28 pairs of the eight models were compared on the COVID window with the squared
error criterion (`results/dm_test.csv`). No pair is significant at 5%: the statistics range from −1.23 to
1.57 and the smallest p-value is 0.117. The differences in the tables above are not statistically
separable on that window.

**News text instead of volatility.** A separate, earlier experiment (133 weeks, 1999-05-21 to 2001-11-30,
WTI spot target, `results/nlp/`) compares a model fed the UMAP-100 compression of the news embeddings with
the volatility-only model over the same weeks:

| Input | R2 | MAE | RMSE | MAPE | QLIKE |
|---|---|---|---|---|---|
| News embeddings (UMAP-100) | 0.2563 | 0.021333 | 0.027709 | 0.186647 | 0.035151 |
| Volatility only | **0.7240** | **0.012154** | **0.016881** | **0.111498** | **0.009759** |

The other text variants (`Embedding` = 384-d, `UMAP75`, `IE`, `Sentiment`) are stored in `results/nlp/`
with no metrics printed in the notebook, so none are reported here.

Caveats that come with the numbers: the walk-forward re-fit sees data up to the week before each
prediction, but the scalers in `data/scaler/` were fitted on the full sample in the preprocessing
notebook; the `-ES` columns are read from the `*_ES2_re.csv` files (S&P 500 return + sentiment), while the
committed `*_ES_re.csv` files (level + sentiment) are unused in the evaluation; and the GARCH model enters
only as a feature — the repository holds no GARCH benchmark score.

## Repository layout

| Path | Content |
|---|---|
| `main.py` | entry point: pretraining, transfer and the walk-forward evaluation |
| `src/model.py` | `single_biLSTM` (the model) and an unused `MLP` head |
| `src/parser.py` | every hyperparameter and the backbone weight file (argparse defaults) |
| `src/run.py` | `Run`: data loading, training, validation curves, walk-forward test |
| `src/train.py` | training loop, early stopping, best-weight bookkeeping |
| `src/dataset.py` | 20-week sliding-window datasets for training and for the test walk |
| `src/metrics.py`, `src/seed.py` | MAE/RMSE/R2 helpers, seeding |
| `src/garch.py` | rolling-window GARCH(1,1) variance forecasts (`arch`) |
| `notebooks/1_preprocessing.ipynb` | builds every CSV under `data/` from Yahoo Finance, Alpha Vantage, FRED-style files and the news parquets |
| `notebooks/2_garch.ipynb` | the rolling GARCH conditional-volatility series |
| `notebooks/3_evaluation.ipynb` | all tables above: metrics, regimes, DM test, descriptive stats |
| `data/{spot_std,future_std}/` | the model inputs used by `main.py` (WTI spot, Brent futures) |
| `data/{SP,BZ}_vol_*.csv` | one file per feature set (`only_scaled{O,X}`, `SP500`, `SP500_r`, `Sentiment`, `ES`, `ES2`) |
| `data/exterior/` | raw and merged exogenous series (`exterior.csv` is the merge) |
| `data/nlp/` | news headlines, 384-d embeddings, sentiment, UMAP-75/100 and IE compressions |
| `data/scaler/` | the `StandardScaler` objects needed to invert the predictions |
| `results/*.csv` | walk-forward predictions per model, and `dm_test.csv` |
| `results/bilstm/`, `results/nlp/` | predictions of the backbone runs and of the text experiment |
| `results/images/` | the figures of the project |
| `weights/backbone/` | pretrained biLSTM weights per series and feature set, plus `weight_initial_condition.txt` (the argparse namespace each was trained with) |
| `ARCHITECTURE.md` | module map, import graph, conventions, complexity ranking |

## Get Started

**1. Install.** Python 3.11 (the version the project's bytecode was built with); no versions are pinned in
the repository, so `requirements.txt` lists the packages only.

```bash
conda create -n volatility python=3.11 && conda activate volatility
pip install -r requirements.txt
# PyTorch: install the build that matches your CUDA from https://pytorch.org
```

**2. Data.** Everything `main.py` reads is committed under `data/`, so nothing has to be downloaded. To
rebuild it, run `notebooks/1_preprocessing.ipynb` from the `notebooks/` directory; it needs an
[Alpha Vantage](https://www.alphavantage.co/support/#api-key) key (set `api_key` in the first cells, where
the placeholder `YOUR_ALPHAVANTAGE_API_KEY` sits) and network access to Yahoo Finance.

**3. Train.** The switches are the argparse defaults in `src/parser.py`; the file has no CLI wrapper, so
edit the defaults and run from the repository root.

```bash
python main.py
```

| Flag in `src/parser.py` | Effect |
|---|---|
| `Transfer=False` | stage 1: pretrain on `data/spot_std/SP_vol.csv` and write the backbone weights |
| `Transfer=True` | stage 2: load `backbone_weight_path` and run every CSV in `data/future_std/` |
| `retrain=True` | re-fit the output layer at each step of the walk-forward test |
| `backbone_weight_path` | which pretrained backbone to transfer from (`weights/backbone/*.pth`) |
| `backbone_val_end`, `transfer_test_end` | end of pretraining (`2005-01-01`) and of the test walk (`2024-01-01`) |
| `input_size` | must match the number of columns of the dataset (2 for volatility + one feature) |
| `device` | `cuda` by default; the runs in this repository were made on a GPU |

Predictions land in `results/bilstm/additionalX_{BZ,SP}.csv` — that path carries the series code but
not the feature set, so each run has to be kept aside before the next. The committed per-model files
`results/{BZBZ,SPBZ}_{featureset}_re.csv` are those runs kept under their own names, and are the names
`notebooks/3_evaluation.ipynb` reads.

**4. Evaluate.** Run `notebooks/3_evaluation.ipynb` from `notebooks/`; it reloads the scalers, inverts the
predictions and prints every table above.
