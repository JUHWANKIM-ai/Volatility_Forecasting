# Architecture

Weekly realized-volatility forecasting for crude oil with a bidirectional LSTM that is pretrained on one
oil series and transferred to another. This file maps the code; `README.md` describes the experiment and
its results.

## Structure

```
.
├── main.py                 entry point: builds a Run per input file and drives it
├── src/                    all python modules (flat package, no sub-packages)
│   ├── __init__.py         re-exports seed / parser / metrics (importing src runs the parser)
│   ├── parser.py           argparse config + instantiates the model and loads the backbone weights
│   ├── model.py            single_biLSTM (used), MLP (alternative head, unused)
│   ├── dataset.py          CustomDataset (train/val), TestDataset (walk-forward test)
│   ├── train.py            Train_Module: train_fn / eval_fn / train_and_eval, early stopping
│   ├── run.py              Run: load_data, run_model, check_validation, evaluate_testset
│   ├── metrics.py          MAE / RMSE / R2 helpers
│   ├── seed.py             seed_everything
│   └── garch.py            rolling-window GARCH(1,1) forecasts (`arch`), used by notebook 2
├── notebooks/
│   ├── 1_preprocessing.ipynb   downloads prices and exogenous series, builds data/*.csv
│   ├── 2_garch.ipynb           rolling GARCH conditional volatility for data/exterior/
│   └── 3_evaluation.ipynb      metrics, regime split, Diebold–Mariano test, descriptive stats
├── data/                   model inputs (committed)
├── results/                predictions and figures (committed)
└── weights/backbone/       pretrained .pth files + weight_initial_condition.txt
```

Entry points:

| Command | What runs |
|---|---|
| `python main.py` | `Execution('data/future_std', None).main()` — reads the dataset(s), trains or transfers, writes predictions |
| `notebooks/1_preprocessing.ipynb` | regenerates the CSVs under `data/` from the raw sources |
| `notebooks/2_garch.ipynb` | regenerates the GARCH conditional-volatility series |
| `notebooks/3_evaluation.ipynb` | reads `results/*.csv` and prints every number in the README |

Control flow of a run:

```
main.py Execution.main
  └─ src.run.Run(file, config)            # config comes from src.parser at import time
       ├─ load_data(retraining)           # train/val split, CustomDataset, TestDataset
       ├─ run_model(retraining)           # Adam + ReduceLROnPlateau -> Train_Module.train_and_eval
       ├─ check_validation()              # loss / R2 curves
       └─ evaluate_testset(retrain)       # walk-forward loop: predict, re-fit the output layer, repeat
```

## Dependencies

Internal imports (fan-in = how many modules import it):

| Module | Imported by | Fan-in |
|---|---|---|
| `src/metrics.py` | `src/run.py`, `src/train.py`, `src/__init__.py` | 3 |
| `src/parser.py` | `main.py`, `src/__init__.py` | 2 |
| `src/seed.py` | `src/run.py`, `src/__init__.py` | 2 |
| `src/train.py` | `src/run.py` | 1 |
| `src/dataset.py` | `src/run.py` | 1 |
| `src/model.py` | `src/parser.py` | 1 |
| `src/run.py` | `main.py` | 1 |
| `src/garch.py` | `notebooks/2_garch.ipynb` | 1 |

`src/run.py` is the only module that touches the filesystem layout: it derives the weight path
(`weights/backbone/BiLSTM_{BZ,SP}.pth`) and the prediction path (`results/bilstm/additionalX_{BZ,SP}.csv`)
from the last ten characters of the input filename. `src/parser.py` is the single place where
hyperparameters and the backbone weight file are chosen.

External packages (from the actual imports, no versions are pinned anywhere in the repo):

| Package | Used by |
|---|---|
| `torch` | `src/model.py`, `src/train.py`, `src/run.py`, `src/seed.py`, `main.py` |
| `pandas`, `numpy` | everywhere |
| `scikit-learn` | `src/metrics.py` (metrics), `src/run.py` (`train_test_split`), notebooks (`StandardScaler`) |
| `matplotlib` | `src/run.py`, all notebooks |
| `joblib` | notebooks (scaler dump/load) |
| `arch` | `src/garch.py`, `notebooks/2_garch.ipynb` |
| `yfinance`, `requests` | `notebooks/1_preprocessing.ipynb` (Yahoo Finance, Alpha Vantage) |
| `scipy`, `statsmodels` | `notebooks/3_evaluation.ipynb` (Jarque–Bera, ADF, ACF for the DM test) |

## Conventions

- **Series codes.** `SP` = WTI spot (Alpha Vantage `WTI`), `BZ` = Brent futures (`BZ=F`, extended before
  2007 with `data/exterior/BZ_future.csv`). `data/spot_std/` holds the WTI target, `data/future_std/` the
  Brent target.
- **Feature-set suffixes** on dataset, weight and result files: none = volatility only, `SP500` = S&P 500
  index level, `SP500_r` = S&P 500 log return, `Sentiment` = news sentiment index, `ES` = S&P 500 level +
  sentiment, `ES2` = S&P 500 return + sentiment.
- **Result files.** `results/{source}{target}_{featureset}_re.csv` with columns `Predictions`,
  `Ground Truths` in the standardized scale; `_re` marks the walk-forward re-fit runs. The scaler that
  inverts them is `data/scaler/scaler_{0,1}_*.joblib` (0 = Brent, 1 = WTI).
- **Working directory.** `main.py` and `src/` use paths relative to the repository root; notebooks use
  `../data`, `../results` and are meant to be run from `notebooks/`.
- **Everything is float64.** The LSTM layers call `.double()`, so inputs must stay `float64`.
- **Configuration is import-time.** `src/parser.py` runs `argparse` and loads the backbone weights while
  being imported, and `src/__init__.py` imports it, so importing anything from `src` already parses
  `sys.argv` and reads `weights/backbone/`. Flags are declared with `type=bool`, which means
  `--retrain False` is truthy; the defaults in the file are the real switches.
- **Star imports carry the third-party namespace.** `src/run.py` gets `pd`, `torch` and `time` through
  `from src.train import *`. Replacing those star imports requires adding the explicit imports first.
- **File names** are lowercase snake_case for code; data, weight and figure names keep their ticker and
  feature-set spelling (`BZ_vol_ES2.csv`, `BiLSTM_SP_onlyvol.pth`) because the code and the notebooks
  index them by that string.

## Impact

What this cleanup changed (no behavior was altered; only locations, import paths and the strings that
point at them):

| Before | After |
|---|---|
| `Database/` | `data/` (`Exterior`→`exterior`, `NLP`→`nlp`, `Scaler`→`scaler`) |
| `Files/` | `results/` (`BiLSTM`→`bilstm`, `NLP`→`nlp`) |
| `Weight/` | `weights/` (`Backbone`→`backbone`) |
| `Modules/*.py`, `utils/*.py`, `utils/garch/garch.py` | `src/*.py` (`Metrics.py`→`metrics.py`) |
| `EDA.ipynb`, `preprocessing.ipynb`, `utils/garch/garch.ipynb` | `notebooks/3_evaluation.ipynb`, `notebooks/1_preprocessing.ipynb`, `notebooks/2_garch.ipynb` |
| `.idea/`, `__pycache__/` tracked | untracked, ignored |
| Alpha Vantage API keys in the preprocessing notebook | replaced with `YOUR_ALPHAVANTAGE_API_KEY` |

Every data path in the notebooks and in the code was rewritten and then checked to resolve to a file that
exists; `import garch` in notebook 2 became `sys.path.append('../src')` + `import garch`.

Risk ranking of the code, by cyclomatic complexity (`lizard`). The repository has no tests, so a real CRAP
score cannot be computed — this is the complexity half of it only:

| Function | CC | NLOC |
|---|---|---|
| `Train_Module.train_and_eval` (`src/train.py`) | 5 | 53 |
| `Run.evaluate_testset` (`src/run.py`) | 5 | 42 |
| `Run.run_model` (`src/run.py`) | 5 | 29 |
| `Execution.main` (`main.py`) | 5 | 14 |
| `Execution.get_file_list` (`main.py`) | 4 | 6 |
| `Run.load_data` (`src/run.py`) | 3 | 26 |

Nothing is deeply nested; the difficulty of this code is in the state that `Run` carries between
`evaluate_testset` and `load_data` (`self.test_index`, `self.test_data`, `self.model`), not in branching.

Known rough edges, left untouched because they cannot be verified without data and a GPU:

- `Run.evaluate_testset` breaks out of the test loader after the first prediction of every pass while
  `retrain` is on, so the output layer is re-fit at every step; the Korean comments in the file describe a
  longer re-fit period than the code implements.
- `MLP.__init__` in `src/model.py` ends with `self.fc = nn.Linear(...),` — a trailing comma makes it a
  tuple, so the `backbone2` branch would fail. No MLP weights or results are committed.
- `utils/__pycache__/save_model.cpython-311.pyc` existed without a `save_model.py`; nothing imports it, so
  only the bytecode was dropped.
- `data/scaler/` contains `scaler_{0,1}_onlyvol.joblib`, but the last cell of notebook 1 dumps
  `scaler_{0,1}_onlyvol3.joblib`; and notebook 2 names a `garch/` output directory under the results root
  that has never existed (the variable is unused).
- `Execution.get_file_list` splits glob results on `/`, which `glob` does not produce on Windows.
