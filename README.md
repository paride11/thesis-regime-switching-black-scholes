# Regime-Switching Black-Scholes for Equity-Linked Securities

Research code for my bachelor thesis on pricing Equity-Linked Securities with a Regime-Switching Black-Scholes framework.

The repository is organized as a reproducible Python project: data preparation, model estimation, pricing routines, figures, and results are separated into clear folders so the analysis can be reviewed and rerun from the command line or from the notebooks.

## Repository layout

```text
regime-switching-black-scholes-els/
├── data/
│   ├── raw/              # Raw WRDS/SPX inputs; large WRDSoptions.csv is not tracked
│   └── processed/        # Option-price datasets used by the analysis
├── docs/                 # Bachelor thesis PDF
├── notebooks/            # Research notebooks
├── scripts/              # Command-line entry points
├── src/
│   ├── data/             # Dataset construction logic
│   └── rsbs/             # HMM, PDE, pricing, evaluation, and plotting routines
├── figures/              # Generated figures
├── results/              # Generated tables/results
└── tests/                # Lightweight smoke tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional plotting style:

```bash
pip install mpl-bsic
```

The code falls back to default Matplotlib styling if `mpl-bsic` is unavailable.

## Data

The full WRDS option chain is large and should not normally be committed to Git. Place it here before rebuilding the dataset:

```text
data/raw/WRDSoptions.csv
```

The smaller base dataset is expected at:

```text
data/raw/WRDSspx_rf.csv
```

Processed datasets are stored under:

```text
data/processed/option_price_atm.csv
data/processed/option_price_otm_90.csv
```

## Rebuild the option dataset

```bash
python scripts/build_option_dataset.py
```

## Run the pricing pipeline

```bash
python scripts/run_regime_switching_black_scholes.py
```

## Notebooks

The notebooks in `notebooks/` provide the research workflow and visual outputs:

```text
notebooks/feature_engineering.ipynb
notebooks/regime_switching_black_scholes.ipynb
```

## Project components

- `src/data/option_dataset.py`: construction of ATM and OTM option-price datasets from WRDS/SPX inputs.
- `src/rsbs/core.py`: numerical routines for HMM regime estimation, Black-Scholes pricing, finite-difference PDE solving, ELS valuation, evaluation metrics, and plotting helpers.
- `scripts/build_option_dataset.py`: reproducible data-preparation entry point.
- `scripts/run_regime_switching_black_scholes.py`: full pricing and evaluation pipeline.

## Verification checklist

Before publishing or updating the repository, run:

```bash
python -m compileall src scripts
python scripts/build_option_dataset.py
python scripts/run_regime_switching_black_scholes.py
pytest
```

Large raw WRDS files are intentionally excluded from version control. Keep them locally under `data/raw/`.
