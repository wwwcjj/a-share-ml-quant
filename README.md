# A-Share ML Quant

Research pipeline for A-share stock selection with LightGBM.

## Files

- `factors.py` — shared factor definitions (14 price/volume factors), used by both backtest and signal
- `fetch_all.py` — downloads ~10 years of daily bars for the full A-share listing (~5,100 codes)
- `pipeline.py` — backtest: time-ordered split with purge/embargo, account-level simulation (6,000 RMB, board lots, commission/stamp/transfer fees), then emits a candidate portfolio
- `strategy.py` — standalone: retrains on all labelled history and prints the latest candidate target portfolio
- `verify.py` — read-only audit of the dataset and output artifacts
- `out/` — sample outputs (metrics, IC series, NAV curve, strategy card)

## Environment

Python 3.12 with: numpy, pandas, lightgbm, scikit-learn, akshare, pyarrow, matplotlib, requests

## Usage

1. `python fetch_all.py` — build `data/all_prices.parquet` (hours; resumable)
2. `python pipeline.py` — backtest and print a candidate portfolio
3. `python strategy.py --capital 6000 --names 10 --hold 5` — refresh the trading candidate list
4. `python verify.py` — audit data and outputs

## Methodology notes

- Signals are formed after the close of day t; the research assumption is entry at the next trading day's open and exit at the close of day t+hold.
- The train/test boundary purges and embargoes a full holding period so labels cannot overlap.
- Cross-sectional winsorisation and standardisation use only the same signal date's own cross-section.

## Known limitations

- The universe is the currently listed set of stocks, so the sample carries survivorship bias.
- Limit-up/limit-down fills, suspensions, slippage and exact broker fee schedules are not modelled.
- `data/` is intentionally excluded from Git because the parquet file exceeds GitHub's 100 MiB file limit; rebuild it with `fetch_all.py`.
- This repository is research code. It is not investment advice.
