# Predicting Stock Market Volatility Using Time Series Models

Do stock returns behave the way basic financial models assume, and which model forecasts volatility best?

This project tests the assumptions behind standard volatility models on 15+ years of daily S&P 500 data, then compares classical econometric models (GARCH, EGARCH, GJR-GARCH) against an LSTM neural network and a hybrid GARCH-LSTM on one-day-ahead variance forecasts. Differences between models are checked with Diebold-Mariano significance tests.

## Key findings

- **Returns violate the basic assumptions.** They are stationary (ADF p ≈ 3e-26) but strongly non-normal: skew -0.61, excess kurtosis 13.5, and a fitted Student-t with about 5.8 degrees of freedom.
- **Volatility clusters.** Returns are nearly uncorrelated, but squared returns are strongly autocorrelated (Ljung-Box statistic 4,282 at 10 lags, p < 0.001). GARCH(1,1) persistence (α + β) is about 0.997.
- **Asymmetry matters.** EGARCH and GJR-GARCH fit better in-sample than GARCH(1,1) (AIC 8,430 and 8,455 vs. 8,575). The EGARCH leverage coefficient is negative (γ = -0.20), consistent with negative shocks raising volatility more than positive ones.
- **GJR-GARCH was the best forecaster.** Out of sample (845 trading days, June 2023 to October 2026) it had the lowest QLIKE and significantly beat GARCH(1,1) (Diebold-Mariano p = 0.001) and a rolling-variance baseline (p = 0.001). EGARCH and GJR-GARCH were statistically indistinguishable (p = 0.66).
- **Rolling re-estimation helped slightly.** Re-fitting every 20 days improved QLIKE by a small margin, significant for GJR-GARCH (p = 0.016) and GARCH(1,1) (p = 0.03) but not for EGARCH (p = 0.13).
- **Neural networks did not beat GJR-GARCH.** The LSTM and the hybrid GARCH-LSTM were not statistically different from GJR-GARCH (p = 0.56 and p = 0.72), so extra complexity did not improve forecasts.

## Results

One-day-ahead variance forecasts on the test period, scored against squared returns. Lower is better. QLIKE is the primary metric because RMSE and MAE are dominated by a few extreme days.

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| GJR-GARCH (rolling) | 3.1104 | 0.9344 | **0.5788** |
| GJR-GARCH (fixed) | 3.1085 | 0.9430 | 0.5829 |
| EGARCH (rolling) | 3.1447 | 0.9030 | 0.5856 |
| EGARCH (fixed) | 3.1395 | 0.9078 | 0.5882 |
| Hybrid GARCH-LSTM | 3.3299 | 0.8974 | 0.5967 |
| LSTM | 3.3102 | 0.9352 | 0.6028 |
| GARCH(1,1) (rolling) | 3.3397 | 0.9974 | 0.6370 |
| GARCH(1,1) (fixed) | 3.3483 | 1.0071 | 0.6408 |
| Rolling 21-day variance (baseline) | 3.4695 | 1.0043 | 0.7476 |

"Fixed" models use parameters estimated once on the training window. "Rolling" models are re-estimated every 20 trading days on all data available at that point.

Selected Diebold-Mariano comparisons (QLIKE loss, HAC standard errors):

| Comparison | DM statistic | p-value | Result |
|---|---|---|---|
| GJR-GARCH vs. GARCH(1,1) | -3.40 | 0.001 | GJR better |
| EGARCH vs. GARCH(1,1) | -2.39 | 0.017 | EGARCH better |
| GJR-GARCH vs. EGARCH | -0.44 | 0.658 | no difference |
| GJR-GARCH vs. rolling-variance baseline | -3.34 | 0.001 | GJR better |
| GJR-GARCH rolling vs. fixed | -2.42 | 0.016 | rolling better |
| Hybrid GARCH-LSTM vs. GJR-GARCH | 0.35 | 0.724 | no difference |
| Hybrid GARCH-LSTM vs. LSTM | -0.55 | 0.585 | no difference |

## Replication across assets

The pipeline was rerun on three other US indices and two individual stocks (3-seed neural network ensembles; same 2010 start and 80/20 split).

| Asset | GARCH(1,1) | EGARCH | GJR-GARCH | LSTM | Hybrid | Hybrid vs GJR (%) | DM p | 90% MCS |
|---|---|---|---|---|---|---|---|---|
| ^GSPC | 0.6408 | 0.5882 | 0.5829 | 0.5624 | 0.5623 | -3.53 | 0.052 | EGARCH, Hybrid, LSTM |
| ^IXIC | 1.3214 | 1.2986 | 1.2972 | 1.2626 | 1.2960 | -0.09 | 0.011 | EGARCH, Hybrid, LSTM |
| ^DJI | 0.5008 | 0.4853 | 0.4742 | 0.4897 | 0.4708 | -0.70 | 0.009 | EGARCH, GARCH(1,1), Hybrid, LSTM |
| ^RUT | 1.4905 | 1.4952 | 1.5075 | 1.4764 | 1.5074 | -0.01 | 0.986 | all except baseline |
| AAPL | 1.8692 | 1.8419 | 1.8509 | 1.8682 | 1.8495 | -0.08 | 0.354 | all except baseline |
| JPM | 1.6747 | 1.6530 | 1.6730 | 1.6438 | 1.6607 | -0.74 | 0.153 | all except baseline |

- The rolling-variance baseline was excluded from the 90% Model Confidence Set on every asset; EGARCH, the QLIKE-trained LSTM and the hybrid were never excluded.
- The hybrid's QLIKE was below GJR-GARCH's on all six assets, but the improvement was large only for the S&P 500 and JPM, and significance in the other cases reflects very small differences between nearly identical forecasts.
- The asymmetric models beat GARCH(1,1) in 5 of 6 assets, but GARCH(1,1) stayed in the confidence set for 4 of 6.
- Caveat: the assets overlap heavily and share one test period, so these are not independent replications.

### Rolling-origin robustness (notebook 10)
To check that results weren't specific to one test window, I re-evaluated all models on five
non-overlapping ~1.7-year test blocks (2018–2026, expanding training window), including the
2020 COVID crash.

- The hybrid GARCH-LSTM had lower QLIKE than GJR-GARCH in 4 of 5 blocks, but by small
  margins (0.005–0.020), and lost in the 2020–21 block.
- The LSTM without a GARCH anchor lost badly in the crisis block (+0.30 QLIKE), while the
  hybrid lost only +0.03.
- Pooled over 2,111 test days, no model was statistically distinguishable from GJR-GARCH
  (hybrid DM p = 0.62). Only the 21-day rolling-variance baseline was significantly worse.
  The 90% Model Confidence Set retained all five GARCH-family and neural models.
- Conclusion: neural gains are modest and regime-dependent; GJR-GARCH remains a strong,
  hard-to-beat benchmark.
  
## Approach

1. **Data:** Daily S&P 500 (`^GSPC`) prices from 2010 via `yfinance`; log returns in percent.
2. **Exploratory analysis:** ADF stationarity test, return distribution (skew, kurtosis, Q-Q plot), autocorrelation of returns vs. squared returns, Ljung-Box and Engle ARCH-LM tests.
3. **Models:** GARCH(1,1), EGARCH and GJR-GARCH with Student-t errors; an LSTM on lagged returns and volatility features; a hybrid LSTM that also receives the GJR-GARCH forecast as an input.
4. **Evaluation:** Chronological 80/20 split. Parameters and scaling are estimated on training data only, and each forecast uses only earlier information. Models are scored with RMSE, MAE and QLIKE and compared with Diebold-Mariano tests.

## Project structure

```
.
├── notebooks/
│   ├── 01_data_collection.ipynb    # download prices, log returns, realized volatility
│   ├── 02_eda.ipynb                # stationarity, fat tails, ARCH effects
│   ├── 03_garch_models.ipynb       # GARCH / EGARCH / GJR-GARCH, fixed-parameter forecasts
│   ├── 04_lstm_model.ipynb         # LSTM forecasts and comparison
│   ├── 05_rolling_garch_dm.ipynb   # rolling re-estimation, Diebold-Mariano tests
│   └── 06_hybrid_garch_lstm.ipynb  # hybrid GARCH-LSTM
├── src/
│   └── utils.py                    # data loading, returns, QLIKE, evaluation helpers
├── data/                           # generated CSVs (not tracked in Git)
└── requirements.txt
```

## How to run

Python 3.11 is recommended (TensorFlow does not yet support the newest Python releases).

```bash
git clone https://github.com/laurenbe06/volatility-forecasting.git
cd volatility-forecasting
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The `data/` folder is not tracked, so run the notebooks **in order** (01 to 06) to generate it:

- Notebook 01 creates `GSPC_processed.csv`.
- Notebook 03 saves `garch_forecasts.csv` and notebook 04 saves `lstm_forecasts.csv`, which notebooks 05 and 06 read.

A dev container configuration is included (`.devcontainer/`), so the project can also be opened in GitHub Codespaces or VS Code Dev Containers with dependencies installed automatically.

Prices are downloaded from Yahoo Finance up to the current date, so results will shift slightly from those above. LSTM results also vary slightly between runs even with fixed seeds.

## Limitations

- Squared returns are a noisy proxy for true variance. Intraday realized variance would be a better benchmark.
- About 28 pairwise Diebold-Mariano tests were run without multiple-testing correction, so borderline p-values (0.02 to 0.03) should be read cautiously.
- LSTM training was not tuned: in the hybrid model, validation loss was lowest after the first epoch, so the neural network results are likely not at their best.
- The mean model is a constant, although returns show slight negative autocorrelation at lag 1.
- Results are for one index and one test period.

## Possible extensions

- Mincer-Zarnowitz regressions and a Model Confidence Set for more rigorous forecast evaluation
- Value-at-Risk backtesting (Kupiec and Christoffersen tests)
- Better LSTM training (lower learning rate, log-variance targets, learning a correction to the GARCH forecast)
- Intraday realized variance, other indices and individual stocks

## Tools

Python, pandas, NumPy, statsmodels, arch, scikit-learn, TensorFlow/Keras, matplotlib

## Author

Lauren Berning