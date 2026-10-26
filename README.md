# Predicting Stock Market Volatility Using Time Series Models

Do stock returns behave the way basic financial models assume, and which model forecasts volatility best?

This project tests the assumptions behind standard volatility models on 15+ years of daily S&P 500 data, then compares classical econometric models (GARCH, EGARCH, GJR-GARCH, HAR) against LSTM neural networks and a hybrid GARCH-LSTM on one-day-ahead variance forecasts. Differences between models are checked with Diebold-Mariano tests, the Model Confidence Set and Mincer-Zarnowitz calibration regressions, and the results are stress-tested across six assets, five test windows and two volatility proxies.

## Key findings

- **Returns violate the basic assumptions.** They are stationary (ADF p ≈ 3e-26) but strongly non-normal: skew -0.61, excess kurtosis 13.5, and a fitted Student-t with about 5.8 degrees of freedom.
- **Volatility clusters.** Returns are nearly uncorrelated, but squared returns are strongly autocorrelated (Ljung-Box statistic 4,282 at 10 lags, p < 0.001). GARCH(1,1) persistence (α + β) is about 0.997.
- **Asymmetry matters.** EGARCH and GJR-GARCH fit better in-sample than GARCH(1,1) (AIC 8,430 and 8,455 vs. 8,575). The EGARCH leverage coefficient is negative (γ = -0.20), consistent with negative shocks raising volatility more than positive ones. Out of sample, GJR-GARCH significantly beat GARCH(1,1) under both volatility proxies (p ≤ 0.001) and in 5 of 6 assets.
- **Rolling re-estimation helped.** Re-fitting GJR-GARCH every 20 days beat keeping its parameters fixed (p = 0.016 with squared returns, p = 0.006 with the range-based proxy).
- **How the neural networks were trained mattered.** Training the LSTM with the QLIKE loss used for evaluation, instead of MSE, significantly improved it (p = 0.001). A hybrid network that learns a correction to the GJR-GARCH forecast had the lowest QLIKE of any model and beat GJR-GARCH on the original test period (p = 0.04).
- **But the neural edge was fragile.** Under a less noisy range-based proxy the hybrid's advantage was only borderline (p = 0.10). Pooled across five rolling test windows, no model was distinguishable from GJR-GARCH (p = 0.62), and both neural models lost to GJR-GARCH during the 2020 COVID crash. GJR-GARCH remains a strong, hard-to-beat benchmark.
- **Range data helped HAR, but not enough to win.** HAR built on intraday-range data was far better than HAR built on squared returns, but it only tied GJR-GARCH.
- **A proxy-scaling error can reverse model rankings.** Rescaling the range-based proxy with a fixed factor estimated on the training period inflated it by about 34% in the test period, which made HAR look like the best model and most models look miscalibrated. A rolling, real-time scale factor removed both artifacts.

## Results

One-day-ahead variance forecasts on the test period (845 trading days, June 2023 to October 2026), scored against squared returns. Lower is better. QLIKE is the primary metric because RMSE and MAE are dominated by a few extreme days.

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| Hybrid GARCH-LSTM (QLIKE-trained) | 3.1081 | 0.9021 | **0.5546** |
| LSTM (QLIKE-trained) | 3.3094 | 0.8659 | 0.5687 |
| GJR-GARCH (rolling) | 3.1104 | 0.9344 | 0.5788 |
| GJR-GARCH (fixed) | 3.1085 | 0.9430 | 0.5829 |
| EGARCH (rolling) | 3.1447 | 0.9030 | 0.5856 |
| EGARCH (fixed) | 3.1395 | 0.9078 | 0.5882 |
| Hybrid GARCH-LSTM (MSE-trained) | 3.3299 | 0.8974 | 0.5967 |
| LSTM (MSE-trained) | 3.3102 | 0.9352 | 0.6028 |
| GARCH(1,1) (rolling) | 3.3397 | 0.9974 | 0.6370 |
| GARCH(1,1) (fixed) | 3.3483 | 1.0071 | 0.6408 |
| Rolling 21-day variance (baseline) | 3.4695 | 1.0043 | 0.7476 |

"Fixed" models use parameters estimated once on the training window. "Rolling" models are re-estimated every 20 trading days on all data available at that point. The MSE-trained networks are the first versions (notebooks 04 and 06); the QLIKE-trained versions (notebook 07) are 5-seed ensembles.

Selected Diebold-Mariano comparisons (QLIKE loss, HAC standard errors):

| Comparison | DM statistic | p-value | Result |
|---|---|---|---|
| GJR-GARCH vs. GARCH(1,1) | -3.40 | 0.001 | GJR better |
| EGARCH vs. GARCH(1,1) | -2.39 | 0.017 | EGARCH better |
| GJR-GARCH vs. EGARCH | -0.44 | 0.658 | no difference |
| GJR-GARCH vs. rolling-variance baseline | -3.34 | 0.001 | GJR better |
| GJR-GARCH rolling vs. fixed | -2.42 | 0.016 | rolling better |
| Hybrid (QLIKE) vs. GJR-GARCH | -2.06 | 0.040 | hybrid better |
| LSTM (QLIKE) vs. GJR-GARCH | -0.42 | 0.676 | no difference |
| Hybrid (QLIKE) vs. LSTM (QLIKE) | -0.49 | 0.624 | no difference |
| LSTM (QLIKE) vs. LSTM (MSE) | -3.47 | 0.001 | QLIKE training better |

Because many pairwise tests were run, the Model Confidence Set (Hansen, Lunde and Nason, 2011) gives a more reliable summary. The 90% confidence set on this test period contains EGARCH (fixed and rolling), the QLIKE-trained LSTM and the hybrid. In Mincer-Zarnowitz calibration regressions, only the rolling-variance baseline was rejected at the 5% level, but these tests have little power with a squared-return proxy.

## Replication across assets

The pipeline was rerun on three other US indices and two individual stocks (3-seed neural network ensembles; same 2010 start and 80/20 split; squared-return proxy).

| Asset | GARCH(1,1) | EGARCH | GJR-GARCH | LSTM | Hybrid | Hybrid vs GJR (%) | DM p | 90% MCS |
|---|---|---|---|---|---|---|---|---|
| ^GSPC | 0.6408 | 0.5882 | 0.5829 | 0.5624 | 0.5623 | -3.53 | 0.052 | EGARCH, Hybrid, LSTM |
| ^IXIC | 1.3214 | 1.2986 | 1.2972 | 1.2626 | 1.2960 | -0.09 | 0.011 | EGARCH, Hybrid, LSTM |
| ^DJI | 0.5008 | 0.4853 | 0.4742 | 0.4897 | 0.4708 | -0.70 | 0.009 | EGARCH, GARCH(1,1), Hybrid, LSTM |
| ^RUT | 1.4905 | 1.4952 | 1.5075 | 1.4764 | 1.5074 | -0.01 | 0.986 | all except baseline |
| AAPL | 1.8692 | 1.8419 | 1.8509 | 1.8682 | 1.8495 | -0.08 | 0.354 | all except baseline |
| JPM | 1.6747 | 1.6530 | 1.6730 | 1.6438 | 1.6607 | -0.74 | 0.153 | all except baseline |

- The rolling-variance baseline was excluded from the 90% Model Confidence Set on every asset; EGARCH, the QLIKE-trained LSTM and the hybrid were never excluded.
- The hybrid's QLIKE was below GJR-GARCH's on all six assets, but the improvement was meaningful only for the S&P 500 (3.5%). Elsewhere it was under 1%, and significance for the Nasdaq and the Dow reflects very small differences between nearly identical forecasts.
- The asymmetric models beat GARCH(1,1) in 5 of 6 assets, but GARCH(1,1) stayed in the confidence set for 4 of 6.
- Caveat: the assets overlap heavily and share one test period, so these are not independent replications, and 12 tests were run without multiple-testing correction.

### Rolling-origin robustness (notebook 10)
To check that results weren't specific to one test window, I re-evaluated all models on five
non-overlapping ~1.7-year test blocks (2018–2026, expanding training window), including the
2020 COVID crash.

- The hybrid GARCH-LSTM had lower QLIKE than GJR-GARCH in 4 of 5 blocks, but by small
  margins (0.005–0.020), and lost in the 2020–21 block.
- The LSTM without a GARCH anchor lost badly in the crisis block (+0.30 QLIKE), while the
  hybrid lost only +0.03. GARCH(1,1) was the best model in that block.
- Pooled over 2,111 test days, no model was statistically distinguishable from GJR-GARCH
  (hybrid DM p = 0.62). Only the 21-day rolling-variance baseline was significantly worse.
  The 90% Model Confidence Set retained all five GARCH-family and neural models.
- Conclusion: neural gains are modest and regime-dependent; GJR-GARCH remains a strong,
  hard-to-beat benchmark.

### Range-based proxy and HAR (notebook 11)
Squared returns are an unbiased but very noisy proxy for true variance, which weakens every test
above. Notebook 11 builds a less noisy proxy from daily open, high, low and close prices
(Garman-Klass plus the squared overnight return), rescaled to the level of squared returns, and
adds HAR models (Corsi, 2009).

| Model | QLIKE (squared returns) | QLIKE (range-based) | Rank (range-based) |
|---|---|---|---|
| Hybrid GARCH-LSTM (QLIKE-trained) | **0.5546** | **0.5463** | 1 |
| GJR-GARCH (rolling) | 0.5788 | 0.5583 | 2 |
| GJR-GARCH (fixed) | 0.5829 | 0.5610 | 3 |
| EGARCH (rolling) | 0.5856 | 0.5671 | 4 |
| EGARCH (fixed) | 0.5882 | 0.5683 | 5 |
| log-HAR (range inputs) | 0.5654 | 0.5708 | 6 |
| HAR (range inputs) | 0.5676 | 0.5746 | 7 |
| LSTM (QLIKE-trained) | 0.5687 | 0.5793 | 8 |
| GARCH(1,1) (rolling) | 0.6370 | 0.6036 | 9 |
| GARCH(1,1) (fixed) | 0.6408 | 0.6055 | 10 |
| HAR (squared-return inputs) | 0.6386 | 0.6194 | 11 |
| Rolling 21-day variance (baseline) | 0.7476 | 0.6922 | 12 |

- QLIKE levels are not comparable across proxies, but rankings are (Spearman correlation 0.70).
- Under the range-based proxy, rolling GJR-GARCH still beats fixed GJR-GARCH (p = 0.006), while
  the hybrid's advantage over GJR-GARCH becomes borderline (p = 0.10). HAR and log-HAR are not
  distinguishable from GJR-GARCH; HAR with squared-return inputs is significantly worse (p < 0.001).
- The 90% Model Confidence Set contains the same seven models under both proxies (EGARCH fixed
  and rolling, rolling GJR-GARCH, HAR, log-HAR, LSTM and hybrid), and Mincer-Zarnowitz
  regressions do not reject calibration for most models under either proxy.
- The better proxy gave only a modest gain in statistical power: Diebold-Mariano statistics were
  about 1.3x larger, and calibration standard errors relative to the slope improved by about 8%.
- **Scaling pitfall:** the ratio of squared returns to the range estimator drifted from 1.4–2.0
  in 2010–2016 to about 1.1 by 2024–2026, and Yahoo Finance's index data has stale opening
  prices on 29% of days in 2012–2013. A fixed training-period scale factor (1.51 vs. 1.12 in the
  test period) inflated the proxy by about 34%, which produced a spurious ranking reversal
  (HAR moved from 8th to 1st) and apparent miscalibration. A rolling 252-day scale factor,
  computed from past data only, brought the test-period ratio to 0.99 and removed both artifacts.

## Approach

1. **Data:** Daily S&P 500 (`^GSPC`) prices from 2010 via `yfinance`; log returns in percent. Notebook 09 adds four other assets.
2. **Exploratory analysis:** ADF stationarity test, return distribution (skew, kurtosis, Q-Q plot), autocorrelation of returns vs. squared returns, Ljung-Box and Engle ARCH-LM tests.
3. **Models:** GARCH(1,1), EGARCH and GJR-GARCH with Student-t errors; HAR and log-HAR on daily, weekly and monthly averages of realized variance; an LSTM on lagged returns and volatility features; and a hybrid LSTM that learns a log-multiplier on the GJR-GARCH forecast, starting from exactly the GARCH forecast. The final networks are trained with a QLIKE loss and averaged over 5 random seeds.
4. **Evaluation:** Chronological 80/20 split. Parameters and scaling are estimated on training data only, and each forecast uses only earlier information. Models are scored with RMSE, MAE and QLIKE and compared with Diebold-Mariano tests, the Model Confidence Set and Mincer-Zarnowitz calibration regressions, then checked across assets, rolling test windows and two volatility proxies.

## Project structure

```
.
├── notebooks/
│   ├── 01_data_collection.ipynb    # download prices, log returns, realized volatility
│   ├── 02_eda.ipynb                # stationarity, fat tails, ARCH effects
│   ├── 03_garch_models.ipynb       # GARCH / EGARCH / GJR-GARCH, fixed-parameter forecasts
│   ├── 04_lstm_model.ipynb         # LSTM forecasts and comparison
│   ├── 05_rolling_garch_dm.ipynb   # rolling re-estimation, Diebold-Mariano tests
│   ├── 06_hybrid_garch_lstm.ipynb  # hybrid GARCH-LSTM
│   ├── 07_improved_nn.ipynb        # QLIKE-trained LSTM and hybrid, 5-seed ensembles
│   ├── 08_mcs_calibration.ipynb    # Model Confidence Set, Mincer-Zarnowitz calibration
│   ├── 09_replication.ipynb        # replication across six assets
│   ├── 10_rolling_origin.ipynb     # five rolling test windows, including COVID
│   └── 11_range_proxy_har.ipynb    # range-based proxy, HAR models, proxy-scaling check
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

The `data/` folder is not tracked, so run the notebooks **in order** (01 to 11) to generate it:

- Notebook 01 creates `GSPC_processed.csv`.
- Notebook 03 saves `garch_forecasts.csv` and notebook 04 saves `lstm_forecasts.csv`, which notebooks 05 and 06 read.
- Notebooks 05–07 save the rolling GARCH, hybrid and QLIKE-trained network forecasts, which notebooks 08 and 11 read.
- Notebooks 09 and 10 download their own data and retrain all models, so they take several minutes each.

A dev container configuration is included (`.devcontainer/`), so the project can also be opened in GitHub Codespaces or VS Code Dev Containers with dependencies installed automatically.

Prices are downloaded from Yahoo Finance up to the current date, so results will shift slightly from those above. LSTM results also vary slightly between runs even with fixed seeds, which is why the final networks are reported as seed ensembles.

## Limitations

- Even the range-based proxy is built from daily data. Intraday realized variance (for example, from 5-minute returns) would give considerably more statistical power.
- The replication (notebook 09) and rolling-origin evaluation (notebook 10) still use the squared-return proxy and 3-seed ensembles.
- Many pairwise Diebold-Mariano tests were run. The Model Confidence Set addresses this within each comparison, but borderline p-values (0.02 to 0.10) should be read cautiously.
- The hybrid's correction is clipped to a factor of e^±4, and the neural models' failure during the COVID crash has not yet been diagnosed.
- The mean model is a constant, although returns show slight negative autocorrelation at lag 1.

## Possible extensions

- Rerun the rolling-origin evaluation with the corrected range-based proxy and HAR added
- Diagnose the neural models' failure during the COVID crash (clipped corrections, inputs outside the training range)
- Add range-based variance as an input feature to the hybrid
- Value-at-Risk backtesting (Kupiec and Christoffersen tests)
- Intraday realized variance

## Tools

Python, pandas, NumPy, statsmodels, arch, scikit-learn, TensorFlow/Keras, matplotlib

## Author

Lauren Berning