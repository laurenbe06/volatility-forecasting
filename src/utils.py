"""Reusable helpers for the volatility forecasting project."""
from pathlib import Path

import numpy as np # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download_prices(ticker="^GSPC", start="2010-01-01", end=None, save=True):
    """Download daily OHLCV data and (optionally) cache it as CSV in data/."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # Newer yfinance versions return MultiIndex columns even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if save:
        DATA_DIR.mkdir(exist_ok=True)
        name = ticker.replace("^", "")
        df.to_csv(DATA_DIR / f"{name}_prices.csv")
    return df


def load_prices(ticker="^GSPC"):
    name = ticker.replace("^", "")
    return pd.read_csv(DATA_DIR / f"{name}_prices.csv", index_col=0, parse_dates=True)


def add_returns(df, price_col="Close"):
    """Add log returns (in %, which is what the `arch` package prefers)."""
    df = df.copy()
    df["log_return"] = 100 * np.log(df[price_col] / df[price_col].shift(1))
    return df.dropna()


def realized_vol(returns, window=21, annualize=True):
    """Rolling realized volatility: std of returns over `window` trading days."""
    vol = returns.rolling(window).std()
    return vol * np.sqrt(252) if annualize else vol


def train_test_split_ts(series, test_frac=0.2):
    """Chronological split - never shuffle time series."""
    split = int(len(series) * (1 - test_frac))
    return series.iloc[:split], series.iloc[split:]


def qlike(realized_var, forecast_var):
    """QLIKE loss (lower is better). Inputs are variances, not std devs."""
    r = np.asarray(realized_var, dtype=float)
    f = np.asarray(forecast_var, dtype=float)
    return np.mean(np.log(f) + r / f)