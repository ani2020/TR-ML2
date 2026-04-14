"""
Data Module - Plugin architecture for data loading with local caching.
Supports CSV, Yahoo Finance, and extensible external sources.
"""

import os
import json
import hashlib
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Base Plugin Interface
# ─────────────────────────────────────────────
class DataSourcePlugin(ABC):
    """Abstract base class for all data source plugins."""

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ─────────────────────────────────────────────
# Yahoo Finance Plugin
# ─────────────────────────────────────────────
class YahooFinancePlugin(DataSourcePlugin):
    """Fetches daily OHLCV data from Yahoo Finance via yfinance."""

    @property
    def name(self) -> str:
        return "yahoo_finance"

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        logger.info(f"[YahooFinance] Fetching {symbol} from {start} to {end}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} [{start} → {end}]")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        logger.info(f"[YahooFinance] Retrieved {len(df)} rows for {symbol}")
        return df


# ─────────────────────────────────────────────
# CSV Plugin
# ─────────────────────────────────────────────
class CSVPlugin(DataSourcePlugin):
    """
    Loads data from a CSV file.
    Expected CSV columns: Date, Open, High, Low, Close, Volume
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    @property
    def name(self) -> str:
        return "csv"

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        logger.info(f"[CSV] Loading {self.filepath}")
        df = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index)
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.loc[start:end, ["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
class FeatureEngineer:
    """
    Computes technical indicators and derived features.
    Plug-and-play: add new feature functions to FEATURE_REGISTRY.
    """

    FEATURE_REGISTRY: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a feature function."""
        def decorator(fn):
            cls.FEATURE_REGISTRY[name] = fn
            return fn
        return decorator

    @classmethod
    def compute(cls, df: pd.DataFrame, features: Optional[list] = None) -> pd.DataFrame:
        out = df.copy()
        selected = features or list(cls.FEATURE_REGISTRY.keys())
        for fname in selected:
            if fname in cls.FEATURE_REGISTRY:
                try:
                    out = cls.FEATURE_REGISTRY[fname](out)
                    logger.debug(f"[Features] Computed: {fname}")
                except Exception as e:
                    logger.warning(f"[Features] Failed to compute {fname}: {e}")
            else:
                logger.warning(f"[Features] Unknown feature: {fname}")
        out.dropna(inplace=True)
        return out


# ─── Register Default Features ────────────────
@FeatureEngineer.register("returns")
def feat_returns(df):
    df["returns"] = df["Close"].pct_change()
    return df

@FeatureEngineer.register("log_returns")
def feat_log_returns(df):
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df

@FeatureEngineer.register("sma_20")
def feat_sma20(df):
    df["sma_20"] = df["Close"].rolling(20).mean()
    return df

@FeatureEngineer.register("sma_50")
def feat_sma50(df):
    df["sma_50"] = df["Close"].rolling(50).mean()
    return df

@FeatureEngineer.register("ema_50")
def feat_ema50(df):
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    return df

@FeatureEngineer.register("rsi_14")
def feat_rsi(df):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

@FeatureEngineer.register("macd")
def feat_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

@FeatureEngineer.register("bollinger")
def feat_bollinger(df):
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    return df

@FeatureEngineer.register("adx")
def feat_adx(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    dm_plus = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0
    mask2 = dm_minus < dm_plus
    dm_minus[mask2] = 0
    di_plus = 100 * dm_plus.rolling(14).mean() / (atr + 1e-10)
    di_minus = 100 * dm_minus.rolling(14).mean() / (atr + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    df["adx"] = dx.rolling(14).mean()
    df["di_plus"] = di_plus
    df["di_minus"] = di_minus
    return df

@FeatureEngineer.register("volume_ratio")
def feat_volume_ratio(df):
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + 1e-10)
    return df

@FeatureEngineer.register("volatility_20")
def feat_vol20(df):
    if "returns" not in df.columns:
        df["returns"] = df["Close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    return df

@FeatureEngineer.register("momentum_10")
def feat_momentum(df):
    df["momentum_10"] = df["Close"].pct_change(10)
    return df

@FeatureEngineer.register("price_above_ema50")
def feat_price_ema(df):
    if "ema_50" not in df.columns:
        df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["price_above_ema50"] = (df["Close"] > df["ema_50"]).astype(int)
    return df

# ──────────────────────────────────────────────────────────────────────────
# VWAP  (Volume-Weighted Average Price)
# ──────────────────────────────────────────────────────────────────────────
# VWAP = cumulative(price × volume) / cumulative(volume)
# We compute a rolling 20-bar VWAP (intraday anchoring is not possible on
# daily data; a rolling window is the standard daily-bar equivalent).
#
# Derived columns produced:
#   vwap_20           — rolling 20-bar VWAP price level
#   vwap_deviation    — (Close - vwap_20) / vwap_20  →  % above/below VWAP
#   price_above_vwap  — 1 when Close > vwap_20, else 0  (binary flag)
# ──────────────────────────────────────────────────────────────────────────
@FeatureEngineer.register("vwap")
def feat_vwap(df):
    """
    Rolling 20-bar VWAP and deviation from VWAP.
 
    Typical price (tp) = (High + Low + Close) / 3
    VWAP(20)           = sum(tp * Volume, 20) / sum(Volume, 20)
    vwap_deviation     = (Close - vwap_20) / vwap_20
    price_above_vwap   = binary flag
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cumulative_tp_vol = (tp * df["Volume"]).rolling(window=20).sum()
    cumulative_vol    = df["Volume"].rolling(window=20).sum()
 
    df["vwap_20"]          = cumulative_tp_vol / (cumulative_vol + 1e-10)
    df["vwap_deviation"]   = (df["Close"] - df["vwap_20"]) / (df["vwap_20"] + 1e-10)
    df["price_above_vwap"] = (df["Close"] > df["vwap_20"]).astype(int)
    return df

# ──────────────────────────────────────────────────────────────────────────
# ATR  (Average True Range)
# ──────────────────────────────────────────────────────────────────────────
# True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
# ATR(14)    = EMA_14(True Range)   [Wilder's smoothing]
#
# Derived columns produced:
#   atr_14        — raw 14-period ATR in price units
#   atr_pct       — ATR / Close  →  normalised (%) — better for cross-asset
#   atr_norm      — same as atr_pct, alias used by HMM and XGBoost features
#   atr_bands_upper/lower — Close ± 2*ATR  (volatility envelope)
# ──────────────────────────────────────────────────────────────────────────
@FeatureEngineer.register("atr")
def feat_atr(df):
    """
    ATR-14 (Wilder's smoothing), percentage ATR, and ATR-based price bands.
 
    atr_14            — ATR in price units
    atr_pct / atr_norm — ATR / Close (dimensionless, cross-comparable)
    atr_bands_upper   — Close + 2 * atr_14
    atr_bands_lower   — Close - 2 * atr_14
    """
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
 
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
 
    # Wilder's smoothing = EMA with alpha = 1/period
    df["atr_14"]          = tr.ewm(alpha=1/14, adjust=False).mean()
    df["atr_pct"]         = df["atr_14"] / (close + 1e-10)
    df["atr_norm"]        = df["atr_pct"]           # alias for HMM / XGB
    df["atr_bands_upper"] = close + 2 * df["atr_14"]
    df["atr_bands_lower"] = close - 2 * df["atr_14"]
    return df
# ─────────────────────────────────────────────
# Data Module (Orchestrator)
# ─────────────────────────────────────────────
class DataModule:
    """
    Orchestrates data fetching, feature engineering, and caching.

    Usage:
        dm = DataModule(cache_dir="cache/")
        dm.register_plugin("yahoo", YahooFinancePlugin())
        raw_df = dm.get_raw_data("SPY", "2014-01-01", "2024-01-01", source="yahoo")
        feat_df = dm.get_feature_data("SPY", "2014-01-01", "2024-01-01", source="yahoo")
    """

    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._plugins: Dict[str, DataSourcePlugin] = {}
        # Register defaults
        self.register_plugin("yahoo", YahooFinancePlugin())

    def register_plugin(self, name: str, plugin: DataSourcePlugin):
        self._plugins[name] = plugin
        logger.info(f"[DataModule] Registered plugin: {name} ({plugin.name})")

    def _cache_key(self, symbol: str, start: str, end: str, suffix: str) -> str:
        key = f"{symbol}_{start}_{end}_{suffix}"
        return os.path.join(self.cache_dir, f"{key}.parquet")

    def get_raw_data(self, symbol: str, start: str, end: str,
                     source: str = "yahoo", force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._cache_key(symbol, start, end, "raw")
        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"[DataModule] Loading raw data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        if source not in self._plugins:
            raise ValueError(f"Unknown source: {source}. Available: {list(self._plugins)}")

        df = self._plugins[source].fetch(symbol, start, end)
        df.to_parquet(cache_path)
        logger.info(f"[DataModule] Cached raw data → {cache_path}")
        return df

    def get_feature_data(self, symbol: str, start: str, end: str,
                         source: str = "yahoo", features: Optional[list] = None,
                         force_refresh: bool = False) -> pd.DataFrame:
        feat_key = hashlib.md5(str(sorted(features or [])).encode()).hexdigest()[:8]
        cache_path = self._cache_key(symbol, start, end, f"features_{feat_key}")

        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"[DataModule] Loading feature data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        raw = self.get_raw_data(symbol, start, end, source, force_refresh)
        df = FeatureEngineer.compute(raw, features)
        df.to_parquet(cache_path)
        logger.info(f"[DataModule] Cached feature data → {cache_path} ({len(df)} rows, {len(df.columns)} cols)")
        return df

    def list_cached_files(self):
        files = os.listdir(self.cache_dir)
        logger.info(f"[DataModule] Cache contents: {files}")
        return files
