"""
data/data_module.py
Plugin-based data module.
- Defines a BaseDataSource interface.
- Implements YahooFinanceSource and CSVSource.
- FeatureEngine computes all technical features.
- DataModule orchestrates caching and feature computation.
"""

import os
import abc
import hashlib
import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger("trading.data")


# ══════════════════════════════════════════════════════════════
# 1.  BASE DATA SOURCE  (plugin interface)
# ══════════════════════════════════════════════════════════════
class BaseDataSource(abc.ABC):
    """
    All data sources must implement this interface.
    Return a DataFrame with columns:
        Open, High, Low, Close, Volume  (index = DatetimeIndex)
    """

    @abc.abstractmethod
    def fetch(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        ...

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        ...


# ══════════════════════════════════════════════════════════════
# 2.  BUILT-IN SOURCES
# ══════════════════════════════════════════════════════════════
class YahooFinanceSource(BaseDataSource):
    """Pull OHLCV data from Yahoo Finance via yfinance."""

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    def fetch(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        logger.info(f"[YahooFinance] Downloading {ticker}  {start} → {end}  interval={interval}")
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker} ({start} – {end})")

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.dropna(inplace=True)
        logger.info(f"[YahooFinance] Fetched {len(df)} rows for {ticker}")
        return df


class CSVSource(BaseDataSource):
    """
    Load data from a local CSV file.

    Expected CSV structure (configurable via column_map):
        date, open, high, low, close, volume
    """

    DEFAULT_COLUMN_MAP = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }

    def __init__(self, file_path: str, column_map: Optional[dict] = None, date_format: str = "%Y-%m-%d"):
        self.file_path = file_path
        self.column_map = column_map or self.DEFAULT_COLUMN_MAP
        self.date_format = date_format

    @property
    def source_name(self) -> str:
        return f"csv:{self.file_path}"

    def fetch(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        logger.info(f"[CSVSource] Reading {self.file_path}")
        df = pd.read_csv(self.file_path)
        # Rename columns
        rev_map = {v: k.capitalize() for k, v in self.column_map.items()}
        df.rename(columns=rev_map, inplace=True)
        date_col = "Date"
        df[date_col] = pd.to_datetime(df[date_col], format=self.date_format)
        df.set_index(date_col, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.loc[start:end]
        df.dropna(inplace=True)
        return df


# ══════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINE  (plug-and-play)
# ══════════════════════════════════════════════════════════════
class BaseFeature(abc.ABC):
    """Each feature computes one or more columns from a price DataFrame."""

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Receive OHLCV dataframe, return same df with new columns added."""
        ...

    @property
    @abc.abstractmethod
    def feature_names(self) -> List[str]:
        ...


class ReturnFeatures(BaseFeature):
    """Log returns, rolling volatility, momentum."""

    @property
    def feature_names(self):
        return ["log_return", "volatility_5d", "volatility_10d", "volatility_20d",
                "momentum_5d", "momentum_10d", "momentum_20d"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        for w in [5, 10, 20]:
            df[f"volatility_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
            df[f"momentum_{w}d"] = df["Close"].pct_change(w)
        return df


class VolumeFeatures(BaseFeature):
    @property
    def feature_names(self):
        return ["volume_ratio", "volume_sma_20", "volume_zscore"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sma20 = df["Volume"].rolling(20).mean()
        df["volume_sma_20"] = sma20
        df["volume_ratio"] = df["Volume"] / sma20
        df["volume_zscore"] = (df["Volume"] - sma20) / df["Volume"].rolling(20).std()
        return df


class RSIFeature(BaseFeature):
    def __init__(self, period: int = 14):
        self.period = period

    @property
    def feature_names(self):
        return [f"rsi_{self.period}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(self.period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{self.period}"] = 100 - 100 / (1 + rs)
        return df


class MACDFeature(BaseFeature):
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def feature_names(self):
        return ["macd", "macd_signal", "macd_hist"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema_fast = df["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df


class EMAFeature(BaseFeature):
    def __init__(self, periods: List[int] = None):
        self.periods = periods or [20, 50, 200]

    @property
    def feature_names(self):
        return [f"ema_{p}" for p in self.periods]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for p in self.periods:
            df[f"ema_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
        return df


class ADXFeature(BaseFeature):
    def __init__(self, period: int = 14):
        self.period = period

    @property
    def feature_names(self):
        return [f"adx_{self.period}", "plus_di", "minus_di"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high, low, close = df["High"], df["Low"], df["Close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        mask = plus_dm < minus_dm
        plus_dm[mask] = 0
        mask2 = minus_dm < plus_dm
        minus_dm[mask2] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.period).mean()
        df["plus_di"] = 100 * plus_dm.rolling(self.period).mean() / atr
        df["minus_di"] = 100 * minus_dm.rolling(self.period).mean() / atr
        dx = 100 * (df["plus_di"] - df["minus_di"]).abs() / (df["plus_di"] + df["minus_di"])
        df[f"adx_{self.period}"] = dx.rolling(self.period).mean()
        return df


class BollingerFeature(BaseFeature):
    def __init__(self, period: int = 20, std: float = 2.0):
        self.period = period
        self.std = std

    @property
    def feature_names(self):
        return ["bb_upper", "bb_lower", "bb_mid", "bb_pct_b", "bb_width"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mid = df["Close"].rolling(self.period).mean()
        band = df["Close"].rolling(self.period).std() * self.std
        df["bb_upper"] = mid + band
        df["bb_lower"] = mid - band
        df["bb_mid"] = mid
        df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
        return df


class PriceStructureFeature(BaseFeature):
    @property
    def feature_names(self):
        return ["high_low_range", "close_open_range", "upper_shadow", "lower_shadow"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_range"] = (df["Close"] - df["Open"]) / df["Open"]
        df["upper_shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
        df["lower_shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]
        return df


# ══════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════
class FeatureEngine:
    """
    Runs a pipeline of BaseFeature objects on raw OHLCV data.
    New features can be added by registering a BaseFeature subclass.
    """

    def __init__(self):
        self._features: List[BaseFeature] = []
        self._register_defaults()

    def _register_defaults(self):
        self.register(ReturnFeatures())
        self.register(VolumeFeatures())
        self.register(RSIFeature(14))
        self.register(MACDFeature())
        self.register(EMAFeature([20, 50, 200]))
        self.register(ADXFeature(14))
        self.register(BollingerFeature())
        self.register(PriceStructureFeature())

    def register(self, feature: BaseFeature):
        """Plug-in a new feature calculator."""
        self._features.append(feature)
        logger.debug(f"[FeatureEngine] Registered: {feature.__class__.__name__}")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[FeatureEngine] Computing features on {len(df)} rows …")
        for feat in self._features:
            df = feat.compute(df)
        df.dropna(inplace=True)
        logger.info(f"[FeatureEngine] {len(df)} rows after dropna, {len(df.columns)} columns total")
        return df


# ══════════════════════════════════════════════════════════════
# 5.  DATA MODULE
# ══════════════════════════════════════════════════════════════
class DataModule:
    """
    Facade that:
      - resolves the active data source
      - manages local CSV cache
      - runs the feature engine
    """

    def __init__(self, config, source: Optional[BaseDataSource] = None):
        self.cfg = config.data
        self.source = source or YahooFinanceSource()
        self.feature_engine = FeatureEngine()
        Path(self.cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────
    def _raw_cache_path(self) -> str:
        return os.path.join(
            self.cfg.cache_dir,
            self.cfg.raw_cache_file.format(ticker=self.cfg.ticker)
        )

    def _feat_cache_path(self) -> str:
        return os.path.join(
            self.cfg.cache_dir,
            self.cfg.feature_cache_file.format(ticker=self.cfg.ticker)
        )

    # ── public API ───────────────────────────────────────────
    def get_raw(self) -> pd.DataFrame:
        cache = self._raw_cache_path()
        if not self.cfg.force_refresh and os.path.exists(cache):
            logger.info(f"[DataModule] Loading raw data from cache: {cache}")
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            return df

        df = self.source.fetch(
            self.cfg.ticker,
            self.cfg.start_date,
            self.cfg.end_date,
            self.cfg.interval,
        )
        df.to_csv(cache)
        logger.info(f"[DataModule] Raw data cached → {cache}")
        return df

    def get_features(self) -> pd.DataFrame:
        cache = self._feat_cache_path()
        if not self.cfg.force_refresh and os.path.exists(cache):
            logger.info(f"[DataModule] Loading feature data from cache: {cache}")
            df = pd.read_csv(cache, index_col="Date", parse_dates=True)
            return df

        raw = self.get_raw()
        df = self.feature_engine.compute(raw)
        df.to_csv(cache)
        logger.info(f"[DataModule] Feature data cached → {cache}")
        return df

    def get_window(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[start:end].copy()
