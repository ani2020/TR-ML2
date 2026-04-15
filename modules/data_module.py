"""
Data Module - Plugin architecture for data loading with local caching.
Supports DuckDB (primary), CSV, Yahoo Finance, and extensible external sources.

Data source priority
--------------------
  1. DuckDB  (primary — futures_continuous.duckdb or market_data.duckdb)
  2. CSV     (static file, for one-off runs)
  3. Yahoo   (fallback only if explicitly configured — not automatic)

DuckDB plugin modes
-------------------
  "futures"  — reads futures_continuous.duckdb :: futures_continuous
               uses adj_open/adj_high/adj_low/adj_close as OHLC
               also carries spot, oi, chgoi, basis, basis_pct, dte_norm,
               roll_vol_20, roll_vol_60 as passthrough extra columns

  "index"    — reads market_data.duckdb :: index_ohlcv
               standard open/high/low/close

  "equity"   — reads market_data.duckdb :: equity_ohlcv
               standard open/high/low/close

VIX data is always joined from market_data.duckdb :: india_vix when
available, regardless of the primary mode. VIX columns are used by
the new vix_* feature functions.
"""

import os
import hashlib
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB file paths — override via DuckDBPlugin constructor if needed
# ---------------------------------------------------------------------------
_DEFAULT_FUTURES_DB     = "../futures_continuous.duckdb"
_DEFAULT_MARKET_DB      = "../market_data.duckdb"
_DEFAULT_FUTURES_METHOD = "backward_ratio"


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
        Extra columns (futures mode): Spot, FutOI, FutChgOI, Basis, BasisPct,
          DteNorm, RollVol20, RollVol60, vix_close, vix_prev_close, etc.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ─────────────────────────────────────────────
# DuckDB Plugin  (primary source)
# ─────────────────────────────────────────────
class DuckDBPlugin(DataSourcePlugin):
    """
    Reads OHLCV data from local DuckDB databases.

    Modes
    -----
    "futures"  — futures_continuous.duckdb :: futures_continuous
                 adj_* price columns used as OHLC
    "index"    — market_data.duckdb :: index_ohlcv
    "equity"   — market_data.duckdb :: equity_ohlcv

    VIX is always joined from market_data.duckdb :: india_vix when present.
    Missing VIX rows are forward-filled (max 5 days) then left as NaN.

    Parameters
    ----------
    mode            : "futures" | "index" | "equity"
    futures_db      : path to futures_continuous.duckdb
    market_db       : path to market_data.duckdb
    futures_method  : adjustment method row filter (default: backward_ratio)
    """

    def __init__(
        self,
        mode:           str = "futures",
        futures_db:     str = _DEFAULT_FUTURES_DB,
        market_db:      str = _DEFAULT_MARKET_DB,
        futures_method: str = _DEFAULT_FUTURES_METHOD,
    ):
        assert mode in ("futures", "index", "equity"), \
            f"DuckDBPlugin mode must be 'futures', 'index', or 'equity', got '{mode}'"
        self.mode           = mode
        self.futures_db     = futures_db
        self.market_db      = market_db
        self.futures_method = futures_method

    @property
    def name(self) -> str:
        return f"duckdb_{self.mode}"

    def fetch(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb required: pip install duckdb")

        logger.info(f"[DuckDB] Fetching {symbol} ({self.mode}) from {start} to {end}")

        if self.mode == "futures":
            df = self._fetch_futures(duckdb, symbol, start, end)
        elif self.mode == "index":
            df = self._fetch_index(duckdb, symbol, start, end)
        else:
            df = self._fetch_equity(duckdb, symbol, start, end)

        if df.empty:
            raise ValueError(
                f"[DuckDB] No data for {symbol} ({self.mode}) [{start} → {end}].\n"
                f"Run 'python continuous_futures.py fetch' or "
                f"'python market_data_eod.py init' first."
            )

        # Join VIX from market_data.duckdb
        df = self._join_vix(duckdb, df, start, end)

        logger.info(f"[DuckDB] Retrieved {len(df)} rows, {len(df.columns)} columns for {symbol}")
        return df

    # ── Fetch helpers ─────────────────────────

    def _fetch_futures(self, duckdb, symbol: str, start: str, end: str) -> pd.DataFrame:
        if not os.path.exists(self.futures_db):
            raise FileNotFoundError(
                f"futures_continuous.duckdb not found at '{self.futures_db}'. "
                f"Run: python continuous_futures.py fetch -s {symbol} ..."
            )
        con = duckdb.connect(self.futures_db, read_only=True)
        try:
            df = con.execute("""
                SELECT timestamp, expiry,
                       adj_open, adj_high, adj_low, adj_close, adj_prevclose,
                       volume, oi, chgoi, lot,
                       spot, basis, basis_pct, dte_norm,
                       roll_vol_20, roll_vol_60,
                       log_ret, spot_ret
                FROM futures_continuous
                WHERE symbol = ?
                  AND method = ?
                  AND timestamp >= ?::DATE
                  AND timestamp <= ?::DATE
                ORDER BY timestamp
            """, [symbol.upper(), self.futures_method, start, end]).df()
        finally:
            con.close()

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df.index.name = None

        # Rename to toolkit standard OHLCV + passthrough extras
        df = df.rename(columns={
            "adj_open":      "Open",
            "adj_high":      "High",
            "adj_low":       "Low",
            "adj_close":     "Close",
            "adj_prevclose": "PrevClose",
            "volume":        "Volume",
            "oi":            "FutOI",
            "chgoi":         "FutChgOI",
            "spot":          "Spot",
            "basis":         "Basis",
            "basis_pct":     "BasisPct",
            "dte_norm":      "DteNorm",
            "roll_vol_20":   "RollVol20",
            "roll_vol_60":   "RollVol60",
            "log_ret":       "FutLogRet",
            "spot_ret":      "SpotRet",
        })
        df.dropna(subset=["Close"], inplace=True)
        return df

    def _fetch_index(self, duckdb, symbol: str, start: str, end: str) -> pd.DataFrame:
        if not os.path.exists(self.market_db):
            raise FileNotFoundError(
                f"market_data.duckdb not found at '{self.market_db}'. "
                f"Run: python market_data_eod.py init --index {symbol} ..."
            )
        # Resolve short name (NIFTY) to full index name (NIFTY 50)
        _INDEX_MAP = {
            "NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FINANCIAL SERVICES",
            "MIDCPNIFTY": "NIFTY MIDCAP 50", "NIFTYNXT50": "NIFTY NEXT 50",
            "SENSEX": "SENSEX", "NIFTYIT": "NIFTY IT",
        }
        index_name = _INDEX_MAP.get(symbol.upper(), symbol.upper())

        con = duckdb.connect(self.market_db, read_only=True)
        try:
            df = con.execute("""
                SELECT date, open, high, low, close, prev_close, volume
                FROM index_ohlcv
                WHERE index_name = ?
                  AND date >= ?::DATE
                  AND date <= ?::DATE
                ORDER BY date
            """, [index_name, start, end]).df()
        finally:
            con.close()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = None
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "prev_close": "PrevClose", "volume": "Volume",
        })
        df.dropna(subset=["Close"], inplace=True)
        return df

    def _fetch_equity(self, duckdb, symbol: str, start: str, end: str) -> pd.DataFrame:
        if not os.path.exists(self.market_db):
            raise FileNotFoundError(
                f"market_data.duckdb not found at '{self.market_db}'. "
                f"Run: python market_data_eod.py init --equity {symbol} ..."
            )
        con = duckdb.connect(self.market_db, read_only=True)
        try:
            df = con.execute("""
                SELECT date, open, high, low, close, volume
                FROM equity_ohlcv
                WHERE symbol = ?
                  AND date >= ?::DATE
                  AND date <= ?::DATE
                ORDER BY date
            """, [symbol.upper(), start, end]).df()
        finally:
            con.close()

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.index.name = None
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        df.dropna(subset=["Close"], inplace=True)
        return df

    def _join_vix(self, duckdb, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        """Left-join India VIX onto the main DataFrame by date index."""
        if not os.path.exists(self.market_db):
            logger.warning("[DuckDB] market_data.duckdb not found — VIX features will be NaN")
            return df
        try:
            con = duckdb.connect(self.market_db, read_only=True)
            vix = con.execute("""
                SELECT date,
                       open  AS vix_open,
                       high  AS vix_high,
                       low   AS vix_low,
                       close AS vix_close,
                       prev_close AS vix_prev_close
                FROM india_vix
                WHERE date >= ?::DATE AND date <= ?::DATE
                ORDER BY date
            """, [start, end]).df()
            con.close()

            if vix.empty:
                logger.warning("[DuckDB] No VIX data in india_vix — VIX features will be NaN")
                return df

            vix["date"] = pd.to_datetime(vix["date"])
            vix = vix.set_index("date")
            vix.index.name = None

            df = df.join(vix, how="left")
            # Forward-fill up to 5 days for weekends / public holidays
            for col in ["vix_open", "vix_high", "vix_low", "vix_close", "vix_prev_close"]:
                if col in df.columns:
                    df[col] = df[col].ffill(limit=5)

            logger.info(f"[DuckDB] VIX joined: {len(vix)} rows")
        except Exception as exc:
            logger.warning(f"[DuckDB] VIX join failed: {exc} — continuing without VIX")
        return df


# ─────────────────────────────────────────────
# Yahoo Finance Plugin  (explicit fallback)
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


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════
class FeatureEngineer:
    """
    Computes technical indicators and derived features.
    Plug-and-play: add new feature functions via @FeatureEngineer.register(name).
    """

    FEATURE_REGISTRY: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str):
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


# ═══════════════════════════════════════════════════════════════════════════
# REGISTERED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Core price ────────────────────────────────────────────────────────────

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
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

@FeatureEngineer.register("macd")
def feat_macd(df):
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df

@FeatureEngineer.register("bollinger")
def feat_bollinger(df):
    sma = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = sma + 2 * std
    df["bb_lower"] = sma - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma + 1e-10)
    return df

@FeatureEngineer.register("adx")
def feat_adx(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    dm_plus  = (high - high.shift()).clip(lower=0)
    dm_minus = (low.shift() - low).clip(lower=0)
    dm_plus[dm_plus   < dm_minus] = 0
    dm_minus[dm_minus < dm_plus]  = 0
    di_plus  = 100 * dm_plus.rolling(14).mean()  / (atr + 1e-10)
    di_minus = 100 * dm_minus.rolling(14).mean() / (atr + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    df["adx"]      = dx.rolling(14).mean()
    df["di_plus"]  = di_plus
    df["di_minus"] = di_minus
    return df

@FeatureEngineer.register("volume_ratio")
def feat_volume_ratio(df):
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"]  = df["Volume"] / (df["volume_sma_20"] + 1e-10)
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

# ── VWAP ──────────────────────────────────────────────────────────────────

@FeatureEngineer.register("vwap")
def feat_vwap(df):
    """
    Rolling 20-bar VWAP, deviation, and binary flag.
    vwap_20          — rolling VWAP price level
    vwap_deviation   — (Close − vwap_20) / vwap_20
    price_above_vwap — 1 when Close > vwap_20
    """
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_tp_vol = (tp * df["Volume"]).rolling(20).sum()
    cum_vol    = df["Volume"].rolling(20).sum()
    df["vwap_20"]          = cum_tp_vol / (cum_vol + 1e-10)
    df["vwap_deviation"]   = (df["Close"] - df["vwap_20"]) / (df["vwap_20"] + 1e-10)
    df["price_above_vwap"] = (df["Close"] > df["vwap_20"]).astype(int)
    return df

# ── ATR ───────────────────────────────────────────────────────────────────

@FeatureEngineer.register("atr")
def feat_atr(df):
    """
    ATR-14 (Wilder's smoothing), percentage ATR, ATR-based bands.
    atr_14            — ATR in price units
    atr_pct / atr_norm — ATR / Close (dimensionless)
    atr_bands_upper/lower — Close ± 2 × atr_14
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]          = tr.ewm(alpha=1/14, adjust=False).mean()
    df["atr_pct"]         = df["atr_14"] / (close + 1e-10)
    df["atr_norm"]        = df["atr_pct"]
    df["atr_bands_upper"] = close + 2 * df["atr_14"]
    df["atr_bands_lower"] = close - 2 * df["atr_14"]
    return df

# ── VIX features ──────────────────────────────────────────────────────────

@FeatureEngineer.register("vix_features")
def feat_vix(df):
    """
    India VIX-derived features. Requires vix_close / vix_prev_close columns
    joined by DuckDBPlugin. Gracefully produces NaN when absent.

    vix_level          — India VIX close
    vix_returns        — log(vix_close / vix_prev_close)
    vix_change         — alias of vix_returns (used by HMM registry)
    vix_sma20          — 20-day SMA of VIX
    vix_vs_sma20       — vix_level / vix_sma20 − 1  (above/below trend)
    vix_percentile_252 — rolling 252-day percentile rank  (0–100)
    vix_regime         — 1 when VIX > 20 (high-vol environment)
    vix_divergence     — vix_returns + fut_log_ret  (cross-asset divergence)
    fut_price_vix_sig  — vix_returns × fut_log_ret  (interaction signal)
    """
    _vix_nan_cols = [
        "vix_level", "vix_returns", "vix_change", "vix_sma20",
        "vix_vs_sma20", "vix_percentile_252", "vix_regime",
        "vix_divergence", "fut_price_vix_sig",
    ]

    if "vix_close" not in df.columns:
        logger.warning("[Features] vix_close not in DataFrame — VIX features will be NaN")
        for col in _vix_nan_cols:
            df[col] = np.nan
        return df

    vix      = df["vix_close"]
    prev_vix = df["vix_prev_close"] if "vix_prev_close" in df.columns else vix.shift(1)

    df["vix_level"]   = vix
    df["vix_returns"] = np.log(vix / (prev_vix + 1e-10))
    df["vix_change"]  = df["vix_returns"]   # HMM registry alias

    df["vix_sma20"]    = vix.rolling(20).mean()
    df["vix_vs_sma20"] = vix / (df["vix_sma20"] + 1e-10) - 1.0

    # Rolling 252-day percentile rank (where is today's VIX vs its own history?)
    df["vix_percentile_252"] = (
        vix.rolling(252, min_periods=60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )
    df["vix_regime"] = (vix > 20.0).astype(int)

    # Determine which futures return column is available
    fut_ret = None
    for candidate in ("FutLogRet", "fut_log_ret", "log_returns"):
        if candidate in df.columns:
            fut_ret = df[candidate].fillna(0)
            break

    if fut_ret is not None:
        vr = df["vix_returns"].fillna(0)
        df["vix_divergence"]    = vr + fut_ret     # additive cross-asset spread
        df["fut_price_vix_sig"] = vr * fut_ret     # interaction (same sign = risk-off)
    else:
        df["vix_divergence"]    = np.nan
        df["fut_price_vix_sig"] = np.nan

    return df

# ── Futures-specific features ─────────────────────────────────────────────

@FeatureEngineer.register("futures_features")
def feat_futures(df):
    """
    Features derived from continuous futures series.
    No-ops gracefully (NaN) when futures-specific columns are absent.

    fut_log_ret        — log(adj_close / adj_prev_close)
    fut_ret_5d_vol     — 5-day rolling annualised vol of fut_log_ret
    fut_ret_10d_vol    — 10-day rolling annualised vol
    fut_ret_20d_vol    — 20-day rolling annualised vol
    fut_zscore_20      — (fut_log_ret − 20d mean) / 20d std
    ret_divergence     — fut_log_ret − spot_log_ret
    basis_pct          — (futures − spot) / spot  (pass-through from DuckDB)
    dte_norm           — days-to-expiry / 30  (pass-through)
    roll_vol_20/60     — realised vol from continuous series (pass-through)
    """
    # fut_log_ret: prefer pre-computed from DuckDB, else compute from Close
    if "FutLogRet" in df.columns:
        df["fut_log_ret"] = df["FutLogRet"]
    else:
        df["fut_log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    lr = df["fut_log_ret"]

    df["fut_ret_5d_vol"]  = lr.rolling(5).std()  * np.sqrt(252)
    df["fut_ret_10d_vol"] = lr.rolling(10).std() * np.sqrt(252)
    df["fut_ret_20d_vol"] = lr.rolling(20).std() * np.sqrt(252)

    mu20 = lr.rolling(20).mean()
    sd20 = lr.rolling(20).std()
    df["fut_zscore_20"] = (lr - mu20) / (sd20 + 1e-10)

    # Return divergence: futures log return − spot log return
    if "SpotRet" in df.columns:
        df["ret_divergence"] = lr - df["SpotRet"].fillna(0)
    elif "Spot" in df.columns:
        spot_ret = np.log(df["Spot"] / df["Spot"].shift(1))
        df["ret_divergence"] = lr - spot_ret
    else:
        df["ret_divergence"] = np.nan

    # Pass-through columns already computed in continuous_futures.py
    for db_col, feat_col in [
        ("BasisPct", "basis_pct"),
        ("DteNorm",  "dte_norm"),
        ("RollVol20","roll_vol_20"),
        ("RollVol60","roll_vol_60"),
    ]:
        if db_col in df.columns and feat_col not in df.columns:
            df[feat_col] = df[db_col]

    return df

# ── Open Interest features ────────────────────────────────────────────────

@FeatureEngineer.register("oi_features")
def feat_oi(df):
    """
    Open interest features from the futures series.
    Requires FutOI column (present when DuckDBPlugin mode='futures').

    oi_log_change      — log(OI / OI_prev), clipped ±0.5
    oi_zscore_20       — 20-day z-score of oi_log_change
    fut_price_oi_sig   — fut_log_ret × oi_log_change  (price × OI interaction)
    trend_strength     — sign(fut_log_ret) × sign(oi_log_change)
                         +1 = price↑ & OI↑  (trend confirmation)
                         −1 = price↓ & OI↓  (short covering / weak demand)
                          0 = divergence
    oi_sma_20          — 20-day SMA of raw OI
    oi_vs_sma          — OI / oi_sma_20 − 1  (OI expansion / contraction)
    """
    if "FutOI" not in df.columns:
        logger.warning("[Features] FutOI not in DataFrame — OI features will be NaN")
        for col in ["oi_log_change", "oi_zscore_20", "fut_price_oi_sig",
                    "trend_strength", "oi_sma_20", "oi_vs_sma"]:
            df[col] = np.nan
        return df

    oi = df["FutOI"].replace(0, np.nan)

    raw_oi_chg          = np.log(oi / oi.shift(1))
    df["oi_log_change"] = raw_oi_chg.clip(-0.5, 0.5)

    mu_oi = df["oi_log_change"].rolling(20).mean()
    sd_oi = df["oi_log_change"].rolling(20).std()
    df["oi_zscore_20"] = (df["oi_log_change"] - mu_oi) / (sd_oi + 1e-10)

    # Use fut_log_ret if already computed, else fall back to close pct change
    fut_ret = df.get("fut_log_ret", df["Close"].pct_change())
    df["fut_price_oi_sig"] = fut_ret.fillna(0) * df["oi_log_change"].fillna(0)

    df["trend_strength"] = (
        np.sign(fut_ret.fillna(0)) * np.sign(df["oi_log_change"].fillna(0))
    ).astype(float)

    df["oi_sma_20"] = oi.rolling(20).mean()
    df["oi_vs_sma"] = oi / (df["oi_sma_20"] + 1e-10) - 1.0

    return df


# ─────────────────────────────────────────────
# Data Module (Orchestrator)
# ─────────────────────────────────────────────
class DataModule:
    """
    Orchestrates data fetching, feature engineering, and caching.

    Default source: duckdb_futures
    DuckDB is always local so raw fetch is never cached (feature cache still applies).

    Usage
    -----
        dm = DataModule(cache_dir="cache/")

        # Continuous futures (default)
        df = dm.get_feature_data("NIFTY", "2015-01-01", "2026-04-13",
                                  source="duckdb_futures")

        # NSE index spot prices from DuckDB
        df = dm.get_feature_data("NIFTY 50", "2015-01-01", "2026-04-13",
                                  source="duckdb_index")

        # Yahoo Finance (explicit opt-in — register first)
        dm.register_plugin("yahoo", YahooFinancePlugin())
        df = dm.get_feature_data("^NSEI", "2015-01-01", "2026-04-13",
                                  source="yahoo")
    """

    def __init__(
        self,
        cache_dir:   str = "cache/",
        futures_db:  str = _DEFAULT_FUTURES_DB,
        market_db:   str = _DEFAULT_MARKET_DB,
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._plugins: Dict[str, DataSourcePlugin] = {}

        # Register all three DuckDB sources by default
        for mode in ("futures", "index", "equity"):
            self.register_plugin(
                f"duckdb_{mode}",
                DuckDBPlugin(mode=mode, futures_db=futures_db, market_db=market_db)
            )

    def register_plugin(self, name: str, plugin: DataSourcePlugin):
        self._plugins[name] = plugin
        logger.info(f"[DataModule] Registered plugin: {name} ({plugin.name})")

    def _cache_key(self, symbol: str, start: str, end: str, suffix: str) -> str:
        key = f"{symbol}_{start}_{end}_{suffix}"
        return os.path.join(self.cache_dir, f"{key}.parquet")

    def get_raw_data(
        self,
        symbol:        str,
        start:         str,
        end:           str,
        source:        str  = "duckdb_futures",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        # DuckDB is local — always read fresh (no raw-level cache needed)
        if source.startswith("duckdb"):
            if source not in self._plugins:
                raise ValueError(
                    f"Source '{source}' not registered. Available: {list(self._plugins)}"
                )
            return self._plugins[source].fetch(symbol, start, end)

        # Non-DuckDB sources: use parquet cache
        cache_path = self._cache_key(symbol, start, end, f"raw_{source}")
        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"[DataModule] Loading raw data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        if source not in self._plugins:
            raise ValueError(
                f"Unknown source: '{source}'. Available: {list(self._plugins)}"
            )
        df = self._plugins[source].fetch(symbol, start, end)
        df.to_parquet(cache_path)
        logger.info(f"[DataModule] Cached raw data → {cache_path}")
        return df

    def get_feature_data(
        self,
        symbol:        str,
        start:         str,
        end:           str,
        source:        str         = "duckdb_futures",
        features:      Optional[list] = None,
        force_refresh: bool        = False,
    ) -> pd.DataFrame:
        feat_key   = hashlib.md5(str(sorted(features or [])).encode()).hexdigest()[:8]
        cache_path = self._cache_key(symbol, start, end, f"features_{source}_{feat_key}")

        if not force_refresh and os.path.exists(cache_path):
            logger.info(f"[DataModule] Loading feature data from cache: {cache_path}")
            return pd.read_parquet(cache_path)

        raw = self.get_raw_data(symbol, start, end, source, force_refresh)
        df  = FeatureEngineer.compute(raw, features)
        df.to_parquet(cache_path)
        logger.info(
            f"[DataModule] Cached feature data → {cache_path} "
            f"({len(df)} rows, {len(df.columns)} cols)"
        )
        return df

    def list_cached_files(self) -> List[str]:
        files = os.listdir(self.cache_dir)
        logger.info(f"[DataModule] Cache contents: {files}")
        return files
