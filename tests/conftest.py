"""
tests/conftest.py
=================
Shared pytest fixtures, dependency stubs, and synthetic data generators.

Environment notes
-----------------
This sandbox has pandas 3.0.1 which requires pyarrow at import time.
We install a comprehensive pyarrow stub BEFORE importing pandas.
Run tests with:   PANDAS_FUTURE_INFER_STRING=0 python -m pytest tests/
Or via the helper: python tests/run_tests.py
"""

import sys
import os
import types
import math

# ── Must happen before ANY pandas import ─────────────────────────────────────


def _install_pyarrow_stub():
    """
    Build a pyarrow stub whose classes pass isinstance() checks that
    pandas 3.0 performs internally (pa.DataType, pa.Array, pa.ChunkedArray, etc.)
    and whose compute submodule has all the function names pandas accesses.
    """
    # Real classes so isinstance() works
    class _PA_DataType:
        pass

    class _PA_Array:
        pass

    class _PA_ChunkedArray:
        pass

    class _PA_Table:
        pass

    class _PA_Schema:
        pass

    class _PA_Scalar:
        pass

    class _Auto(_PA_DataType, _PA_Array, _PA_ChunkedArray, _PA_Scalar):
        """
        Catch-all stub that:
          - passes isinstance() for all pyarrow base classes
          - returns itself for any attribute access
          - is callable, iterable (returns empty), subscriptable
        """

        def __getattr__(self, n):
            return _Auto()

        def __call__(self, *a, **kw):
            return _Auto()

        def __iter__(self):
            return iter([])

        def __bool__(self):
            return True

        def __repr__(self):
            return "pa.Auto()"

        def __str__(self):
            return "auto"

        def __eq__(self, other):
            return False

        def __ne__(self, other):
            return True

        def __hash__(self):
            return 0

        def __len__(self):
            return 0

        def __getitem__(self, key):
            return _Auto()

        def __setitem__(self, key, val):
            pass

        def __contains__(self, item):
            return False

        def __int__(self):
            return 0

        def __float__(self):
            return 0.0

        def __index__(self):
            return 0

    _auto = _Auto()

    class _AutoModule(types.ModuleType):
        """Module whose missing attributes resolve to the _Auto singleton."""

        def __getattr__(self, name):
            return _auto

    # Build root pa module
    pa = _AutoModule("pyarrow")
    pa.__version__ = "14.0.0"
    pa.__path__ = []
    pa.__package__ = "pyarrow"
    pa.DataType = _PA_DataType
    pa.Array = _PA_Array
    pa.ChunkedArray = _PA_ChunkedArray
    pa.Table = _PA_Table
    pa.Schema = _PA_Schema
    pa.Scalar = _PA_Scalar

    # Build pa.compute with every function name pandas references
    pc = _AutoModule("pyarrow.compute")
    _compute_fns = [
        "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
        "add", "subtract", "multiply", "divide", "negate",
        "sum", "mean", "min", "max", "count", "any", "all",
        "cast", "if_else", "is_null", "is_valid", "is_nan", "is_inf",
        "and_", "or_", "invert", "and_kleene", "or_kleene",
        "utf8_lower", "utf8_upper", "utf8_length", "utf8_slice_codeunits",
        "binary_length", "list_flatten", "list_parent_indices",
        "sort_indices", "unique", "value_counts", "take", "filter",
        "fill_null", "fill_null_forward", "fill_null_backward",
        "round", "floor", "ceil", "abs", "sign", "power",
        "year", "month", "day", "day_of_week", "hour", "minute", "second",
        "strftime", "strptime",
        "make_struct", "list_slice",
        "index", "array_sort_indices",
    ]
    for fn in _compute_fns:
        setattr(pc, fn, _auto)

    # Register all pyarrow submodule names
    _submodules = [
        "pyarrow", "pyarrow.compute", "pyarrow.lib", "pyarrow.types",
        "pyarrow.compat", "pyarrow.ipc", "pyarrow.parquet",
        "pyarrow.csv", "pyarrow.json", "pyarrow.fs",
        "pyarrow._stubs_not_implemented",
        "pyarrow.vendored", "pyarrow.vendored.version",
        "pyarrow.flight", "pyarrow.dataset", "pyarrow.gandiva",
        "pyarrow.plasma", "pyarrow.cuda",
    ]
    for name in _submodules:
        m = _AutoModule(name)
        m.__path__ = []
        m.__package__ = name.split(".")[0]
        sys.modules[name] = m

    sys.modules["pyarrow"] = pa
    sys.modules["pyarrow.compute"] = pc
    return _auto, _Auto, _PA_DataType


_auto, _Auto, _PA_DataType = _install_pyarrow_stub()


def _install_hmmlearn_stub():
    """
    Proper GaussianHMM stub that bypasses sklearn's StandardScaler
    sparse-matrix isinstance check by patching the scaler at call time.
    """
    import numpy as np

    class _FakeMonitor:
        converged = True
        iter_ = 5

    class _FakeGaussianHMM:
        def __init__(self, n_components=3, covariance_type="full",
                     n_iter=100, random_state=42, verbose=False, **kw):
            self.n_components = n_components
            self.covariance_type = covariance_type
            self.n_iter = n_iter
            self.random_state = random_state
            self.monitor_ = _FakeMonitor()
            self.transmat_ = np.eye(n_components) / n_components

        def fit(self, X, lengths=None):
            return self

        def predict(self, X):
            n = self.n_components
            return np.tile(np.arange(n), len(X) // n + 1)[: len(X)]

        def predict_proba(self, X):
            n = self.n_components
            return np.ones((len(X), n)) / n

    hl = types.ModuleType("hmmlearn")
    hl.__path__ = []
    hlh = types.ModuleType("hmmlearn.hmm")
    hlh.GaussianHMM = _FakeGaussianHMM
    sys.modules["hmmlearn"] = hl
    sys.modules["hmmlearn.hmm"] = hlh


def _install_xgboost_stub():
    """
    XGBoost stub whose feature_importances_ dynamically matches
    the number of features used in fit().
    """
    import numpy as np

    class _FakeXGBClassifier:
        def __init__(self, **kw):
            self._n_features = 10

        def fit(self, X, y):
            self._n_features = X.shape[1] if hasattr(X, "shape") else 10
            return self

        def predict(self, X):
            n = len(X)
            return np.tile([0, 1], n // 2 + 1)[:n]

        def predict_proba(self, X):
            n = len(X)
            p = np.tile([0.45, 0.55], n // 2 + 1)[:n]
            return np.column_stack([1 - p, p])

        @property
        def feature_importances_(self):
            return np.ones(self._n_features) / self._n_features

    xgb_m = types.ModuleType("xgboost")
    xgb_m.XGBClassifier = _FakeXGBClassifier
    sys.modules["xgboost"] = xgb_m


def _install_arch_stub():
    """
    arch library stub.  The production code's _check_arch() does:
        try: import arch; return True
        except ImportError: return False
    So to force the scipy fallback we make 'arch' importable but ensure
    that 'from arch import arch_model' raises ImportError (no such attr).
    We also monkey-patch GARCHVolatilityModule._check_arch directly once
    the module is loaded to be safe.
    """
    arch_m = types.ModuleType("arch")
    arch_m.__path__ = []
    # Do NOT add arch_model — importing arch succeeds but arch_model doesn't exist
    sys.modules["arch"] = arch_m
    for sub in ["arch.univariate", "arch.univariate.base"]:
        m = types.ModuleType(sub)
        m.__path__ = []
        sys.modules[sub] = m


def _install_sklearn_scaler_patch():
    """
    Patch StandardScaler so it doesn't call scipy.sparse.issparse()
    which internally tries isinstance(X, tuple_of_sparse_types) where
    the tuple may contain our _Auto stub instead of a real class.
    """
    try:
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        _orig_fit_transform = StandardScaler.fit_transform

        def _safe_fit_transform(self, X, y=None, **params):
            # Ensure X is a plain numpy array before sklearn sees it
            if not isinstance(X, np.ndarray):
                X = np.asarray(X, dtype=np.float64)
            return _orig_fit_transform(self, X, y, **params)

        StandardScaler.fit_transform = _safe_fit_transform

        _orig_transform = StandardScaler.transform

        def _safe_transform(self, X, copy=None):
            if not isinstance(X, np.ndarray):
                X = np.asarray(X, dtype=np.float64)
            return _orig_transform(self, X, copy=copy)

        StandardScaler.transform = _safe_transform

    except Exception:
        pass  # If patch fails, tests relying on HMM will skip gracefully


# Install all stubs before any project imports
_install_hmmlearn_stub()
_install_xgboost_stub()
_install_arch_stub()
for _stub in ["yfinance"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)

# NOW it's safe to import pandas and project modules
import numpy as np
import pandas as pd

# Patch sklearn AFTER pandas import (sklearn imports fine with real pyarrow logic)
_install_sklearn_scaler_patch()

# CRITICAL: patch sklearn's is_pyarrow_data — it tries isinstance() with our
# _Auto stub class as the second arg, which is invalid in Python 3.12+
try:
    import sklearn.utils._dataframe as _sdf
    _sdf.is_pyarrow_data = lambda X: False
except Exception:
    pass

# CRITICAL: force GARCH module to use scipy fallback in this environment
# by patching _check_arch after the volatility module is imported
def _patch_garch_check_arch():
    try:
        from modules.volatility_module import GARCHVolatilityModule
        GARCHVolatilityModule._check_arch = lambda self: False
    except Exception:
        pass
_patch_garch_check_arch()

# Make project root importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pytest
except ImportError:
    # Minimal pytest shim (used by run_tests.py)
    pytest = None

# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════


def make_ohlcv(
    n: int = 500,
    start: str = "2018-01-01",
    seed: int = 42,
    freq: str = "B",
    start_price: float = 300.0,
    vol: float = 0.015,
) -> pd.DataFrame:
    """Pure random-walk OHLCV DataFrame (zero drift GBM)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq=freq)
    log_r = rng.normal(0, vol, n)
    close = start_price * np.exp(np.cumsum(log_r))
    noise = rng.uniform(0.001, 0.005, n)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, noise)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, noise)))
    volume = np.abs(rng.normal(5e6, 1e6, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def make_feature_df(
    n: int = 500,
    start: str = "2018-01-01",
    seed: int = 42,
    trending: bool = False,
    all_bullish_regime: bool = False,
) -> pd.DataFrame:
    """
    Full feature-engineered DataFrame ready for signal/simulation modules.
    Includes all columns that the pipeline expects.
    """
    drift = 0.0003 if trending else 0.0
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="B")

    log_r = rng.normal(drift, 0.015, n)
    close = 300.0 * np.exp(np.cumsum(log_r))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vol = np.abs(rng.normal(5e6, 1e6, n))

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )

    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-10)

    tr = df["High"] - df["Low"]
    df["adx"] = (tr.rolling(14).mean() / (df["Close"] + 1e-10) * 1000).clip(0, 100)

    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / (df["volume_sma_20"] + 1e-10)
    df["volatility_20"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["garch_vol"] = df["volatility_20"]
    df["garch_next_vol"] = df["garch_vol"]
    df["price_above_ema50"] = (df["Close"] > df["ema_50"]).astype(int)
    df["momentum_10"] = df["Close"].pct_change(10)

    if all_bullish_regime:
        df["regime_label"] = "Bull"
        df["regime_is_bullish"] = True
        df["regime_is_bearish"] = False
        df["regime_state"] = 4
    else:
        labels = ["Bull", "Sideways", "Bear", "Sideways_Bull", "Bull"]
        block = np.repeat(labels, n // len(labels) + 1)[:n]
        df["regime_label"] = block
        df["regime_is_bullish"] = pd.Series(block).isin(
            {"Bull", "Strong_Bull", "Sideways_Bull"}
        ).values
        df["regime_is_bearish"] = pd.Series(block).isin({"Bear", "Crash"}).values
        df["regime_state"] = 2

    df["regime_prob_Bull"] = 0.6
    df["is_bull_run"] = False
    df["is_bear_run"] = False

    df["xgb_pred"] = rng.integers(0, 2, n)
    df["xgb_prob_up"] = rng.uniform(0.4, 0.6, n)
    df["xgb_confidence"] = np.abs(df["xgb_prob_up"] - 0.5) * 2

    df.dropna(inplace=True)
    return df


def make_trade_df(
    n_trades: int = 20,
    seed: int = 42,
    win_rate: float = 0.5,
    avg_win: float = 150.0,
    avg_loss: float = -100.0,
    capital_per_trade: float = 5000.0,
) -> pd.DataFrame:
    """Synthetic closed-trade DataFrame with controllable win-rate and P&L."""
    rng = np.random.default_rng(seed)
    wins = int(n_trades * win_rate)
    losses = n_trades - wins
    pnls = list(rng.normal(avg_win, 50, wins)) + list(
        rng.normal(avg_loss, 30, losses)
    )
    rng.shuffle(pnls)
    pnls = np.array(pnls)

    entry_prices = rng.uniform(100, 400, n_trades)
    exit_prices = entry_prices + pnls / 10
    dates = pd.date_range("2020-01-01", periods=n_trades * 5, freq="B")
    entry_dates = [str(dates[i * 5].date()) for i in range(n_trades)]
    exit_dates = [str(dates[i * 5 + 4].date()) for i in range(n_trades)]

    return pd.DataFrame(
        {
            "trade_id": range(1, n_trades + 1),
            "entry_date": entry_dates,
            "entry_price": entry_prices,
            "exit_date": exit_dates,
            "exit_price": exit_prices,
            "shares": capital_per_trade / entry_prices,
            "capital_used": capital_per_trade,
            "pnl": pnls,
            "returns_pct": pnls / capital_per_trade,
            "regime_at_entry": "Bull",
            "confidence": 0.6,
            "vote_count": 5,
            "is_closed": True,
            "exit_reason": "Signal",
            "cumulative_return": np.cumsum(pnls) / (capital_per_trade * n_trades),
        }
    )


def make_equity_curve(
    trade_df: pd.DataFrame,
    n_days: int = 252,
    initial_capital: float = 100_000.0,
    seed: int = 42,
) -> pd.Series:
    """Build a smooth equity curve from a trade DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    noise = rng.normal(0, 0.004, n_days)
    total_pnl = trade_df["pnl"].sum() if len(trade_df) > 0 else 0
    drift = total_pnl / (initial_capital * n_days)
    equity = initial_capital * np.exp(np.cumsum(drift + noise))
    return pd.Series(equity, index=dates, name="equity")


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES  (used when running with real pytest)
# ═══════════════════════════════════════════════════════════════════════════

if pytest is not None:

    @pytest.fixture(scope="session")
    def random_ohlcv():
        return make_ohlcv(n=500, seed=42)

    @pytest.fixture(scope="session")
    def random_feature_df():
        return make_feature_df(n=500, seed=42)

    @pytest.fixture(scope="session")
    def trending_feature_df():
        return make_feature_df(n=500, seed=7, trending=True)

    @pytest.fixture(scope="session")
    def bullish_feature_df():
        return make_feature_df(n=500, seed=99, all_bullish_regime=True)

    @pytest.fixture
    def small_trade_df():
        return make_trade_df(30, seed=42, win_rate=0.5, avg_win=150, avg_loss=-100)

    @pytest.fixture
    def winning_trade_df():
        return make_trade_df(30, seed=42, win_rate=0.70, avg_win=200, avg_loss=-80)

    @pytest.fixture
    def losing_trade_df():
        return make_trade_df(30, seed=42, win_rate=0.30, avg_win=80, avg_loss=-200)

    @pytest.fixture
    def flat_equity():
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        return pd.Series(100_000.0, index=dates, name="equity")

    @pytest.fixture
    def rising_equity():
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        return pd.Series(100_000 * (1 + 0.001) ** np.arange(252), index=dates)

    @pytest.fixture
    def crashing_equity():
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        return pd.Series(100_000 * (1 - 0.002) ** np.arange(252), index=dates)
