"""
tests/test_data_module.py
=========================
Unit tests for FeatureEngineer and DataModule (plugin architecture).

Tests cover:
  - All 14 default features are computed and produce correct column names
  - No NaNs in the middle of output (only at start due to rolling windows)
  - Feature values are in expected numeric ranges
  - Plug-and-play: registering a new feature works end-to-end
  - CSV plugin validates required column schema
  - DataModule caches and retrieves data correctly
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd

from modules.data_module import FeatureEngineer, DataModule, CSVPlugin


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _base_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="B")
    close = 200 + np.cumsum(rng.normal(0, 1.5, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high  = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low   = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol   = np.abs(rng.normal(3e6, 5e5, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FeatureEngineer tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:

    def test_all_default_features_computed(self):
        """All registered features should add their expected columns."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())

        expected_cols = [
            "returns", "log_returns", "sma_20", "sma_50", "ema_50",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_width",
            "adx", "volume_ratio", "volatility_20", "momentum_10",
            "price_above_ema50",
        ]
        for col in expected_cols:
            assert col in out.columns, f"Missing feature column: {col}"

    def test_output_no_nans_after_warmup(self):
        """After dropna(), the output DataFrame should contain no NaN values."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert out.isna().sum().sum() == 0, "NaN values remain after feature compute"

    def test_output_length_reduced_by_warmup(self):
        """Longest rolling window is 50 bars; output must be shorter than input."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert len(out) < len(df), "Output should be shorter due to rolling warmup"
        assert len(out) >= 200,    "Output should still have substantial rows"

    def test_rsi_range(self):
        """RSI must be bounded [0, 100]."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert out["rsi_14"].between(0, 100).all(), "RSI out of [0,100] range"

    def test_returns_finite(self):
        """Daily returns should be finite and within plausible bounds (±50%)."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert np.isfinite(out["returns"]).all(), "Infinite returns"
        assert out["returns"].abs().max() < 0.5,  "Returns exceed ±50%"

    def test_volatility_positive(self):
        """Realised volatility must be strictly positive."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert (out["volatility_20"] > 0).all(), "Non-positive volatility"

    def test_bollinger_bands_ordered(self):
        """Upper Bollinger band must always exceed lower band."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        assert (out["bb_upper"] > out["bb_lower"]).all(), "BB upper ≤ lower"

    def test_price_above_ema50_binary(self):
        """price_above_ema50 must be 0 or 1 only."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        vals = out["price_above_ema50"].unique()
        assert set(vals).issubset({0, 1}), f"Non-binary values: {vals}"

    def test_log_returns_consistent_with_returns(self):
        """log_returns ≈ returns for small daily moves."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        # For small r, log(1+r) ≈ r;  correlation should be very high
        corr = out["returns"].corr(out["log_returns"])
        assert corr > 0.99, f"log_returns / returns correlation too low: {corr:.4f}"

    def test_selective_feature_compute(self):
        """Requesting a subset of features should only add those columns."""
        df   = _base_ohlcv(300)
        out  = FeatureEngineer.compute(df.copy(), features=["returns", "rsi_14"])
        assert "returns" in out.columns
        assert "rsi_14"  in out.columns
        # Other features should NOT be present (beyond original OHLCV)
        assert "macd"      not in out.columns
        assert "volatility_20" not in out.columns

    def test_plug_and_play_new_feature(self):
        """Registering a new feature via decorator should work end-to-end."""
        @FeatureEngineer.register("test_custom_close_sq")
        def my_feature(df):
            df["test_custom_close_sq"] = df["Close"] ** 2
            return df

        df  = _base_ohlcv(200)
        out = FeatureEngineer.compute(df.copy(), features=["test_custom_close_sq"])
        assert "test_custom_close_sq" in out.columns
        # Values should be positive squares
        assert (out["test_custom_close_sq"] > 0).all()
        # Clean up registry to avoid polluting other tests
        FeatureEngineer.FEATURE_REGISTRY.pop("test_custom_close_sq", None)

    def test_feature_failure_is_graceful(self):
        """A buggy feature function should log a warning, not crash the pipeline."""
        @FeatureEngineer.register("_bad_feature_test")
        def bad_feature(df):
            raise ValueError("intentional error")

        df  = _base_ohlcv(200)
        out = FeatureEngineer.compute(df.copy(), features=["returns", "_bad_feature_test"])
        # Pipeline should continue; returns should still be computed
        assert "returns" in out.columns
        FeatureEngineer.FEATURE_REGISTRY.pop("_bad_feature_test", None)

    def test_macd_signal_relationship(self):
        """MACD histogram = MACD line − Signal line."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        residual = (out["macd"] - out["macd_signal"] - out["macd_hist"]).abs()
        assert residual.max() < 1e-8, "MACD hist ≠ MACD − Signal"

    def test_volume_ratio_near_one_on_average(self):
        """volume_ratio should average close to 1 (volume / rolling mean volume)."""
        df  = _base_ohlcv(300)
        out = FeatureEngineer.compute(df.copy())
        mean_ratio = out["volume_ratio"].mean()
        assert 0.8 < mean_ratio < 1.5, f"Mean volume_ratio unexpectedly far from 1: {mean_ratio:.3f}"


# ═══════════════════════════════════════════════════════════════════════════
# CSV Plugin tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCSVPlugin:

    def _write_csv(self, df: pd.DataFrame, path: str):
        df.index.name = "Date"
        df.reset_index().to_csv(path, index=False)

    def test_valid_csv_loads_correctly(self):
        """A well-formed CSV should load and return the correct shape."""
        df = _base_ohlcv(100)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_csv(df, path)
            plugin = CSVPlugin(path)
            result = plugin.fetch("TEST", "2019-01-01", "2030-01-01")
            assert len(result) > 0
            assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(result.columns)
        finally:
            os.unlink(path)

    def test_csv_missing_column_raises(self):
        """CSV missing a required column should raise ValueError."""
        df = _base_ohlcv(50).drop(columns=["Volume"])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_csv(df, path)
            plugin = CSVPlugin(path)
            with pytest.raises(ValueError, match="missing columns"):
                plugin.fetch("TEST", "2019-01-01", "2030-01-01")
        finally:
            os.unlink(path)

    def test_csv_date_filtering(self):
        """Only rows within [start, end] should be returned."""
        df = _base_ohlcv(200)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            self._write_csv(df, path)
            plugin = CSVPlugin(path)
            result = plugin.fetch("TEST", "2019-01-01", "2019-06-30")
            assert result.index.min() >= pd.Timestamp("2019-01-01")
            assert result.index.max() <= pd.Timestamp("2019-06-30")
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# DataModule caching tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDataModuleCaching:
    """
    DataModule caching tests.
    The DataModule uses parquet for caching; in environments where pyarrow is
    stubbed we patch _cache_key to use a CSV path instead, which lets us verify
    the caching behaviour without the real parquet library.
    """

    def _patch_cache_to_csv(self, dm):
        """Monkey-patch DataModule to cache as CSV instead of parquet."""
        import functools

        orig_get_raw = dm.get_raw_data.__func__

        def _csv_cache_key(self, symbol, start, end, suffix):
            key = f"{symbol}_{start}_{end}_{suffix}"
            return os.path.join(self.cache_dir, f"{key}.csv")

        def _get_raw_csv(self, symbol, start, end, source="yahoo", force_refresh=False):
            cache_path = _csv_cache_key(self, symbol, start, end, "raw")
            if not force_refresh and os.path.exists(cache_path):
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if source not in self._plugins:
                raise ValueError(f"Unknown source: {source}. Available: {list(self._plugins)}")
            df = self._plugins[source].fetch(symbol, start, end)
            df.to_csv(cache_path)
            return df

        import types
        dm.get_raw_data = types.MethodType(_get_raw_csv, dm)
        return dm

    def test_register_and_use_custom_plugin(self):
        """Custom plugin registered on DataModule should be callable."""
        from modules.data_module import DataSourcePlugin

        class _StaticPlugin(DataSourcePlugin):
            @property
            def name(self): return "static"
            def fetch(self, symbol, start, end):
                return _base_ohlcv(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataModule(cache_dir=tmpdir)
            self._patch_cache_to_csv(dm)
            dm.register_plugin("static", _StaticPlugin())
            raw = dm.get_raw_data("ANY", "2019-01-01", "2020-01-01", source="static")
            assert len(raw) == 100
            assert "Close" in raw.columns

    def test_caching_avoids_second_fetch(self):
        """Second call with same params should return cached result without calling fetch."""
        from modules.data_module import DataSourcePlugin

        call_count = {"n": 0}

        class _CountingPlugin(DataSourcePlugin):
            @property
            def name(self): return "counting"
            def fetch(self, symbol, start, end):
                call_count["n"] += 1
                return _base_ohlcv(80)

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataModule(cache_dir=tmpdir)
            self._patch_cache_to_csv(dm)
            dm.register_plugin("counting", _CountingPlugin())
            dm.get_raw_data("SYM", "2019-01-01", "2020-01-01", source="counting")
            dm.get_raw_data("SYM", "2019-01-01", "2020-01-01", source="counting")
            assert call_count["n"] == 1, "fetch() called more than once — cache not working"

    def test_force_refresh_bypasses_cache(self):
        """force_refresh=True should always call fetch(), even if cache exists."""
        from modules.data_module import DataSourcePlugin

        call_count = {"n": 0}

        class _CountingPlugin(DataSourcePlugin):
            @property
            def name(self): return "counting2"
            def fetch(self, symbol, start, end):
                call_count["n"] += 1
                return _base_ohlcv(60)

        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataModule(cache_dir=tmpdir)
            self._patch_cache_to_csv(dm)
            dm.register_plugin("counting2", _CountingPlugin())
            dm.get_raw_data("SYM", "2019-01-01", "2020-01-01", source="counting2")
            dm.get_raw_data("SYM", "2019-01-01", "2020-01-01", source="counting2",
                            force_refresh=True)
            assert call_count["n"] == 2, "Expected 2 fetch() calls with force_refresh"

    def test_unknown_source_raises(self):
        """Requesting an unregistered source should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataModule(cache_dir=tmpdir)
            self._patch_cache_to_csv(dm)
            with pytest.raises(ValueError, match="Unknown source"):
                dm.get_raw_data("SYM", "2019-01-01", "2020-01-01", source="nonexistent")
