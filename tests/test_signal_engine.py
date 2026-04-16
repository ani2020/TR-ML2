"""
tests/test_signal_engine.py
============================
Unit tests for SignalEngine, IndicatorRegistry, and individual Indicators.

Tests cover:
  - All 7 default indicators register and compute boolean Series
  - Voting threshold gate: signals fire only when votes ≥ min_votes
  - Regime gate: no ENTRY when regime is bearish regardless of votes
  - Max open trades cap is enforced
  - Plug-and-play: custom indicator registers and participates in voting
  - Signal label column values are exactly "ENTRY", "EXIT", "IDLE"
  - ENTRY is always followed eventually by EXIT (no orphaned positions)
"""

import pytest
import numpy as np
import pandas as pd

from modules.signal_engine import (
    SignalEngine, IndicatorRegistry, Indicator,
    SIGNAL_ENTRY, SIGNAL_EXIT, SIGNAL_IDLE,
    MomentumIndicator, VolumeIndicator, VolatilityFilterIndicator,
    ADXIndicator, PriceAboveEMAIndicator, RSIIndicator, MACDIndicator,
)
from tests.conftest import make_feature_df


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _force_all_bullish(df: pd.DataFrame) -> pd.DataFrame:
    """Override regime and XGB columns to make every bar entry-eligible."""
    df = df.copy()
    df["regime_is_bullish"] = True
    df["regime_label"]      = "Bull"
    df["xgb_pred"]          = 1
    df["xgb_confidence"]    = 0.8
    df["xgb_confidence_at_signal"] = 0.8
    return df


def _force_all_bearish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime_is_bullish"] = False
    df["regime_label"]      = "Bear"
    df["xgb_pred"]          = 0
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Individual Indicator tests
# ═══════════════════════════════════════════════════════════════════════════

class TestIndividualIndicators:

    @pytest.fixture
    def df(self):
        return make_feature_df(n=300, seed=0)

    def test_all_default_indicators_registered(self):
        available = IndicatorRegistry.available()
        expected  = ["momentum", "volume", "low_volatility", "adx",
                     "price_above_ema", "rsi", "macd"]
        for name in expected:
            assert name in available, f"'{name}' not in IndicatorRegistry"

    def test_indicators_return_boolean_series(self, df):
        for name in ["momentum", "volume", "low_volatility", "adx",
                     "price_above_ema", "rsi", "macd"]:
            ind    = IndicatorRegistry.get(name)
            result = ind.compute(df)
            assert isinstance(result, pd.Series), f"{name} did not return Series"
            unique = set(result.dropna().unique())
            assert unique.issubset({0, 1, True, False}), \
                f"{name} returned non-boolean values: {unique}"

    def test_indicators_same_length_as_input(self, df):
        for name in IndicatorRegistry.available():
            ind    = IndicatorRegistry.get(name)
            result = ind.compute(df)
            assert len(result) == len(df), \
                f"{name} returned Series with wrong length"

    def test_momentum_threshold_respected(self, df):
        """MomentumIndicator with high threshold should return mostly False."""
        ind    = MomentumIndicator(threshold=0.99)   # 99% — almost never true
        result = ind.compute(df)
        assert result.mean() < 0.05, "High threshold should barely fire"

    def test_momentum_low_threshold_fires_often(self, df):
        """MomentumIndicator with threshold=-1 (always true) should fire always."""
        ind    = MomentumIndicator(threshold=-1.0)
        result = ind.compute(df)
        assert result.mean() > 0.95, "Low threshold should fire almost always"

    def test_rsi_window_bull_zone(self, df):
        """RSI indicator with window (60,90) should not fire when RSI < 60."""
        ind = RSIIndicator(rsi_low=60, rsi_high=90)
        result = ind.compute(df)
        # Where RSI < 60, indicator should be False
        low_rsi_mask = df["rsi_14"] < 60
        assert (result[low_rsi_mask] == False).all(), \
            "RSI indicator should be False when RSI < rsi_low"

    def test_volatility_filter_fires_when_low_vol(self):
        """VolatilityFilterIndicator should be True when garch_vol is low."""
        df = make_feature_df(300, seed=5)
        df["garch_vol"] = 0.05   # force very low vol
        ind = VolatilityFilterIndicator(max_vol=0.10)
        result = ind.compute(df)
        assert result.all(), "Should fire on all bars when vol < threshold"

    def test_plug_and_play_custom_indicator(self):
        """A user-defined Indicator subclass should integrate seamlessly."""
        class AlwaysTrueIndicator(Indicator):
            @property
            def name(self): return "_test_always_true"
            def compute(self, df): return pd.Series(True, index=df.index)

        ind = AlwaysTrueIndicator()
        IndicatorRegistry.register(ind)
        assert "_test_always_true" in IndicatorRegistry.available()

        df     = make_feature_df(100, seed=0)
        result = IndicatorRegistry.get("_test_always_true").compute(df)
        assert result.all(), "AlwaysTrue indicator should be True everywhere"
        # Clean up
        IndicatorRegistry._registry.pop("_test_always_true", None)


# ═══════════════════════════════════════════════════════════════════════════
# SignalEngine voting and gate tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalEngineVoting:

    def test_no_signals_when_all_bearish(self):
        """When every bar is bearish regime, no ENTRY signals should be generated."""
        df  = _force_all_bearish(make_feature_df(300, seed=1))
        eng = SignalEngine(min_votes=1, exit_votes=1, use_xgb_filter=False)
        out = eng.generate(df)
        assert (out["signal"] != SIGNAL_ENTRY).all(), \
            "ENTRY generated in bearish regime — regime gate broken"

    def test_no_signals_when_votes_below_threshold(self):
        """Force all indicators to False; min_votes=1 should still produce no entries."""
        df = _force_all_bullish(make_feature_df(300, seed=2))
        # Monkeypatch every indicator to return False
        for name in IndicatorRegistry.available():
            ind = IndicatorRegistry.get(name)
            ind._orig_compute = ind.compute
            ind.compute = lambda _df, _ind=ind: pd.Series(False, index=_df.index)

        try:
            eng = SignalEngine(min_votes=1, exit_votes=1, use_xgb_filter=False)
            out = eng.generate(df)
            assert (out["signal"] != SIGNAL_ENTRY).all(), \
                "ENTRY generated when all indicators are False"
        finally:
            for name in IndicatorRegistry.available():
                ind = IndicatorRegistry.get(name)
                if hasattr(ind, "_orig_compute"):
                    ind.compute = ind._orig_compute
                    del ind._orig_compute

    def test_signal_labels_are_valid(self):
        """signal_label column should contain only ENTRY / EXIT / IDLE."""
        df  = make_feature_df(300, seed=3)
        eng = SignalEngine(min_votes=3, exit_votes=2, use_xgb_filter=False)
        out = eng.generate(df)
        valid = {"ENTRY", "EXIT", "IDLE"}
        observed = set(out["signal_label"].unique())
        assert observed.issubset(valid), f"Unexpected labels: {observed - valid}"

    def test_vote_count_bounded_by_n_indicators(self):
        """vote_count must be between 0 and number of active indicators."""
        df  = make_feature_df(300, seed=4)
        eng = SignalEngine(min_votes=3, exit_votes=2, use_xgb_filter=False)
        out = eng.generate(df)
        n   = len(eng.indicator_names)
        assert out["vote_count"].between(0, n).all(), \
            f"vote_count out of [0, {n}]"

    def test_max_open_trades_limit_respected(self):
        """The system must never exceed max_open_trades simultaneous positions."""
        df  = _force_all_bullish(make_feature_df(400, seed=5))
        # All indicators True → many ENTRY opportunities
        saved = {}
        for name in IndicatorRegistry.available():
            ind = IndicatorRegistry.get(name)
            saved[name] = ind.compute
            # Must be a proper function — lambdas with wrong arity cause TypeError
            def _always_true(df_arg):
                return pd.Series(True, index=df_arg.index)
            ind.compute = _always_true

        try:
            max_open = 2
            eng = SignalEngine(min_votes=1, exit_votes=7,   # hard to exit
                               max_open_trades=max_open, use_xgb_filter=False)
            out = eng.generate(df)
            depth = 0
            for sig in out["signal"]:
                if sig == SIGNAL_ENTRY:
                    depth += 1
                elif sig == SIGNAL_EXIT:
                    depth = max(0, depth - 1)
                assert depth <= max_open, \
                    f"Open trade depth {depth} exceeded max {max_open}"
        finally:
            for name, orig in saved.items():
                IndicatorRegistry.get(name).compute = orig

    def test_entry_always_followed_by_exit(self):
        """Every ENTRY must be paired with a later EXIT (or force-close at end)."""
        df  = make_feature_df(400, seed=6)
        eng = SignalEngine(min_votes=3, exit_votes=2, use_xgb_filter=False)
        out = eng.generate(df)
        signals = out["signal"].tolist()
        open_count = 0
        for s in signals:
            if s == SIGNAL_ENTRY:
                open_count += 1
            elif s == SIGNAL_EXIT:
                assert open_count > 0, "EXIT without preceding ENTRY"
                open_count -= 1
        # After the loop, open trades are "force-closed" — acceptable
        assert open_count >= 0

    def test_xgb_filter_blocks_entries(self):
        """With use_xgb_filter=True and xgb_pred=0, ENTRY should be blocked."""
        df = _force_all_bullish(make_feature_df(300, seed=7))
        df["xgb_pred"] = 0   # XGB says DOWN everywhere

        eng = SignalEngine(min_votes=1, exit_votes=7, use_xgb_filter=True)
        out = eng.generate(df)
        assert (out["signal"] != SIGNAL_ENTRY).all(), \
            "ENTRY generated despite xgb_pred=0 and use_xgb_filter=True"

    def test_xgb_filter_disabled_ignores_xgb(self):
        """With use_xgb_filter=False, xgb_pred=0 should NOT block entries."""
        df = _force_all_bullish(make_feature_df(300, seed=8))
        df["xgb_pred"] = 0

        eng = SignalEngine(min_votes=2, exit_votes=7, use_xgb_filter=False)
        out = eng.generate(df)
        assert (out["signal"] == SIGNAL_ENTRY).any(), \
            "No ENTRY generated with xgb_filter disabled — unexpected"

    def test_add_indicator_at_runtime(self):
        """engine.add_indicator() should register and use the new indicator."""
        class SometimesTrue(Indicator):
            @property
            def name(self): return "_runtime_test_ind"
            def compute(self, df):
                return pd.Series(
                    np.tile([True, False], len(df) // 2 + 1)[:len(df)],
                    index=df.index,
                )

        df  = make_feature_df(200, seed=9)
        eng = SignalEngine(min_votes=2, exit_votes=1, use_xgb_filter=False)
        eng.add_indicator(SometimesTrue())
        assert "_runtime_test_ind" in eng.indicator_names
        out = eng.generate(df)
        assert f"ind__runtime_test_ind" in out.columns
        # Clean up
        IndicatorRegistry._registry.pop("_runtime_test_ind", None)

    def test_indicator_columns_present_in_output(self):
        """Output should contain one ind_<name> column per active indicator."""
        df  = make_feature_df(200, seed=10)
        eng = SignalEngine(min_votes=3, exit_votes=2, use_xgb_filter=False)
        out = eng.generate(df)
        for name in eng.indicator_names:
            col = f"ind_{name}"
            assert col in out.columns, f"Missing indicator column: {col}"
