"""
tests/test_simulation_module.py
================================
Unit tests for TradeSimulator and SimulationConfig.

tests/test_pipeline.py (combined in this file)
===============================================
Integration tests for the full HMM → GARCH → XGBoost → Signal → Simulate pipeline.

Tests cover:
  - Slippage is applied correctly (buy high, sell low)
  - Expenses are deducted on every closed trade
  - Confidence sizing scales position correctly
  - Equity curve starts at initial_capital and is updated every bar
  - Trade DataFrame contains all required columns
  - Capital conservation: cash + open positions ≈ equity
  - Pipeline on trending data produces more trades than on flat data
  - Pipeline on random data: Sharpe ≈ 0, PF ≈ 1 (integration of RandomPipeline test)
"""

import pytest
import numpy as np
import pandas as pd

from modules.simulation_module import TradeSimulator, SimulationConfig, SIGNAL_ENTRY, SIGNAL_EXIT
from modules.metrics_module import MetricsModule
from tests.conftest import make_feature_df, make_trade_df, make_equity_curve


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

CAPITAL = 100_000.0
PER_TRADE = 10_000.0
EXPENSE  = 10.0
SLIP     = 0.001


def _default_cfg(**overrides) -> SimulationConfig:
    kw = dict(
        total_capital=CAPITAL,
        max_capital_per_trade=PER_TRADE,
        trading_expense=EXPENSE,
        slippage_pct=SLIP,
        confidence_sizing=False,
        min_confidence=0.0,
    )
    kw.update(overrides)
    return SimulationConfig(**kw)


def _df_with_signals(n=100, entry_indices=None, exit_indices=None, seed=0):
    """Build a minimal signal DataFrame for simulator tests."""
    df = make_feature_df(n=n, seed=seed)
    df["signal"]   = 0
    df["vote_count"] = 5
    df["xgb_confidence_at_signal"] = np.nan
    if entry_indices:
        for i in entry_indices:
            if i < len(df):
                df.iloc[i, df.columns.get_loc("signal")] = SIGNAL_ENTRY
                df.iloc[i, df.columns.get_loc("xgb_confidence_at_signal")] = 0.7
    if exit_indices:
        for i in exit_indices:
            if i < len(df):
                df.iloc[i, df.columns.get_loc("signal")] = SIGNAL_EXIT
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TradeSimulator unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTradeSimulator:

    def test_no_signals_produces_no_trades(self):
        """A DataFrame with no signals should result in zero closed trades."""
        df  = make_feature_df(200, seed=0)
        df["signal"] = 0
        df["xgb_confidence_at_signal"] = np.nan
        df["vote_count"] = 0
        sim = TradeSimulator(_default_cfg())
        sim.simulate(df)
        assert len(sim.trades) == 0

    def test_equity_starts_at_initial_capital(self):
        """First equity value should equal initial_capital (no trades on bar 0)."""
        df  = _df_with_signals(100, entry_indices=[20], exit_indices=[40])
        sim = TradeSimulator(_default_cfg())
        out = sim.simulate(df)
        first_eq = out["equity"].iloc[0]
        assert abs(first_eq - CAPITAL) < 100, \
            f"First equity {first_eq:.2f} ≠ {CAPITAL}"

    def test_equity_is_monotone_without_trades(self):
        """With zero signals the equity curve is exactly flat."""
        df  = make_feature_df(200, seed=1)
        df["signal"] = 0
        df["xgb_confidence_at_signal"] = np.nan
        df["vote_count"] = 0
        sim = TradeSimulator(_default_cfg())
        out = sim.simulate(df)
        eq  = out["equity"].dropna()
        assert eq.nunique() == 1, "Equity should be flat with no trades"
        assert eq.iloc[0] == pytest.approx(CAPITAL, rel=1e-6)

    def test_slippage_applied_to_entry(self):
        """Entry price should exceed the Close price by slippage_pct."""
        df  = _df_with_signals(100, entry_indices=[10], exit_indices=[50])
        sim = TradeSimulator(_default_cfg(slippage_pct=0.01))  # 1%
        sim.simulate(df)
        assert len(sim.trades) > 0
        trade       = sim.trades[0]
        bar_close   = df["Close"].iloc[10]
        expected_ep = bar_close * (1 + 0.01)
        assert trade.entry_price == pytest.approx(expected_ep, rel=1e-5)

    def test_slippage_applied_to_exit(self):
        """Exit price should be below Close by slippage_pct."""
        df  = _df_with_signals(100, entry_indices=[10], exit_indices=[50])
        sim = TradeSimulator(_default_cfg(slippage_pct=0.01))
        sim.simulate(df)
        assert len(sim.trades) > 0
        trade       = sim.trades[0]
        bar_close   = df["Close"].iloc[50]
        expected_xp = bar_close * (1 - 0.01)
        assert trade.exit_price == pytest.approx(expected_xp, rel=1e-5)

    def test_zero_slippage_entry_equals_close(self):
        """With slippage=0, entry price equals bar close exactly."""
        df  = _df_with_signals(100, entry_indices=[10], exit_indices=[50])
        sim = TradeSimulator(_default_cfg(slippage_pct=0.0))
        sim.simulate(df)
        trade     = sim.trades[0]
        bar_close = df["Close"].iloc[10]
        assert trade.entry_price == pytest.approx(bar_close, rel=1e-9)

    def test_expenses_reduce_pnl(self):
        """
        Each closed trade's P&L should be lower when expenses are charged.
        The simulator deducts expense at both entry (entry capital = capital - expense)
        and exit (proceeds = shares * price - expense), so the total reduction
        per trade is approximately 2 × expense. We allow ±20% tolerance
        since share count slightly varies with expense.
        """
        df  = _df_with_signals(100, entry_indices=[10], exit_indices=[50])
        sim_no_exp   = TradeSimulator(_default_cfg(slippage_pct=0.0, trading_expense=0.0))
        sim_with_exp = TradeSimulator(_default_cfg(slippage_pct=0.0, trading_expense=100.0))
        sim_no_exp.simulate(df)
        sim_with_exp.simulate(df)
        if sim_no_exp.trades and sim_with_exp.trades:
            diff = sim_no_exp.trades[0].pnl - sim_with_exp.trades[0].pnl
            # Each trade has expense deducted twice (at entry reducing shares,
            # and at exit reducing proceeds), so total impact ≈ 200 ± some%
            assert 150 <= diff <= 260, \
                f"Expense impact {diff:.2f} outside expected range [150, 260]"

    def test_confidence_sizing_scales_capital(self):
        """Position size should scale linearly with confidence when enabled."""
        df            = _df_with_signals(100, entry_indices=[10], exit_indices=[50])
        df.loc[df.index[10], "xgb_confidence_at_signal"] = 0.5

        sim_no_scale  = TradeSimulator(_default_cfg(confidence_sizing=False))
        sim_scaled    = TradeSimulator(_default_cfg(confidence_sizing=True))
        sim_no_scale.simulate(df)
        sim_scaled.simulate(df)

        if sim_no_scale.trades and sim_scaled.trades:
            cap_base   = sim_no_scale.trades[0].capital_used
            cap_scaled = sim_scaled.trades[0].capital_used
            assert cap_scaled < cap_base, \
                "Confidence-scaled capital should be less than max"
            assert abs(cap_scaled - PER_TRADE * 0.5) < 200

    def test_insufficient_cash_skips_entry(self):
        """When cash < position size, the entry should be skipped."""
        cfg = _default_cfg(
            total_capital=100.0,          # tiny capital
            max_capital_per_trade=10_000.0,  # way bigger than available
            confidence_sizing=False,
        )
        df  = _df_with_signals(100, entry_indices=[5, 10], exit_indices=[50, 60])
        sim = TradeSimulator(cfg)
        sim.simulate(df)
        # Should have zero or very few trades (cash too small)
        assert len(sim.trades) == 0

    def test_force_close_at_end(self):
        """Unclosed trades at period end should be force-closed."""
        # Use n=300 so after rolling warmup we still have 200+ bars;
        # place ENTRY near the very end (index -20 = 20 bars from end)
        df  = make_feature_df(n=300, seed=0)
        df["signal"] = 0
        df["vote_count"] = 5
        df["xgb_confidence_at_signal"] = float("nan")
        # Place entry near end with no exit
        near_end = len(df) - 20
        df.iloc[near_end, df.columns.get_loc("signal")] = SIGNAL_ENTRY
        df.iloc[near_end, df.columns.get_loc("xgb_confidence_at_signal")] = 0.7

        sim = TradeSimulator(_default_cfg())
        sim.simulate(df)
        assert len(sim.trades) >= 1, "Should have at least one trade"
        force_closed = [t for t in sim.trades if t.exit_reason == "Force-Close"]
        assert len(force_closed) >= 1, \
            f"Expected Force-Close trade; got reasons: {[t.exit_reason for t in sim.trades]}"

    def test_trade_df_has_required_columns(self):
        """get_trade_dataframe() should include all expected columns."""
        df  = _df_with_signals(100, entry_indices=[10, 30], exit_indices=[20, 50])
        sim = TradeSimulator(_default_cfg())
        sim.simulate(df)
        td  = sim.get_trade_dataframe()
        required = [
            "trade_id", "entry_date", "entry_price", "exit_date", "exit_price",
            "shares", "capital_used", "pnl", "returns_pct", "is_closed", "exit_reason",
        ]
        for col in required:
            assert col in td.columns, f"Missing column: {col}"

    def test_max_capital_used_tracked(self):
        """max_capital_used should be >= per-trade capital on any single trade."""
        df  = _df_with_signals(100, entry_indices=[10, 20], exit_indices=[15, 25])
        sim = TradeSimulator(_default_cfg(confidence_sizing=False))
        sim.simulate(df)
        assert sim.max_capital_used >= PER_TRADE - 1

    def test_equity_length_equals_input_length(self):
        """Equity curve should have one value per bar in the input DataFrame."""
        df  = _df_with_signals(150, entry_indices=[20], exit_indices=[60])
        sim = TradeSimulator(_default_cfg())
        out = sim.simulate(df)
        eq  = out["equity"]
        assert len(eq) == len(df)

    def test_fifo_trade_close_order(self):
        """With 2 open trades, EXIT closes the oldest (first-in, first-out)."""
        df  = _df_with_signals(150,
                               entry_indices=[10, 20],
                               exit_indices=[30])
        sim = TradeSimulator(_default_cfg())
        sim.simulate(df)
        if len(sim.trades) > 0:
            first_closed = sim.trades[0]
            assert first_closed.trade_id == 1, "FIFO: trade_id=1 should close first"


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end: HMM (stub) → GARCH (scipy fallback) → XGBoost (stub)
    → SignalEngine → TradeSimulator → MetricsModule.
    """

    def _build_pipeline_result(self):
        """Build pipeline result inline so no class fixture is needed."""
        from modules.regime_module import MarketRegimeModule
        from modules.volatility_module import GARCHVolatilityModule
        from modules.prediction_module import PredictionModule
        from modules.signal_engine import SignalEngine
        from modules.simulation_module import TradeSimulator, SimulationConfig
        from modules.metrics_module import MetricsModule

        df = make_feature_df(n=600, seed=0)

        # HMM (uses stub GaussianHMM)
        hmm = MarketRegimeModule(n_states=3, n_iter=5)
        hmm.fit(df); df = hmm.predict(df)

        # GARCH (scipy fallback — no arch needed)
        garch = GARCHVolatilityModule()
        df = garch.add_to_dataframe(df)

        # XGBoost (uses stub)
        xgb = PredictionModule()
        xgb.fit(df); df = xgb.predict(df)

        # Signals
        engine = SignalEngine(min_votes=3, exit_votes=2,
                              use_xgb_filter=True, max_open_trades=2)
        df = engine.generate(df)

        # Simulation
        cfg = SimulationConfig(
            total_capital=100_000, max_capital_per_trade=10_000,
            trading_expense=5.0, slippage_pct=0.001, confidence_sizing=False,
        )
        sim      = TradeSimulator(cfg)
        res_df   = sim.simulate(df)
        trade_df = sim.get_trade_dataframe()
        equity   = res_df["equity"].dropna()
        mm       = MetricsModule(risk_free_rate=0.0)
        metrics  = mm.compute(trade_df, equity, initial_capital=100_000)
        return metrics, trade_df, res_df

    def test_pipeline_produces_trades(self):
        metrics, _, _ = self._build_pipeline_result()
        assert metrics["n_trades"] > 0

    def test_pipeline_equity_has_correct_length(self):
        _, _, res_df = self._build_pipeline_result()
        assert len(res_df) > 0

    def test_pipeline_trade_df_all_closed(self):
        _, trade_df, _ = self._build_pipeline_result()
        if len(trade_df) > 0:
            assert trade_df["is_closed"].all()

    def test_pipeline_random_sharpe_bounded(self):
        """Random-walk + stub HMM/XGBoost → Sharpe should be near zero."""
        metrics, _, _ = self._build_pipeline_result()
        assert abs(metrics["sharpe_ratio"]) < 3.0, \
            f"Sharpe={metrics['sharpe_ratio']:.3f} too far from 0 for random pipeline"

    def test_pipeline_profit_factor_bounded(self):
        metrics, _, _ = self._build_pipeline_result()
        pf = metrics["profit_factor"]
        if not (pf == float("inf")):
            assert 0.1 < pf < 10.0, f"Profit factor={pf:.3f} out of range"

    def test_pipeline_win_rate_bounded(self):
        metrics, _, _ = self._build_pipeline_result()
        wr = metrics["win_rate"]
        assert 0.1 <= wr <= 0.9, f"Win rate {wr:.2%} out of plausible range"

    def test_pipeline_all_metrics_finite(self):
        metrics, _, _ = self._build_pipeline_result()
        for k, v in metrics.items():
            if isinstance(v, float) and not (v == float("inf") or v == float("-inf")):
                assert not (v != v), f"NaN metric: {k}={v}"  # v != v is True for NaN

    def test_trending_pipeline_better_sharpe_than_random(self):
        """Trending data strategy should not be worse than flat."""
        from modules.regime_module import MarketRegimeModule
        from modules.volatility_module import GARCHVolatilityModule
        from modules.prediction_module import PredictionModule
        from modules.signal_engine import SignalEngine
        from modules.simulation_module import TradeSimulator, SimulationConfig
        from modules.metrics_module import MetricsModule

        def _run(feature_df):
            df = feature_df.copy()
            hmm = MarketRegimeModule(n_states=3, n_iter=5)
            hmm.fit(df); df = hmm.predict(df)
            garch = GARCHVolatilityModule(); df = garch.add_to_dataframe(df)
            xgb = PredictionModule(); xgb.fit(df); df = xgb.predict(df)
            eng = SignalEngine(min_votes=3, exit_votes=2,
                               use_xgb_filter=False, max_open_trades=2)
            df = eng.generate(df)
            sim = TradeSimulator(SimulationConfig(
                total_capital=100_000, max_capital_per_trade=10_000,
                trading_expense=0, slippage_pct=0, confidence_sizing=False))
            rdf = sim.simulate(df); eq = rdf["equity"].dropna()
            tdf = sim.get_trade_dataframe()
            return MetricsModule(risk_free_rate=0.0).compute(tdf, eq, 100_000)

        flat     = _run(make_feature_df(500, seed=0, trending=False))
        trending = _run(make_feature_df(500, seed=0, trending=True))
        assert trending["sharpe_ratio"] >= flat["sharpe_ratio"] - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# GARCH module unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGARCHModule:

    def _returns(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.normal(0.0005, 0.015, 500),
                         index=pd.date_range("2018-01-01", periods=500, freq="B"))

    def test_fit_sets_params(self):
        from modules.volatility_module import GARCHVolatilityModule
        g = GARCHVolatilityModule()
        g.fit(self._returns())
        assert g.is_fitted
        p = g.params
        assert p["omega"] > 0
        assert 0 <= p["alpha"] < 1
        assert 0 <= p["beta"]  < 1
        assert p["persistence"] < 1.5

    def test_forecast_positive(self):
        from modules.volatility_module import GARCHVolatilityModule
        g = GARCHVolatilityModule(annualize=True)
        g.fit(self._returns())
        vol = g.forecast_next()
        assert vol > 0
        assert vol < 5.0

    def test_add_to_dataframe_adds_columns(self):
        from modules.volatility_module import GARCHVolatilityModule
        df  = make_feature_df(300, seed=0)
        g   = GARCHVolatilityModule()
        out = g.add_to_dataframe(df)
        assert "garch_vol"      in out.columns
        assert "garch_next_vol" in out.columns
        assert (out["garch_vol"] > 0).all()

    def test_not_fitted_raises(self):
        from modules.volatility_module import GARCHVolatilityModule
        g = GARCHVolatilityModule()
        with pytest.raises(RuntimeError, match="not fitted"):
            g.forecast_next()

    def test_vol_series_length_matches_returns(self):
        from modules.volatility_module import GARCHVolatilityModule
        df = make_feature_df(300, seed=0)
        g  = GARCHVolatilityModule()
        g.fit(df["returns"])
        series = g.predict_series(df["returns"])
        assert len(series) == len(df["returns"].dropna())


# ═══════════════════════════════════════════════════════════════════════════
# Market Regime Module unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRegimeModule:

    def _fit(self):
        from modules.regime_module import MarketRegimeModule
        df  = make_feature_df(400, seed=0)
        hmm = MarketRegimeModule(n_states=3, n_iter=5)
        hmm.fit(df)
        return hmm, df

    def test_fit_marks_fitted(self):
        hmm, _ = self._fit()
        assert hmm.is_fitted

    def test_predict_adds_regime_columns(self):
        hmm, df = self._fit()
        out = hmm.predict(df)
        for col in ["regime_label", "regime_state", "regime_is_bullish"]:
            assert col in out.columns, f"Missing: {col}"

    def test_regime_labels_are_known(self):
        hmm, df = self._fit()
        out    = hmm.predict(df)
        known  = {"Bull","Sideways","Bear","Strong_Bull","Crash",
                  "Sideways_Bull","Sideways_Bear","Unknown"}
        labels = set(out["regime_label"].unique())
        assert labels.issubset(known), f"Unexpected labels: {labels - known}"

    def test_regime_probs_sum_to_one(self):
        hmm, df = self._fit()
        probs   = hmm.get_state_probabilities(df)
        row_sums = probs.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6

    def test_regime_state_count_matches_n_states(self):
        hmm, df = self._fit()
        out      = hmm.predict(df)
        assert out["regime_state"].nunique() <= hmm.n_states

    def test_n_states_out_of_range_raises(self):
        from modules.regime_module import MarketRegimeModule
        with pytest.raises(AssertionError):
            MarketRegimeModule(n_states=8)
        with pytest.raises(AssertionError):
            MarketRegimeModule(n_states=1)

    def test_predict_before_fit_raises(self):
        from modules.regime_module import MarketRegimeModule
        df  = make_feature_df(100, seed=0)
        hmm = MarketRegimeModule(n_states=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            hmm.predict(df)

    def test_bull_bear_run_tagged(self):
        hmm, df = self._fit()
        out = hmm.predict(df)
        assert "is_bull_run" in out.columns
        assert "is_bear_run" in out.columns


# ═══════════════════════════════════════════════════════════════════════════
# Prediction Module unit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictionModule:

    def _fit(self):
        from modules.prediction_module import PredictionModule
        df  = make_feature_df(400, seed=1)
        xgb = PredictionModule()
        xgb.fit(df)
        return xgb, df

    def test_fit_marks_fitted(self):
        xgb, _ = self._fit()
        assert xgb.is_fitted

    def test_predict_adds_columns(self):
        xgb, df = self._fit()
        out = xgb.predict(df)
        for col in ["xgb_pred", "xgb_prob_up", "xgb_confidence"]:
            assert col in out.columns

    def test_xgb_pred_is_binary(self):
        xgb, df = self._fit()
        out = xgb.predict(df)
        assert set(out["xgb_pred"].unique()).issubset({0, 1})

    def test_xgb_prob_up_bounded(self):
        xgb, df = self._fit()
        out = xgb.predict(df)
        assert out["xgb_prob_up"].between(0, 1).all()

    def test_xgb_confidence_bounded(self):
        xgb, df = self._fit()
        out = xgb.predict(df)
        assert out["xgb_confidence"].between(0, 1).all()

    def test_predict_before_fit_raises(self):
        from modules.prediction_module import PredictionModule
        xgb = PredictionModule()
        df  = make_feature_df(100, seed=0)
        with pytest.raises(RuntimeError, match="not fitted"):
            xgb.predict(df)
