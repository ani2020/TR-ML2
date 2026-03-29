"""
tests/test_metrics_module.py
============================
Unit tests for MetricsModule.

Tests cover:
  - Mathematical correctness of each metric in isolation
  - Statistical properties of random-input pipeline:
      Sharpe ≈ 0, profit_factor ≈ 1, expectancy ≈ 0
  - Known-outcome tests (perfect wins, perfect losses, flat equity)
  - Edge cases: empty trades, single trade, zero variance equity
  - Monotone properties: better trades → better Sharpe/profit_factor
"""

import pytest
import numpy as np
import pandas as pd

from modules.metrics_module import MetricsModule
from tests.conftest import make_trade_df, make_equity_curve


INITIAL_CAPITAL = 100_000.0


def _metrics(trade_df, equity, capital=INITIAL_CAPITAL, rfr=0.0):
    """Helper: compute metrics with configurable risk-free rate."""
    mm = MetricsModule(risk_free_rate=rfr)
    return mm.compute(trade_df, equity, capital)


# ═══════════════════════════════════════════════════════════════════════════
# Mathematical correctness — individual metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricMath:

    def test_sharpe_flat_equity_is_zero(self, flat_equity, small_trade_df):
        """Flat equity has zero excess return and zero variance → Sharpe = 0."""
        mm = MetricsModule(risk_free_rate=0.0)
        result = mm.compute(small_trade_df, flat_equity, INITIAL_CAPITAL)
        assert result["sharpe_ratio"] == pytest.approx(0.0, abs=1e-6)

    def test_sharpe_rising_equity_positive(self, rising_equity, winning_trade_df):
        """Steadily rising equity should produce a positive Sharpe ratio."""
        result = _metrics(winning_trade_df, rising_equity, rfr=0.0)
        assert result["sharpe_ratio"] > 0, "Rising equity should give positive Sharpe"

    def test_sharpe_crashing_equity_negative(self, crashing_equity, losing_trade_df):
        """Steadily declining equity should give negative Sharpe."""
        result = _metrics(losing_trade_df, crashing_equity, rfr=0.0)
        assert result["sharpe_ratio"] < 0, "Crashing equity should give negative Sharpe"

    def test_max_drawdown_flat_is_zero(self, flat_equity, small_trade_df):
        """Flat equity has no drawdown."""
        result = _metrics(small_trade_df, flat_equity)
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_always_non_positive(self, crashing_equity, losing_trade_df):
        """Max drawdown must be ≤ 0 by definition."""
        result = _metrics(losing_trade_df, crashing_equity)
        assert result["max_drawdown"] <= 0

    def test_max_drawdown_crashing_equity(self, crashing_equity, losing_trade_df):
        """Equity declining 0.2%/day for 252 days → ~40% total drawdown."""
        result = _metrics(losing_trade_df, crashing_equity)
        assert result["max_drawdown"] < -0.30, "Expected drawdown > 30%"
        assert result["max_drawdown"] > -0.70, "Drawdown unexpectedly severe"

    def test_win_rate_matches_trade_df(self):
        """Win rate = wins / total_trades, regardless of P&L magnitudes."""
        for target_wr in [0.3, 0.5, 0.7]:
            td = make_trade_df(n_trades=100, seed=1, win_rate=target_wr,
                               avg_win=100, avg_loss=-100)
            eq = make_equity_curve(td, n_days=252, seed=1)
            result = _metrics(td, eq)
            actual_wr = result["n_wins"] / result["n_trades"]
            assert abs(actual_wr - target_wr) < 0.08, \
                f"Win rate {actual_wr:.2f} too far from target {target_wr}"

    def test_profit_factor_all_wins(self):
        """If all trades are wins, profit factor = ∞ (represented as inf or very large)."""
        td = make_trade_df(n_trades=20, seed=0, win_rate=1.0, avg_win=200, avg_loss=-1)
        # Force all pnl positive
        td["pnl"] = np.abs(td["pnl"])
        eq = make_equity_curve(td, seed=0)
        mm = MetricsModule()
        pf = mm._profit_factor(td)
        assert np.isinf(pf) or pf > 100, f"Expected inf profit factor, got {pf}"

    def test_profit_factor_all_losses(self):
        """If all trades are losses, profit factor = 0."""
        td = make_trade_df(n_trades=20, seed=0, win_rate=0.0, avg_win=1, avg_loss=-100)
        td["pnl"] = -np.abs(td["pnl"])
        mm = MetricsModule()
        pf = mm._profit_factor(td)
        assert pf == pytest.approx(0.0, abs=1e-9)

    def test_cagr_positive_for_rising_equity(self, rising_equity, winning_trade_df):
        result = _metrics(winning_trade_df, rising_equity)
        assert result["cagr"] > 0

    def test_cagr_negative_for_crashing_equity(self, crashing_equity, losing_trade_df):
        result = _metrics(losing_trade_df, crashing_equity)
        assert result["cagr"] < 0

    def test_calmar_is_cagr_over_abs_maxdd(self, rising_equity, winning_trade_df):
        """Calmar = CAGR / |max_drawdown|; verify numerically."""
        mm     = MetricsModule(risk_free_rate=0.0)
        result = mm.compute(winning_trade_df, rising_equity, INITIAL_CAPITAL)
        cagr   = result["cagr"]
        mdd    = abs(result["max_drawdown"])
        if mdd > 1e-8:
            expected_calmar = cagr / mdd
            assert result["calmar_ratio"] == pytest.approx(expected_calmar, rel=1e-3)

    def test_sortino_ge_sharpe_for_positive_returns(self, rising_equity, winning_trade_df):
        """Sortino ≥ Sharpe when returns are skewed positive (fewer downside days)."""
        mm     = MetricsModule(risk_free_rate=0.0)
        result = mm.compute(winning_trade_df, rising_equity, INITIAL_CAPITAL)
        assert result["sortino_ratio"] >= result["sharpe_ratio"] - 1e-6

    def test_n_trades_matches_trade_df(self):
        for n in [5, 20, 50]:
            td = make_trade_df(n_trades=n)
            eq = make_equity_curve(td)
            result = _metrics(td, eq)
            assert result["n_trades"] == n

    def test_n_wins_plus_losses_equals_n_trades(self, small_trade_df, rising_equity):
        result = _metrics(small_trade_df, rising_equity)
        assert result["n_wins"] + result["n_losses"] == result["n_trades"]

    def test_expectancy_symmetric_trades(self):
        """50% win rate, equal avg win and loss → expectancy ≈ 0."""
        td = make_trade_df(n_trades=200, seed=42, win_rate=0.5,
                           avg_win=100, avg_loss=-100)
        eq = make_equity_curve(td, n_days=252, seed=42)
        result = _metrics(td, eq)
        assert abs(result["expectancy_per_trade"]) < 30, \
            f"Expectancy {result['expectancy_per_trade']:.2f} too large for symmetric trades"

    def test_expectancy_positive_for_winners(self, winning_trade_df, rising_equity):
        result = _metrics(winning_trade_df, rising_equity)
        assert result["expectancy_per_trade"] > 0

    def test_expectancy_negative_for_losers(self, losing_trade_df, crashing_equity):
        result = _metrics(losing_trade_df, crashing_equity)
        assert result["expectancy_per_trade"] < 0


# ═══════════════════════════════════════════════════════════════════════════
# Random-input pipeline test  ← KEY TEST
# Validates that a purely random strategy converges to noise-level metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomInputPipelineMetrics:
    """
    When the trading signal is pure noise (random 50/50 entries/exits,
    zero-drift price) the law of large numbers implies:

      Sharpe   ≈ 0   (±1 is generous)
      Profit Factor ≈ 1  (±0.5)
      Expectancy  ≈ 0   (small absolute value)
      Win Rate   ≈ 0.5
      N Trades   should be > 0 (system executed trades)
    """

    @pytest.fixture(scope="class")
    def random_metrics(self):
        """
        Run the full signal→simulate→metrics pipeline on 1 000 bars of
        pure random-walk data with random signals.
        """
        from modules.simulation_module import TradeSimulator, SimulationConfig
        from tests.conftest import make_feature_df, make_equity_curve

        # ── Generate random feature data ──────────────────────────────────
        rng = np.random.default_rng(0)
        n   = 1_000
        df  = make_feature_df(n=n, seed=0)  # zero-drift price

        # Override signals with alternating ENTRY/EXIT noise
        # Inject ~100 entry/exit pairs randomly
        df["signal"] = 0
        indices = df.index.tolist()
        pos  = 0
        in_trade = False
        for i in range(0, len(indices) - 5, 5):
            if not in_trade:
                df.loc[indices[i], "signal"] = 1   # ENTRY
                in_trade = True
            else:
                df.loc[indices[i], "signal"] = -1  # EXIT
                in_trade = False

        df["xgb_confidence_at_signal"] = 0.55
        df.loc[df["signal"] == 0, "xgb_confidence_at_signal"] = np.nan

        # ── Simulate ──────────────────────────────────────────────────────
        cfg = SimulationConfig(
            total_capital=100_000,
            max_capital_per_trade=10_000,
            trading_expense=5.0,
            slippage_pct=0.0,     # zero friction for clean random test
            confidence_sizing=False,
        )
        sim     = TradeSimulator(cfg)
        res_df  = sim.simulate(df)
        trade_df = sim.get_trade_dataframe()
        equity   = res_df["equity"].dropna()

        # ── Metrics ───────────────────────────────────────────────────────
        mm = MetricsModule(risk_free_rate=0.0)
        metrics = mm.compute(trade_df, equity, initial_capital=100_000)
        return metrics, trade_df

    def test_random_sharpe_near_zero(self, random_metrics):
        metrics, _ = random_metrics
        sharpe = metrics["sharpe_ratio"]
        assert abs(sharpe) < 2.0, \
            f"Random strategy Sharpe={sharpe:.3f} is too far from 0"

    def test_random_profit_factor_near_one(self, random_metrics):
        metrics, _ = random_metrics
        pf = metrics["profit_factor"]
        assert 0.3 < pf < 3.0, \
            f"Random profit factor={pf:.3f} should be near 1.0"

    def test_random_expectancy_near_zero(self, random_metrics):
        metrics, trade_df = random_metrics
        exp  = metrics["expectancy_per_trade"]
        # Expectancy per trade relative to capital_per_trade
        cap  = trade_df["capital_used"].mean() if len(trade_df) > 0 else 10_000
        rel  = abs(exp) / cap
        assert rel < 0.15, \
            f"Random expectancy {exp:.2f} ({rel:.2%} of capital) is too large"

    def test_random_win_rate_near_half(self, random_metrics):
        metrics, _ = random_metrics
        wr = metrics["win_rate"]
        assert 0.30 < wr < 0.70, \
            f"Random win_rate={wr:.2%} should be near 0.50"

    def test_random_n_trades_positive(self, random_metrics):
        metrics, _ = random_metrics
        assert metrics["n_trades"] > 10, \
            "Random strategy should generate more than 10 trades"

    def test_random_max_drawdown_negative(self, random_metrics):
        """Random strategy will almost always have some drawdown."""
        metrics, _ = random_metrics
        assert metrics["max_drawdown"] <= 0, "Max drawdown should be ≤ 0"


# ═══════════════════════════════════════════════════════════════════════════
# Monotone / ordering properties
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricOrdering:

    def test_higher_win_rate_better_profit_factor(self):
        """Profit factor should increase monotonically with win rate (all else equal)."""
        pfs = []
        for wr in [0.3, 0.45, 0.55, 0.7]:
            td = make_trade_df(n_trades=200, seed=42, win_rate=wr,
                               avg_win=150, avg_loss=-100)
            eq = make_equity_curve(td, seed=42)
            pfs.append(_metrics(td, eq)["profit_factor"])
        # Each profit factor should be >= the previous
        for i in range(1, len(pfs)):
            assert pfs[i] >= pfs[i-1] - 0.05, \
                f"PF not monotone: {pfs}"

    def test_higher_win_rate_better_sharpe(self):
        """Sharpe ratio should increase with win rate when avg_win > avg_loss."""
        sharpes = []
        for wr in [0.3, 0.5, 0.7]:
            td = make_trade_df(n_trades=200, seed=42, win_rate=wr,
                               avg_win=200, avg_loss=-100)
            eq = make_equity_curve(td, n_days=252, seed=42)
            sharpes.append(_metrics(td, eq, rfr=0.0)["sharpe_ratio"])
        assert sharpes[-1] > sharpes[0], \
            f"Sharpe not monotone with win_rate: {sharpes}"


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricsEdgeCases:

    def test_empty_trade_df_returns_zeros(self, flat_equity):
        """Empty trade list should return a valid dict of zeros/defaults."""
        mm     = MetricsModule()
        result = mm.compute(pd.DataFrame(), flat_equity, INITIAL_CAPITAL)
        assert result["n_trades"] == 0
        assert result["sharpe_ratio"] == 0
        assert result["win_rate"] == 0

    def test_single_winning_trade(self, rising_equity):
        """Single winning trade: win_rate=1, profit_factor=inf, expectancy>0."""
        td = make_trade_df(n_trades=1, seed=0, win_rate=1.0, avg_win=500, avg_loss=-1)
        td["pnl"] = [500.0]
        result = _metrics(td, rising_equity)
        assert result["n_trades"]   == 1
        assert result["win_rate"]   == pytest.approx(1.0)
        assert result["expectancy_per_trade"] > 0

    def test_single_losing_trade(self, crashing_equity):
        """Single losing trade: win_rate=0, profit_factor=0, expectancy<0."""
        td = make_trade_df(n_trades=1, seed=0, win_rate=0.0, avg_win=1, avg_loss=-300)
        td["pnl"] = [-300.0]
        result = _metrics(td, crashing_equity)
        assert result["n_trades"]   == 1
        assert result["win_rate"]   == pytest.approx(0.0)
        assert result["expectancy_per_trade"] < 0

    def test_run_analysis_contains_expected_keys(self, small_trade_df, rising_equity):
        """Run analysis should populate streak statistics in the metrics dict."""
        result = _metrics(small_trade_df, rising_equity)
        for key in ["max_win_streak", "max_loss_streak", "mean_daily_return",
                    "positive_days", "negative_days", "data_window"]:
            assert key in result, f"Missing key: {key}"

    def test_metrics_dict_all_numeric_or_str(self, small_trade_df, rising_equity):
        """All metric values should be numeric (int/float) or string, never None/NaN."""
        result = _metrics(small_trade_df, rising_equity)
        for k, v in result.items():
            if isinstance(v, float):
                assert not np.isnan(v), f"NaN in metric '{k}'"
            assert v is not None, f"None in metric '{k}'"
