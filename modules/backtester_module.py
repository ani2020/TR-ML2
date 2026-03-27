"""
Backtester Module
- Rolling train/test window backtesting
- Full pipeline: features → HMM → GARCH → XGBoost → Signals → Simulate → Metrics
- Logs everything for manual review
"""

import logging
import traceback
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from modules.regime_module import MarketRegimeModule
from modules.volatility_module import GARCHVolatilityModule
from modules.prediction_module import PredictionModule
from modules.signal_engine import SignalEngine
from modules.simulation_module import TradeSimulator, SimulationConfig
from modules.metrics_module import MetricsModule
from modules.logger_module import (
    ResultsLogger, assert_dataframe, assert_no_nan_in_col, debug_dataframe_snapshot
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class BacktestConfig:
    # HMM
    hmm_n_states: int = 5
    hmm_features: Optional[List[str]] = None
    hmm_n_iter: int = 200

    # XGBoost
    xgb_params: Optional[Dict] = None
    xgb_features: Optional[List[str]] = None

    # Signal
    indicator_names: Optional[List[str]] = None
    min_votes: int = 5
    exit_votes: int = 3
    use_xgb_filter: bool = True
    max_open_trades: int = 2

    # Simulation
    total_capital: float = 100_000.0
    max_capital_per_trade: float = 10_000.0
    trading_expense: float = 10.0
    slippage_pct: float = 0.001
    confidence_sizing: bool = True

    # Metrics
    risk_free_rate: float = 0.02


class BacktestResult:
    def __init__(self, window: BacktestWindow, metrics: Dict, trades_df: pd.DataFrame,
                 signals_df: pd.DataFrame, run_id: str, config: BacktestConfig):
        self.window = window
        self.metrics = metrics
        self.trades_df = trades_df
        self.signals_df = signals_df
        self.run_id = run_id
        self.config = config

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "window": asdict(self.window),
            "parameters": asdict(self.config),
            "metrics": self.metrics,
        }


class Backtester:
    """
    Runs rolling window backtests.

    Example windows:
        Train: 2016–2017   Test: 2018
        Train: 2016–2018   Test: 2019
        Train: 2016–2019   Test: 2020
    """

    def __init__(
        self,
        full_feature_df: pd.DataFrame,
        config: BacktestConfig,
        results_dir: str = "results/",
    ):
        self.df = full_feature_df
        self.config = config
        self.results_logger = ResultsLogger(results_dir)
        self._results: List[BacktestResult] = []

    # ── Window Helpers ────────────────────────
    def build_rolling_windows(
        self,
        train_start_year: int,
        first_test_year: int,
        last_test_year: int,
        train_anchor: Optional[str] = None,
    ) -> List[BacktestWindow]:
        """
        Build expanding train windows with rolling single-year test windows.

        Example: train_start=2016, first_test=2018, last_test=2020
          → (2016-2017 → 2018), (2016-2018 → 2019), (2016-2019 → 2020)
        """
        windows = []
        anchor = train_anchor or f"{train_start_year}-01-01"
        for test_year in range(first_test_year, last_test_year + 1):
            train_end = f"{test_year - 1}-12-31"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"
            windows.append(BacktestWindow(
                train_start=anchor,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))
        logger.info(f"[Backtester] Built {len(windows)} rolling windows")
        for w in windows:
            logger.info(f"  Train: {w.train_start} → {w.train_end}  |  Test: {w.test_start} → {w.test_end}")
        return windows

    def add_custom_window(self, windows: List[BacktestWindow], window: BacktestWindow):
        windows.append(window)
        return windows

    # ── Single Run ────────────────────────────
    def run_single(
        self,
        window: BacktestWindow,
        run_id: Optional[str] = None,
    ) -> Optional[BacktestResult]:
        """Execute a full pipeline for one train/test window."""
        run_id = run_id or f"{window.train_start[:4]}-{window.train_end[:4]}_test{window.test_start[:4]}"
        logger.info("=" * 70)
        logger.info(f"[Backtest] RUN: {run_id}")
        logger.info(f"[Backtest] Train: {window.train_start} → {window.train_end}")
        logger.info(f"[Backtest] Test : {window.test_start} → {window.test_end}")

        try:
            # ── i. Slice data ─────────────────
            train_df = self.df.loc[window.train_start:window.train_end].copy()
            test_df  = self.df.loc[window.test_start:window.test_end].copy()

            assert_dataframe(train_df, "train_df", ["Close", "returns", "log_returns"])
            assert_dataframe(test_df, "test_df", ["Close", "returns"])
            logger.info(f"[Backtest] Train: {len(train_df)} bars | Test: {len(test_df)} bars")

            # ── ii. Train HMM ─────────────────
            logger.info("[Backtest] Step 1: Training HMM...")
            hmm = MarketRegimeModule(
                n_states=self.config.hmm_n_states,
                features=self.config.hmm_features,
                n_iter=self.config.hmm_n_iter,
            )
            hmm.fit(train_df)

            # ── iii. Train GARCH ──────────────
            logger.info("[Backtest] Step 2: Fitting GARCH...")
            garch = GARCHVolatilityModule()
            garch.fit(train_df["returns"])

            # ── iv. Train XGBoost ─────────────
            logger.info("[Backtest] Step 3: Adding regimes + GARCH to train data...")
            train_enriched = hmm.predict(train_df)
            train_enriched = garch.add_to_dataframe(train_enriched)

            logger.info("[Backtest] Step 4: Training XGBoost...")
            xgb = PredictionModule(
                feature_cols=self.config.xgb_features,
                xgb_params=self.config.xgb_params,
            )
            xgb.fit(train_enriched)

            # ── v. Predict on test ────────────
            logger.info("[Backtest] Step 5: Predicting on test data...")
            test_enriched = hmm.predict(test_df)
            test_enriched = garch.add_to_dataframe(test_enriched)
            test_enriched = xgb.predict(test_enriched)

            debug_dataframe_snapshot(test_enriched, "test_enriched", n=3)

            # ── vi. Generate signals ──────────
            logger.info("[Backtest] Step 6: Generating signals...")
            engine = SignalEngine(
                indicator_names=self.config.indicator_names,
                min_votes=self.config.min_votes,
                exit_votes=self.config.exit_votes,
                use_xgb_filter=self.config.use_xgb_filter,
                max_open_trades=self.config.max_open_trades,
            )
            signal_df = engine.generate(test_enriched)

            # ── vii. Simulate trades ──────────
            logger.info("[Backtest] Step 7: Simulating trades...")
            sim_cfg = SimulationConfig(
                total_capital=self.config.total_capital,
                max_capital_per_trade=self.config.max_capital_per_trade,
                trading_expense=self.config.trading_expense,
                slippage_pct=self.config.slippage_pct,
                confidence_sizing=self.config.confidence_sizing,
            )
            simulator = TradeSimulator(sim_cfg)
            result_df = simulator.simulate(signal_df)
            trade_df = simulator.get_trade_dataframe()
            equity = result_df["equity"].dropna()

            # ── viii. Calculate metrics ───────
            logger.info("[Backtest] Step 8: Computing metrics...")
            benchmark_returns = test_df["returns"] if "returns" in test_df.columns else None
            metrics_module = MetricsModule(risk_free_rate=self.config.risk_free_rate)
            metrics = metrics_module.compute(
                trade_df=trade_df,
                equity=equity,
                initial_capital=self.config.total_capital,
                benchmark_returns=benchmark_returns,
                start_date=window.test_start,
                end_date=window.test_end,
            )
            metrics["run_id"] = run_id
            metrics["garch_params"] = garch.params

            # ── ix. Log results ───────────────
            logger.info("[Backtest] Step 9: Logging results...")
            self.results_logger.log_backtest(
                run_id=run_id,
                params=asdict(self.config),
                metrics=metrics,
                trade_summary=trade_df,
                extra={"window": asdict(window), "garch": garch.params},
            )

            result = BacktestResult(
                window=window,
                metrics=metrics,
                trades_df=trade_df,
                signals_df=result_df,
                run_id=run_id,
                config=self.config,
            )
            self._results.append(result)
            logger.info(f"[Backtest] ✓ Run complete: {run_id}")
            return result

        except Exception as e:
            logger.error(f"[Backtest] ✗ Run FAILED: {run_id} — {e}")
            logger.error(traceback.format_exc())
            return None

    # ── Multi-Window Run ──────────────────────
    def run_all(self, windows: List[BacktestWindow]) -> List[BacktestResult]:
        """Run all windows sequentially."""
        results = []
        for i, window in enumerate(windows):
            run_id = f"run_{i+1:02d}_train{window.train_start[:4]}-{window.train_end[:4]}_test{window.test_start[:4]}"
            result = self.run_single(window, run_id=run_id)
            if result:
                results.append(result)

        logger.info(f"[Backtester] Completed {len(results)}/{len(windows)} windows successfully")
        self._log_aggregate_summary(results)
        return results

    def _log_aggregate_summary(self, results: List[BacktestResult]):
        if not results:
            return
        logger.info("\n" + "=" * 70)
        logger.info("[Backtester] AGGREGATE SUMMARY")
        header = f"{'Run ID':40s} | {'Sharpe':>7} | {'CAGR':>7} | {'MaxDD':>7} | {'WinRate':>7} | {'Trades':>6}"
        logger.info(header)
        logger.info("-" * 85)
        for r in results:
            m = r.metrics
            logger.info(
                f"{r.run_id:40s} | "
                f"{m.get('sharpe_ratio', 0):>7.2f} | "
                f"{m.get('cagr', 0):>7.2%} | "
                f"{m.get('max_drawdown', 0):>7.2%} | "
                f"{m.get('win_rate', 0):>7.2%} | "
                f"{m.get('n_trades', 0):>6d}"
            )

    @property
    def results(self) -> List[BacktestResult]:
        return self._results
