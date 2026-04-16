"""
ML Trading System - Main Entry Point
=====================================
Orchestrates: Data → HMM Regime → GARCH → XGBoost → Signals → Simulation → Metrics → Visualization

Usage:
    python main.py                          # Full single-run pipeline
    python main.py --mode backtest          # Rolling window backtests
    python main.py --mode grid              # Grid optimization
    python main.py --symbol AAPL --years 10 # Custom symbol/period
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger_module import setup_logger, ResultsLogger, assert_dataframe, debug_dataframe_snapshot
from modules.data_module import DataModule
from modules.regime_module import MarketRegimeModule
from modules.volatility_module import GARCHVolatilityModule
from modules.prediction_module import PredictionModule
from modules.signal_engine import SignalEngine
from modules.simulation_module import TradeSimulator, SimulationConfig
from modules.metrics_module import MetricsModule
from modules.backtester_module import Backtester, BacktestConfig, BacktestWindow
from modules.grid_optimizer import GridOptimizer
from modules.visualization_module import VisualizationModule


DEFAULT_CONFIG = {
    "symbol":               "NIFTY",
    "source":               "duckdb_futures",   # duckdb_futures | duckdb_index | duckdb_equity | yahoo
    "data_start":           "2015-01-01",
    "data_end":             "2026-04-13",
    "hmm_n_states":         5,
    "hmm_n_iter":           200,
    "hmm_features":         ["fut_log_ret", "garch_vol",  "basis_pct", "vix_change"],   # None = use MarketRegimeModule.DEFAULT_FEATURES ["fut_log_ret", "garch_vol",  "basis_pct", "vix_change"]
                                    # Override: ["log_returns", "garch_vol", "vix_level", ...]
    "garch_returns_col":    "fut_log_ret",  # Column GARCH fits on.
                                            # "fut_log_ret" for futures (recommended)
                                            # "returns" for spot / index / equity
    # ── Signal voting ──────────────────────────────────────────────────
    # Indicators used and rationale for futures source:
    #   REMOVED: low_volatility — votes against high-vol trending moves (worst
    #            time to sit out on futures)
    #   REMOVED: low_atr       — same issue; ATR expands in breakouts
    #   REMOVED: volume        — futures contract volume is unreliable across
    #            rollover boundaries in a continuous series
    #   KEPT:    momentum, adx, price_above_ema, rsi, macd, price_above_vwap
    #            — all computed on adj_close (futures prices) and relevant
    #
    # min_votes reduced 5→3 because:
    #   - 9 indicators → 6 after removals; 5/9 ≈ 55% vs 3/6 = 50% (same bar)
    #   - HMM + XGBoost already filter aggressively; voting is a sanity check
    #     not a primary filter
    #
    # Set min_votes=0 to disable voting entirely (rely solely on HMM + XGBoost)
    "signal_indicators":    ["rsi", "price_above_vwap"], #["momentum", "adx", "price_above_ema", "rsi", "macd", "price_above_vwap"]
    "min_votes":            3,      # out of 6 active indicators
    "exit_votes":           3,      # exit when fewer than 2 agree
    "max_open_trades":      2,
    "total_capital":        100_000_000,
    "max_capital_per_trade": 100_000,
    "trading_expense":      100.0,
    "slippage_pct":         0.03,
    "risk_free_rate":       0.0738,
    "use_xgb_filter":       True,
    "confidence_sizing":    True,
}


class MLTradingSystem:
    """Top-level orchestrator for the full ML trading pipeline."""

    def __init__(self, config: dict = None, log_dir: str = "logs/",
                 results_dir: str = "results/", plots_dir: str = "plots/",
                 cache_dir: str = "cache/"):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.log_dir = log_dir
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.cache_dir = cache_dir

        self.logger = setup_logger("ml_trading", log_dir=log_dir, level=logging.INFO)
        self.results_logger = ResultsLogger(results_dir)
        self.data_module = DataModule(cache_dir=cache_dir)
        self.viz = VisualizationModule(output_dir=plots_dir)

        self.logger.info("=" * 70)
        self.logger.info("ML TRADING SYSTEM INITIALIZED")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")
        self.logger.info("=" * 70)

    def load_data(self, symbol: str = None, start: str = None, end: str = None,
                  source: str = None, force_refresh: bool = False) -> pd.DataFrame:
        symbol = symbol or self.config["symbol"]
        start  = start  or self.config["data_start"]
        end    = end    or self.config["data_end"]
        source = source or self.config.get("source", "duckdb_futures")
        self.logger.info(f"[Main] Loading data: {symbol} | {start} → {end} | source={source}")
        df = self.data_module.get_feature_data(
            symbol=symbol, start=start, end=end,
            source=source, force_refresh=force_refresh
        )
        assert_dataframe(df, "feature_df", ["Close", "returns", "log_returns"])
        self.logger.info(f"[Main] Data loaded: {len(df)} bars, {len(df.columns)} features")
        debug_dataframe_snapshot(df, "feature_df")
        return df

    def run_single_analysis(self, df: pd.DataFrame, title: str = "") -> dict:
        """Run the complete ML pipeline on a single DataFrame."""
        self.logger.info(f"[Main] Single analysis: {df.index[0].date()} → {df.index[-1].date()}")

        # 1. GARCH  — must run first so garch_vol is available as HMM feature
        self.logger.info("[Main] Step 1/6: GARCH Volatility Model...")
        garch = GARCHVolatilityModule()
        garch_col = self.config.get("garch_returns_col", "fut_log_ret")
        df = garch.add_to_dataframe(df, returns_col=garch_col)

        # 2. HMM — now has garch_vol + garch_next_vol available as features
        self.logger.info("[Main] Step 2/6: Market Regime Detection (HMM)...")
        hmm = MarketRegimeModule(n_states=self.config["hmm_n_states"],
                                  n_iter=self.config["hmm_n_iter"],
                                  features=self.config.get("hmm_features"))
        hmm.fit(df)
        df = hmm.predict(df)

        # 3. XGBoost
        self.logger.info("[Main] Step 3/6: XGBoost Prediction...")
        xgb = PredictionModule()
        xgb.fit(df)
        df = xgb.predict(df)

        # 4. Signals
        self.logger.info("[Main] Step 4/6: Signal Generation...")
        engine = SignalEngine(
            indicator_names=self.config.get("signal_indicators"),  # None = all registered
            min_votes=self.config["min_votes"],
            exit_votes=self.config["exit_votes"],
            max_open_trades=self.config["max_open_trades"],
            use_xgb_filter=self.config["use_xgb_filter"],
        )
        df = engine.generate(df)

        # 5. Simulation
        self.logger.info("[Main] Step 5/6: Trade Simulation...")
        sim_cfg = SimulationConfig(
            total_capital=self.config["total_capital"],
            max_capital_per_trade=self.config["max_capital_per_trade"],
            trading_expense=self.config["trading_expense"],
            slippage_pct=self.config["slippage_pct"],
            confidence_sizing=self.config["confidence_sizing"],
        )
        simulator = TradeSimulator(sim_cfg)
        result_df = simulator.simulate(df)
        trade_df  = simulator.get_trade_dataframe()

        # 6. Metrics
        self.logger.info("[Main] Step 6/6: Computing Metrics...")
        metrics_mod = MetricsModule(risk_free_rate=self.config["risk_free_rate"])
        equity = result_df["equity"].dropna()
        metrics = metrics_mod.compute(
            trade_df=trade_df, equity=equity,
            initial_capital=self.config["total_capital"],
            benchmark_returns=df["returns"],
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
        )

        # Save & visualize
        run_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_logger.log_backtest(
            run_id=run_id, params=self.config, metrics=metrics,
            trade_summary=trade_df,
            extra={"garch": garch.params},
        )
        plot_path = self.viz.plot_full_analysis(
            df=result_df,
            title=f"ML Trading: {self.config['symbol']} | {title}",
            filename=f"{run_id}_analysis.png",
        )
        self.viz.plot_regime_distribution(
            df=result_df, filename=f"{run_id}_regimes.png"
        )
        self.logger.info(f"[Main] Complete. Plot: {plot_path}")
        return {"metrics": metrics, "trade_df": trade_df,
                "result_df": result_df, "plot_path": plot_path, "run_id": run_id}

    def run_backtest(self, df: pd.DataFrame,
                     train_start_year: int = 2016,
                     first_test_year: int = 2018,
                     last_test_year: int = 2022) -> list:
        self.logger.info("[Main] Starting rolling window backtest...")
        bt_config = BacktestConfig(
            hmm_n_states=self.config["hmm_n_states"],
            min_votes=self.config["min_votes"],
            exit_votes=self.config["exit_votes"],
            max_open_trades=self.config["max_open_trades"],
            total_capital=self.config["total_capital"],
            max_capital_per_trade=self.config["max_capital_per_trade"],
            trading_expense=self.config["trading_expense"],
            slippage_pct=self.config["slippage_pct"],
            use_xgb_filter=self.config["use_xgb_filter"],
            confidence_sizing=self.config["confidence_sizing"],
            risk_free_rate=self.config["risk_free_rate"],
        )
        backtester = Backtester(df, bt_config, results_dir=self.results_dir)
        windows = backtester.build_rolling_windows(
            train_start_year=train_start_year,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
        )
        results = backtester.run_all(windows)
        for result in results:
            try:
                self.viz.plot_full_analysis(
                    df=result.signals_df,
                    title=f"Backtest {result.run_id}",
                    filename=f"{result.run_id}_analysis.png",
                )
            except Exception as e:
                self.logger.warning(f"[Main] Viz failed for {result.run_id}: {e}")
        return results

    def run_grid_optimization(self, df: pd.DataFrame, param_grid: dict = None,
                               train_start_year: int = 2016,
                               first_test_year: int = 2018,
                               last_test_year: int = 2021,
                               objective: str = "composite") -> dict:
        self.logger.info("[Main] Starting grid optimization...")
        default_grid = {
            "hmm_n_states": [3, 5, 7],
            "min_votes": [4, 5, 6],
            "max_capital_per_trade": [5_000, 10_000, 20_000],
        }
        grid = param_grid or default_grid
        base_config = BacktestConfig(
            hmm_n_states=self.config["hmm_n_states"],
            min_votes=self.config["min_votes"],
            exit_votes=self.config["exit_votes"],
            max_open_trades=self.config["max_open_trades"],
            total_capital=self.config["total_capital"],
            trading_expense=self.config["trading_expense"],
            slippage_pct=self.config["slippage_pct"],
        )
        temp_bt = Backtester(df, base_config, results_dir=self.results_dir)
        windows = temp_bt.build_rolling_windows(
            train_start_year=train_start_year,
            first_test_year=first_test_year,
            last_test_year=last_test_year,
        )
        optimizer = GridOptimizer(
            full_feature_df=df, windows=windows,
            param_grid=grid, base_config=base_config,
            results_dir=os.path.join(self.results_dir, "grid/"),
        )
        best = optimizer.run(objective=objective)
        results_df = optimizer.get_results_dataframe()
        params = list(grid.keys())
        if len(params) >= 2:
            self.viz.plot_grid_heatmap(
                results_df=results_df, x_param=params[0], y_param=params[1],
                metric="metric_sharpe_ratio", filename="grid_heatmap_sharpe.png",
            )
        return best


def parse_args():
    parser = argparse.ArgumentParser(description="ML Trading System")
    parser.add_argument("--mode",   choices=["single", "backtest", "grid"], default="single")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--source", default="duckdb_futures",
                        choices=["duckdb_futures", "duckdb_index", "duckdb_equity", "yahoo"],
                        help="Data source (default: duckdb_futures)")
    parser.add_argument("--years",   type=int,   default=10)
    parser.add_argument("--capital", type=float, default=100_000_000)
    parser.add_argument("--states",  type=int,   default=5)
    parser.add_argument("--votes",   type=int,   default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-xgb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    config = {
        "symbol":          args.symbol,
        "source":          args.source,
        "data_start":      start_date,
        "data_end":        end_date,
        "hmm_n_states":    args.states,
        "total_capital":   args.capital,
        "min_votes":       args.votes,
        "use_xgb_filter":  not args.no_xgb,
    }
    system = MLTradingSystem(config=config)
    df = system.load_data(source=args.source, force_refresh=args.refresh)

    if args.mode == "single":
        result = system.run_single_analysis(df, title=f"{args.symbol} Full Analysis")
        print(f"\n✓  Analysis complete")
        print(f"   Sharpe : {result['metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"   CAGR   : {result['metrics'].get('cagr', 0):.2%}")
        print(f"   MaxDD  : {result['metrics'].get('max_drawdown', 0):.2%}")
        print(f"   Trades : {result['metrics'].get('n_trades', 0)}")
        print(f"   Plot   : {result['plot_path']}")
    elif args.mode == "backtest":
        sy = int(start_date[:4])
        results = system.run_backtest(df, train_start_year=sy,
                                       first_test_year=sy+2,
                                       last_test_year=min(sy+6, int(end_date[:4])-1))
        print(f"\n✓  Backtest complete. {len(results)} windows.")
    elif args.mode == "grid":
        best = system.run_grid_optimization(df=df)
        print(f"\n✓  Grid optimization complete.")
        print(f"   Best params : {best['best_params']}")
        print(f"   Best score  : {best['best_score']:.4f}")


if __name__ == "__main__":
    main()
