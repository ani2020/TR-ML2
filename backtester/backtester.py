"""
backtester/backtester.py
Walk-forward backtesting engine.

Flow per window:
  1. Slice train / test data
  2. Train all models on train window
  3. Predict + generate signals on test window
  4. Simulate trades
  5. Compute metrics
  6. Record results to JSON + trade journal
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import MasterConfig
from data.data_module import DataModule
from models.regime_model import RegimeModel
from models.volatility_model import VolatilityModel
from models.prediction_model import PredictionModel
from strategy.signal_engine import SignalEngine
from simulation.trade_simulator import TradeSimulator
from metrics.metrics_module import MetricsModule
from logger.trading_logger import get_logger, RunRecorder

logger = get_logger("trading.backtest")


class BacktestWindow:
    """One train/test window result."""
    def __init__(self, train_start, train_end, test_start, test_end,
                 metrics, trade_df, test_df_signals, run_id):
        self.train_start = train_start
        self.train_end   = train_end
        self.test_start  = test_start
        self.test_end    = test_end
        self.metrics     = metrics
        self.trade_df    = trade_df
        self.test_df_signals = test_df_signals
        self.run_id      = run_id


class Backtester:
    """
    Orchestrates walk-forward backtesting over multiple time windows.
    """

    def __init__(self, config: MasterConfig, feature_df: pd.DataFrame,
                 output_dir: Optional[str] = None):
        self.cfg        = config
        self.feature_df = feature_df
        self.output_dir = output_dir or config.backtest.output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self._metrics_mod = MetricsModule(config)

    # ── run all windows ──────────────────────────────────────
    def run_all(self) -> List[BacktestWindow]:
        results = []
        for i, window in enumerate(self.cfg.backtest.windows):
            train_start, train_end, test_start, test_end = window
            logger.info(f"\n{'='*70}")
            logger.info(f"[Backtester] Window {i+1}/{len(self.cfg.backtest.windows)}: "
                        f"Train {train_start}→{train_end}  Test {test_start}→{test_end}")
            logger.info(f"{'='*70}")
            try:
                res = self._run_window(train_start, train_end, test_start, test_end, window_idx=i)
                results.append(res)
            except Exception as e:
                logger.error(f"[Backtester] Window {i+1} failed: {e}", exc_info=True)
        return results

    # ── single window ────────────────────────────────────────
    def _run_window(self, train_start: str, train_end: str,
                    test_start: str, test_end: str,
                    window_idx: int = 0) -> BacktestWindow:

        run_id = f"w{window_idx+1}_{train_start[:4]}_{test_start[:4]}"
        recorder = RunRecorder(output_dir=self.output_dir, run_id=run_id)

        # ── 1. Slice data ─────────────────────────────────────
        assert self.feature_df is not None and not self.feature_df.empty, \
            "feature_df must be non-empty"
        train_df = self.feature_df.loc[train_start:train_end].copy()
        test_df  = self.feature_df.loc[test_start:test_end].copy()
        assert len(train_df) > 100, f"Train window too short: {len(train_df)} rows"
        assert len(test_df)  > 10,  f"Test window too short: {len(test_df)} rows"

        logger.info(f"  Train rows: {len(train_df)}  Test rows: {len(test_df)}")
        recorder.add("data_summary", {
            "train_rows": len(train_df), "test_rows": len(test_df),
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
        })

        # ── 2. Train models ───────────────────────────────────
        t0 = time.time()

        # HMM Regime
        regime_model = RegimeModel(self.cfg)
        regime_model.fit(train_df)
        train_df = regime_model.predict(train_df)
        regime_model.log_state_probabilities(train_df)

        recorder.add("regime_state_map", regime_model.state_label_map())
        recorder.add("regime_distribution",
                     train_df["hmm_regime"].value_counts().to_dict())

        # GARCH Volatility
        vol_model = VolatilityModel(self.cfg)
        vol_model.fit(train_df)
        train_df = vol_model.add_to_df(train_df)

        # XGBoost Prediction
        pred_model = PredictionModel(self.cfg)
        pred_model.fit(train_df)
        train_df = pred_model.predict(train_df)

        logger.info(f"  Models trained in {time.time()-t0:.1f}s")

        # ── 3. Predict on test window ─────────────────────────
        test_df = regime_model.predict(test_df)
        test_df = vol_model.add_to_df(test_df)
        test_df = pred_model.predict(test_df)

        # ── 4. Generate signals ───────────────────────────────
        signal_engine = SignalEngine(self.cfg)
        test_df = signal_engine.generate(test_df)

        # Also generate on train (for in-sample visualisation)
        train_df = signal_engine.generate(train_df)

        # ── 5. Simulate trades ────────────────────────────────
        simulator = TradeSimulator(
            self.cfg, ticker=self.cfg.data.ticker,
            run_id=run_id, output_dir=self.output_dir
        )
        test_df_sim = simulator.run(test_df)
        trade_df    = simulator.trade_summary()

        logger.info(f"  Trades executed: {len(trade_df)}")

        # ── 6. Compute metrics ────────────────────────────────
        equity_curve = test_df_sim["equity_curve"]
        bench_returns = test_df["log_return"].dropna() if "log_return" in test_df.columns else None
        window_label  = f"Train:{train_start}→{train_end}  Test:{test_start}→{test_end}"
        metrics = self._metrics_mod.compute(
            trade_df, equity_curve, bench_returns, window_label=window_label
        )

        # ── 7. Record results ─────────────────────────────────
        recorder.add("config", self.cfg.to_dict())
        recorder.add("metrics", metrics)
        recorder.add("trade_summary", trade_df.to_dict(orient="records") if not trade_df.empty else [])
        recorder.add("feature_importance",
                     pred_model.feature_importance.to_dict() if pred_model.feature_importance is not None else {})
        saved_path = recorder.save()
        logger.info(f"  Run record saved → {saved_path}")

        # ── 8. Assertions / debug ─────────────────────────────
        assert "equity_curve" in test_df_sim.columns, "equity_curve missing from sim output"
        assert "hmm_regime" in test_df.columns, "hmm_regime missing from test predictions"
        assert metrics.get("sharpe") is not None, "Sharpe ratio not computed"

        if not trade_df.empty:
            assert "entry_price" in trade_df.columns, "entry_price missing from trade summary"
            assert "exit_price" in trade_df.columns, "exit_price missing from trade summary"
            logger.debug(f"  Trade summary (first 5):\n{trade_df.head(5).to_string()}")

        return BacktestWindow(
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            metrics=metrics, trade_df=trade_df,
            test_df_signals=test_df_sim, run_id=run_id
        )

    # ── aggregate summary across all windows ─────────────────
    def summary_table(self, results: List[BacktestWindow]) -> pd.DataFrame:
        rows = []
        for r in results:
            row = {
                "train": f"{r.train_start[:4]}→{r.train_end[:4]}",
                "test":  f"{r.test_start[:4]}→{r.test_end[:4]}",
            }
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)
