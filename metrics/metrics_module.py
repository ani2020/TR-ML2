"""
metrics/metrics_module.py
Comprehensive performance metrics for the ML Trading System.

Includes: Sharpe, Sortino, Calmar, Max Drawdown, Win Rate,
          Alpha, Profit Factor, Expectancy, CAGR, and more.
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading.metrics")


class MetricsModule:
    """
    Compute all performance metrics from:
      - trade_summary_df  : DataFrame from TradeSimulator.trade_summary()
      - equity_curve      : list/array of portfolio values
      - benchmark_returns : optional pd.Series of benchmark daily returns
    """

    def __init__(self, config):
        self.cfg = config.metrics
        self._rf  = config.metrics.risk_free_rate
        self._td  = config.metrics.trading_days

    # ── main entry ───────────────────────────────────────────
    def compute(self,
                trade_df: pd.DataFrame,
                equity_curve: pd.Series,
                benchmark_returns: Optional[pd.Series] = None,
                window_label: str = "") -> Dict:

        if trade_df is None or trade_df.empty:
            logger.warning("[Metrics] No trades to analyse.")
            return {"window": window_label, "error": "no_trades"}

        metrics = {"window": window_label}
        metrics.update(self._trade_metrics(trade_df))
        metrics.update(self._equity_metrics(equity_curve))
        if benchmark_returns is not None:
            metrics.update(self._alpha(equity_curve, benchmark_returns))
        metrics.update(self._run_analysis(trade_df, equity_curve))

        self._log(metrics)
        return metrics

    # ── trade-level metrics ──────────────────────────────────
    def _trade_metrics(self, df: pd.DataFrame) -> Dict:
        closed = df[df["status"].isin(["CLOSED", "FORCE_CLOSED"])]
        if closed.empty:
            return {}

        n_trades  = len(closed)
        wins      = closed[closed["pnl"] > 0]
        losses    = closed[closed["pnl"] <= 0]
        win_rate  = len(wins) / n_trades if n_trades else 0
        avg_win   = wins["pnl"].mean() if not wins.empty else 0
        avg_loss  = losses["pnl"].mean() if not losses.empty else 0
        gross_profit = wins["pnl"].sum()
        gross_loss   = abs(losses["pnl"].sum())
        profit_factor= gross_profit / gross_loss if gross_loss else float("inf")
        expectancy   = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        max_capital  = df["capital_used"].max() if "capital_used" in df.columns else float("nan")

        return {
            "n_trades":       n_trades,
            "win_rate":       round(win_rate, 4),
            "avg_win":        round(avg_win, 2),
            "avg_loss":       round(avg_loss, 2),
            "gross_profit":   round(gross_profit, 2),
            "gross_loss":     round(gross_loss, 2),
            "profit_factor":  round(profit_factor, 4),
            "expectancy":     round(expectancy, 2),
            "max_capital_used": round(max_capital, 2),
        }

    # ── equity / return metrics ──────────────────────────────
    def _equity_metrics(self, equity: pd.Series) -> Dict:
        if equity is None or len(equity) < 2:
            return {}

        eq = np.array(equity, dtype=float)
        daily_ret = np.diff(eq) / eq[:-1]

        # Basic
        total_return = (eq[-1] - eq[0]) / eq[0]
        n_days = len(eq)
        n_years = n_days / self._td
        cagr = (eq[-1] / eq[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / np.where(peak == 0, 1, peak)
        max_dd = dd.min()

        # Sharpe
        excess = daily_ret - self._rf / self._td
        sharpe = (np.mean(excess) / (np.std(excess) + 1e-10)) * math.sqrt(self._td)

        # Sortino (downside deviation)
        neg_ret = excess[excess < 0]
        down_dev = math.sqrt(np.mean(neg_ret ** 2)) if len(neg_ret) > 0 else 1e-10
        sortino = (np.mean(excess) / down_dev) * math.sqrt(self._td)

        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else float("inf")

        return {
            "total_return": round(total_return, 4),
            "cagr":         round(cagr, 4),
            "sharpe":       round(sharpe, 4),
            "sortino":      round(sortino, 4),
            "calmar":       round(calmar, 4),
            "max_drawdown": round(max_dd, 4),
        }

    # ── alpha vs benchmark ───────────────────────────────────
    def _alpha(self, equity: pd.Series, bench_ret: pd.Series) -> Dict:
        eq = np.array(equity, dtype=float)
        strat_ret = np.diff(eq) / eq[:-1]
        n = min(len(strat_ret), len(bench_ret))
        strat_ret  = strat_ret[-n:]
        bench_arr  = bench_ret.values[-n:]
        cov_mat    = np.cov(strat_ret, bench_arr)
        beta       = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-10)
        alpha      = (np.mean(strat_ret) - self._rf / self._td -
                      beta * (np.mean(bench_arr) - self._rf / self._td)) * self._td
        return {"alpha": round(alpha, 4), "beta": round(beta, 4)}

    # ── run analysis ─────────────────────────────────────────
    def _run_analysis(self, df: pd.DataFrame, equity: pd.Series) -> Dict:
        """Data window and mean return summary."""
        closed = df[df["status"].isin(["CLOSED", "FORCE_CLOSED"])]
        mean_return = closed["pnl_pct"].mean() if not closed.empty else 0.0
        entry_dates = pd.to_datetime(df["entry_date"], errors="coerce").dropna()
        window_start = str(entry_dates.min().date()) if not entry_dates.empty else ""
        window_end   = str(entry_dates.max().date()) if not entry_dates.empty else ""
        return {
            "data_window_start": window_start,
            "data_window_end":   window_end,
            "mean_return_per_trade": round(mean_return, 4),
        }

    # ── logging ──────────────────────────────────────────────
    def _log(self, m: Dict):
        logger.info("=" * 60)
        logger.info(f"[Metrics] Window: {m.get('window', '')}")
        for k, v in m.items():
            if k != "window":
                logger.info(f"  {k:<28s}: {v}")
        logger.info("=" * 60)
