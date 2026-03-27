"""
Metrics Module - Comprehensive trading performance metrics.
Sharpe, Sortino, Calmar, Max DD, Win Rate, Alpha, CAGR, and more.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02   # annualized, configurable


class MetricsModule:
    """
    Calculates all standard trading performance metrics from
    a trade list and equity curve.
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE,
                 trading_days: int = TRADING_DAYS):
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    # ── Core helpers ──────────────────────────
    def _daily_rf(self) -> float:
        return (1 + self.risk_free_rate) ** (1 / self.trading_days) - 1

    def _cagr(self, equity: pd.Series) -> float:
        if len(equity) < 2:
            return 0.0
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        years = len(equity) / self.trading_days
        if years <= 0:
            return 0.0
        return (1 + total_return) ** (1 / years) - 1

    def _max_drawdown(self, equity: pd.Series) -> float:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return float(drawdown.min())

    def _sharpe(self, returns: pd.Series) -> float:
        excess = returns - self._daily_rf()
        if excess.std() == 0:
            return 0.0
        return float((excess.mean() / excess.std()) * np.sqrt(self.trading_days))

    def _sortino(self, returns: pd.Series) -> float:
        excess = returns - self._daily_rf()
        downside = excess[excess < 0]
        if len(downside) == 0 or downside.std() == 0:
            return float("inf") if excess.mean() > 0 else 0.0
        return float((excess.mean() / downside.std()) * np.sqrt(self.trading_days))

    def _calmar(self, equity: pd.Series) -> float:
        cagr = self._cagr(equity)
        mdd = abs(self._max_drawdown(equity))
        if mdd == 0:
            return float("inf") if cagr > 0 else 0.0
        return cagr / mdd

    def _profit_factor(self, trade_df: pd.DataFrame) -> float:
        gross_profit = trade_df.loc[trade_df["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trade_df.loc[trade_df["pnl"] <= 0, "pnl"].sum())
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def _expectancy(self, trade_df: pd.DataFrame) -> float:
        """Average expected P&L per trade."""
        return float(trade_df["pnl"].mean())

    def _alpha(self, equity: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        """Simple alpha vs benchmark (buy & hold). If no benchmark, vs risk-free."""
        if benchmark_returns is None or len(benchmark_returns) < 2:
            return self._cagr(equity) - self.risk_free_rate

        strategy_returns = equity.pct_change().dropna()
        bench = benchmark_returns.reindex(strategy_returns.index).fillna(0)
        # OLS alpha annualized
        cov = np.cov(strategy_returns, bench)
        if cov[1, 1] == 0:
            return 0.0
        beta = cov[0, 1] / cov[1, 1]
        alpha_daily = strategy_returns.mean() - beta * bench.mean()
        return float(alpha_daily * self.trading_days)

    # ── Run Analysis ──────────────────────────
    def _run_analysis(self, equity: pd.Series, start_date=None, end_date=None) -> Dict:
        """Analyze consecutive win/loss runs."""
        daily_ret = equity.pct_change().dropna()

        # Build run-length encoding
        runs = []
        current_sign = None
        current_len = 0
        current_sum = 0.0

        for r in daily_ret:
            s = 1 if r >= 0 else -1
            if s == current_sign:
                current_len += 1
                current_sum += r
            else:
                if current_sign is not None:
                    runs.append({"sign": current_sign, "length": current_len, "total_return": current_sum})
                current_sign = s
                current_len = 1
                current_sum = r
        if current_sign is not None:
            runs.append({"sign": current_sign, "length": current_len, "total_return": current_sum})

        run_df = pd.DataFrame(runs)
        pos_runs = run_df[run_df["sign"] == 1]
        neg_runs = run_df[run_df["sign"] == -1]

        return {
            "data_window": f"{start_date} → {end_date}",
            "mean_daily_return": float(daily_ret.mean()),
            "mean_daily_return_pct": f"{daily_ret.mean():.4%}",
            "total_days": len(daily_ret),
            "positive_days": int((daily_ret > 0).sum()),
            "negative_days": int((daily_ret <= 0).sum()),
            "max_win_streak": int(pos_runs["length"].max()) if len(pos_runs) > 0 else 0,
            "max_loss_streak": int(neg_runs["length"].max()) if len(neg_runs) > 0 else 0,
            "avg_win_streak": float(pos_runs["length"].mean()) if len(pos_runs) > 0 else 0,
            "avg_loss_streak": float(neg_runs["length"].mean()) if len(neg_runs) > 0 else 0,
        }

    # ── Main Compute ──────────────────────────
    def compute(
        self,
        trade_df: pd.DataFrame,
        equity: pd.Series,
        initial_capital: float,
        benchmark_returns: Optional[pd.Series] = None,
        start_date=None,
        end_date=None,
    ) -> Dict[str, Any]:
        """
        Compute all metrics.

        Parameters:
            trade_df         : DataFrame of closed trades (from TradeSimulator)
            equity           : Portfolio equity curve (pd.Series)
            initial_capital  : Starting capital
            benchmark_returns: Optional buy-and-hold returns for alpha calculation
            start_date       : Test window start (for labeling)
            end_date         : Test window end (for labeling)

        Returns:
            dict of all metrics
        """
        if trade_df is None or len(trade_df) == 0:
            logger.warning("[Metrics] No trades to compute metrics from.")
            return self._empty_metrics(initial_capital)

        equity_clean = equity.dropna()
        daily_returns = equity_clean.pct_change().dropna()

        n_trades = len(trade_df)
        wins = (trade_df["pnl"] > 0).sum()
        losses = (trade_df["pnl"] <= 0).sum()
        win_rate = wins / n_trades if n_trades > 0 else 0

        total_return = (equity_clean.iloc[-1] / initial_capital - 1) if len(equity_clean) > 0 else 0
        max_cap_used = float(trade_df["capital_used"].sum()) if "capital_used" in trade_df.columns else 0

        mdd = self._max_drawdown(equity_clean)
        cagr = self._cagr(equity_clean)
        sharpe = self._sharpe(daily_returns)
        sortino = self._sortino(daily_returns)
        calmar = self._calmar(equity_clean)
        pf = self._profit_factor(trade_df)
        exp = self._expectancy(trade_df)
        alpha = self._alpha(equity_clean, benchmark_returns)
        runs = self._run_analysis(equity_clean, start_date, end_date)

        metrics = {
            # Returns
            "total_return_pct": total_return,
            "cagr": cagr,
            "mean_trade_return": float(trade_df["returns_pct"].mean()),

            # Risk-adjusted
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "alpha": alpha,

            # Drawdown
            "max_drawdown": mdd,
            "max_drawdown_pct": f"{mdd:.2%}",

            # Trade stats
            "n_trades": n_trades,
            "n_wins": int(wins),
            "n_losses": int(losses),
            "win_rate": win_rate,
            "profit_factor": pf,
            "expectancy_per_trade": exp,
            "avg_win": float(trade_df.loc[trade_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0,
            "avg_loss": float(trade_df.loc[trade_df["pnl"] <= 0, "pnl"].mean()) if losses > 0 else 0,

            # Capital
            "initial_capital": initial_capital,
            "final_equity": float(equity_clean.iloc[-1]) if len(equity_clean) > 0 else initial_capital,
            "max_capital_used": max_cap_used,

            # Run analysis
            **runs,
        }

        self._log_metrics(metrics)
        return metrics

    def _empty_metrics(self, capital: float) -> Dict:
        return {
            "n_trades": 0, "sharpe_ratio": 0, "sortino_ratio": 0,
            "calmar_ratio": 0, "max_drawdown": 0, "win_rate": 0,
            "cagr": 0, "alpha": 0, "profit_factor": 0, "expectancy_per_trade": 0,
            "initial_capital": capital, "final_equity": capital,
            "total_return_pct": 0, "max_capital_used": 0,
        }

    def _log_metrics(self, m: Dict):
        logger.info("=" * 60)
        logger.info("[Metrics] PERFORMANCE SUMMARY")
        logger.info(f"  Total Return   : {m['total_return_pct']:+.2%}")
        logger.info(f"  CAGR           : {m['cagr']:+.2%}")
        logger.info(f"  Sharpe Ratio   : {m['sharpe_ratio']:.3f}")
        logger.info(f"  Sortino Ratio  : {m['sortino_ratio']:.3f}")
        logger.info(f"  Calmar Ratio   : {m['calmar_ratio']:.3f}")
        logger.info(f"  Alpha          : {m['alpha']:+.4f}")
        logger.info(f"  Max Drawdown   : {m['max_drawdown']:.2%}")
        logger.info(f"  Win Rate       : {m['win_rate']:.1%}")
        logger.info(f"  Profit Factor  : {m['profit_factor']:.2f}")
        logger.info(f"  Expectancy     : ${m['expectancy_per_trade']:+.2f}/trade")
        logger.info(f"  N Trades       : {m['n_trades']}")
        logger.info(f"  Max Cap Used   : ${m['max_capital_used']:,.2f}")
        logger.info(f"  Mean Return    : {m.get('mean_daily_return_pct', 'N/A')}")
        logger.info("=" * 60)
