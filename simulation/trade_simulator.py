"""
simulation/trade_simulator.py
Trade Simulation Module.

Handles:
  - Capital allocation (total, per-trade max)
  - Commissions and slippage
  - Confidence-based position sizing
  - FIFO trade matching (Entry → Exit)
  - Trade journal with full summary
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from logger.trading_logger import TradeJournal

logger = logging.getLogger("trading.simulator")

SIGNAL_ENTRY = 1
SIGNAL_EXIT  = -1


@dataclass
class Trade:
    trade_id:     int
    ticker:       str
    entry_date:   str
    entry_price:  float
    shares:       float
    capital_used: float
    commission:   float
    slippage:     float
    confidence:   float
    regime:       str
    exit_date:    Optional[str]   = None
    exit_price:   Optional[float] = None
    exit_commission: float        = 0.0
    exit_slippage:   float        = 0.0
    pnl:          float           = 0.0
    pnl_pct:      float           = 0.0
    status:       str             = "OPEN"   # OPEN | CLOSED


class TradeSimulator:
    """
    Simulates trades from a signal-annotated DataFrame.
    Returns a trade summary DataFrame and the equity curve.
    """

    def __init__(self, config, ticker: str = "ASSET",
                 run_id: str = "run", output_dir: str = "backtest_results"):
        self.cfg    = config.sim
        self.ticker = ticker
        self._trades: List[Trade] = []
        self._open_trades: List[Trade] = []
        self._trade_counter = 0
        self._cash = self.cfg.total_capital
        self._journal = TradeJournal(output_dir=output_dir, run_id=run_id)
        self._run_id = run_id

    # ── helpers ──────────────────────────────────────────────
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = price * self.cfg.slippage_pct
        return price + slip if is_buy else price - slip

    def _apply_commission(self, capital: float) -> float:
        return capital * self.cfg.commission

    def _position_size(self, price: float, confidence: float) -> float:
        """Size position between 50%–100% of per-trade max based on confidence."""
        if self.cfg.confidence_sizing:
            scale = 0.5 + 0.5 * confidence
        else:
            scale = 1.0
        capital = min(self.cfg.per_trade_max_capital * scale, self._cash * 0.95)
        shares = capital / price
        return max(shares, 0)

    # ── simulation ───────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through signal df and simulate trades.
        Returns the original df augmented with equity curve columns.
        """
        logger.info(f"[TradeSimulator] Starting simulation on {len(df)} bars …")
        self._cash = self.cfg.total_capital
        self._trades = []
        self._open_trades = []
        self._trade_counter = 0

        equity_curve = []

        for date, row in df.iterrows():
            sig       = int(row.get("signal", 0))
            price     = float(row["Close"])
            regime    = str(row.get("hmm_regime", "Unknown"))
            confidence= float(row.get("xgb_confidence", 0.5))
            vol       = float(row.get("garch_vol_forecast", row.get("volatility_20d", 0.02)))

            # ── ENTRY ─────────────────────────────────────────
            if sig == SIGNAL_ENTRY:
                exec_price = self._apply_slippage(price, is_buy=True)
                shares = self._position_size(exec_price, confidence)
                if shares <= 0:
                    equity_curve.append(self._portfolio_value(price))
                    continue
                capital_used = shares * exec_price
                commission   = self._apply_commission(capital_used)
                total_cost   = capital_used + commission
                slippage_cost= capital_used - shares * price

                if total_cost > self._cash:
                    logger.debug(f"  {date} ENTRY skipped – insufficient cash "
                                 f"(need {total_cost:.2f}, have {self._cash:.2f})")
                    equity_curve.append(self._portfolio_value(price))
                    continue

                self._cash -= total_cost
                self._trade_counter += 1
                t = Trade(
                    trade_id    = self._trade_counter,
                    ticker      = self.ticker,
                    entry_date  = str(date.date()),
                    entry_price = exec_price,
                    shares      = shares,
                    capital_used= capital_used,
                    commission  = commission,
                    slippage    = slippage_cost,
                    confidence  = confidence,
                    regime      = regime,
                )
                self._open_trades.append(t)
                self._trades.append(t)

                self._journal.log("ENTRY", date=t.entry_date, ticker=self.ticker,
                                  direction="LONG", price=exec_price, shares=shares,
                                  capital=capital_used, commission=commission,
                                  slippage=slippage_cost, confidence=confidence,
                                  regime=regime, pnl="", cum_pnl="")
                logger.debug(f"  {date} ENTRY  px={exec_price:.2f}  shares={shares:.2f}  "
                             f"capital={capital_used:.2f}  regime={regime}")

            # ── EXIT ──────────────────────────────────────────
            elif sig == SIGNAL_EXIT and self._open_trades:
                # FIFO: close oldest open trade
                t = self._open_trades.pop(0)
                exec_price     = self._apply_slippage(price, is_buy=False)
                proceeds       = t.shares * exec_price
                commission_ex  = self._apply_commission(proceeds)
                slippage_ex    = t.shares * (price - exec_price)
                net_proceeds   = proceeds - commission_ex

                self._cash    += net_proceeds
                gross_pnl      = proceeds - t.capital_used
                net_pnl        = gross_pnl - t.commission - commission_ex
                pnl_pct        = net_pnl / t.capital_used if t.capital_used else 0.0

                t.exit_date        = str(date.date())
                t.exit_price       = exec_price
                t.exit_commission  = commission_ex
                t.exit_slippage    = slippage_ex
                t.pnl              = net_pnl
                t.pnl_pct          = pnl_pct
                t.status           = "CLOSED"

                cum_pnl = sum(tr.pnl for tr in self._trades if tr.status == "CLOSED")
                self._journal.log("EXIT", date=t.exit_date, ticker=self.ticker,
                                  direction="LONG", price=exec_price, shares=t.shares,
                                  capital=proceeds, commission=commission_ex,
                                  slippage=slippage_ex, confidence=t.confidence,
                                  regime=regime, pnl=net_pnl, cum_pnl=cum_pnl)
                logger.debug(f"  {date} EXIT  px={exec_price:.2f}  pnl={net_pnl:.2f}  "
                             f"pnl%={pnl_pct:.2%}")

            equity_curve.append(self._portfolio_value(price))

        # Force-close any still-open trades at last price
        last_price = float(df["Close"].iloc[-1])
        for t in list(self._open_trades):
            t.exit_date  = str(df.index[-1].date())
            t.exit_price = last_price
            proceeds     = t.shares * last_price
            net_pnl      = proceeds - t.capital_used - t.commission
            t.pnl        = net_pnl
            t.pnl_pct    = net_pnl / t.capital_used if t.capital_used else 0.0
            t.status     = "FORCE_CLOSED"
            self._cash  += proceeds

        self._open_trades.clear()
        self._journal.close()

        df = df.copy()
        df["equity_curve"] = equity_curve
        df["drawdown"]     = self._compute_drawdown(equity_curve)

        logger.info(f"[TradeSimulator] Simulation complete. "
                    f"Trades={len(self._trades)}  "
                    f"Final equity={equity_curve[-1]:,.2f}")
        return df

    def _portfolio_value(self, current_price: float) -> float:
        open_value = sum(t.shares * current_price for t in self._open_trades)
        return self._cash + open_value

    def _compute_drawdown(self, equity: list) -> list:
        arr = np.array(equity, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / np.where(peak == 0, 1, peak)
        return dd.tolist()

    # ── summary ──────────────────────────────────────────────
    def trade_summary(self) -> pd.DataFrame:
        """Return a DataFrame with one row per trade."""
        if not self._trades:
            return pd.DataFrame()
        rows = []
        cum_pnl = 0.0
        for t in self._trades:
            cum_pnl += t.pnl
            row = asdict(t)
            row["cum_pnl"] = cum_pnl
            rows.append(row)
        return pd.DataFrame(rows)
