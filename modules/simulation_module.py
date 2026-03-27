"""
Trade Simulation Module
- Simulates trade execution from signal DataFrame
- Handles capital allocation, slippage, expenses
- Position sizing based on XGBoost confidence
- Logs every trade; produces trade summary
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

SIGNAL_ENTRY = 1
SIGNAL_EXIT  = -1
SIGNAL_IDLE  = 0


@dataclass
class Trade:
    trade_id: int
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    capital_used: float = 0.0
    regime_at_entry: str = "Unknown"
    confidence: float = 0.5
    vote_count: int = 0
    pnl: float = 0.0
    returns_pct: float = 0.0
    cumulative_return: float = 0.0
    is_closed: bool = False
    exit_reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class SimulationConfig:
    total_capital: float = 100_000.0
    max_capital_per_trade: float = 10_000.0
    trading_expense: float = 10.0     # $ flat fee per trade
    slippage_pct: float = 0.001       # 0.1% slippage
    confidence_sizing: bool = True    # scale position by XGB confidence
    min_confidence: float = 0.1       # minimum confidence to trade


class TradeSimulator:
    """
    Simulates trades from a signal DataFrame.

    Rules:
      - ENTRY: open a new position (up to max_open_trades)
      - EXIT: close oldest open position
      - Position size = max_capital_per_trade * confidence (if confidence_sizing)
      - Slippage applied to entry and exit prices
      - Expenses deducted per trade (entry + exit)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.trades: List[Trade] = []
        self._open_trades: List[Trade] = []
        self._trade_counter = 0
        self._cash = self.cfg.total_capital
        self._equity_curve: List[float] = []
        self._running_capital_used = 0.0
        self._max_capital_used = 0.0

        logger.info(
            f"[Simulator] Config: capital=${self.cfg.total_capital:,.0f}, "
            f"per_trade_max=${self.cfg.max_capital_per_trade:,.0f}, "
            f"expense=${self.cfg.trading_expense:.2f}, slippage={self.cfg.slippage_pct:.3%}"
        )

    def _apply_slippage(self, price: float, direction: str) -> float:
        """Apply slippage: buy at higher price, sell at lower."""
        if direction == "buy":
            return price * (1 + self.cfg.slippage_pct)
        return price * (1 - self.cfg.slippage_pct)

    def _position_size(self, confidence: float) -> float:
        """Calculate capital to deploy based on confidence."""
        if not self.cfg.confidence_sizing:
            return self.cfg.max_capital_per_trade
        scaled = self.cfg.max_capital_per_trade * max(confidence, self.cfg.min_confidence)
        return min(scaled, self.cfg.max_capital_per_trade)

    def simulate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run simulation over signal DataFrame.
        Returns DataFrame with trade P&L columns and equity curve.
        """
        self.trades = []
        self._open_trades = []
        self._trade_counter = 0
        self._cash = self.cfg.total_capital
        self._equity_curve = []

        out = df.copy()
        out["equity"] = np.nan
        out["trade_pnl"] = 0.0
        out["cumulative_pnl"] = 0.0

        for i, (idx, row) in enumerate(df.iterrows()):
            signal = int(row.get("signal", SIGNAL_IDLE))
            price = float(row["Close"])
            confidence = float(row.get("xgb_confidence_at_signal", 0.5))
            if pd.isna(confidence):
                confidence = 0.5
            regime = str(row.get("regime_label", "Unknown"))
            votes = int(row.get("vote_count", 0))

            # ── ENTRY ──────────────────────────
            if signal == SIGNAL_ENTRY and confidence >= self.cfg.min_confidence:
                entry_price = self._apply_slippage(price, "buy")
                capital = self._position_size(confidence)

                if capital > self._cash:
                    logger.debug(f"[Sim] Insufficient cash on {idx.date()} — skipping entry")
                else:
                    shares = (capital - self.cfg.trading_expense) / entry_price
                    self._cash -= capital
                    self._running_capital_used += capital
                    self._max_capital_used = max(self._max_capital_used, self._running_capital_used)
                    self._trade_counter += 1

                    trade = Trade(
                        trade_id=self._trade_counter,
                        entry_date=str(idx.date()),
                        entry_price=entry_price,
                        shares=shares,
                        capital_used=capital,
                        regime_at_entry=regime,
                        confidence=confidence,
                        vote_count=votes,
                    )
                    self._open_trades.append(trade)
                    logger.info(
                        f"[ENTRY] #{self._trade_counter} | {idx.date()} | "
                        f"Price={entry_price:.2f} | Shares={shares:.2f} | "
                        f"Capital=${capital:.2f} | Regime={regime} | "
                        f"Confidence={confidence:.2%} | Votes={votes}"
                    )

            # ── EXIT ───────────────────────────
            elif signal == SIGNAL_EXIT and self._open_trades:
                exit_price = self._apply_slippage(price, "sell")
                # Close oldest trade (FIFO)
                trade = self._open_trades.pop(0)
                proceeds = trade.shares * exit_price - self.cfg.trading_expense
                pnl = proceeds - trade.capital_used
                ret_pct = pnl / trade.capital_used

                self._cash += proceeds
                self._running_capital_used -= trade.capital_used

                trade.exit_date = str(idx.date())
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.returns_pct = ret_pct
                trade.is_closed = True
                trade.exit_reason = "Signal"

                closed_pnl = sum(t.pnl for t in self.trades if t.is_closed)
                trade.cumulative_return = (self._cash + sum(
                    t.shares * price for t in self._open_trades
                ) - self.cfg.total_capital) / self.cfg.total_capital

                self.trades.append(trade)
                out.at[idx, "trade_pnl"] = pnl

                logger.info(
                    f"[EXIT]  #{trade.trade_id} | {idx.date()} | "
                    f"Entry={trade.entry_price:.2f} Exit={exit_price:.2f} | "
                    f"PnL=${pnl:+.2f} ({ret_pct:+.2%}) | "
                    f"Regime={regime}"
                )

            # Mark-to-market equity
            open_value = sum(t.shares * price for t in self._open_trades)
            total_equity = self._cash + open_value
            self._equity_curve.append(total_equity)
            out.at[idx, "equity"] = total_equity

        # Force-close any open trades at last price
        last_price = float(df["Close"].iloc[-1])
        last_date = df.index[-1]
        for trade in self._open_trades:
            exit_price = self._apply_slippage(last_price, "sell")
            proceeds = trade.shares * exit_price - self.cfg.trading_expense
            pnl = proceeds - trade.capital_used
            trade.exit_date = str(last_date.date())
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.returns_pct = pnl / trade.capital_used
            trade.is_closed = True
            trade.exit_reason = "Force-Close"
            trade.cumulative_return = (self._cash + proceeds - self.cfg.total_capital) / self.cfg.total_capital
            self.trades.append(trade)
            logger.info(f"[FORCE-CLOSE] #{trade.trade_id} at {last_date.date()} | PnL=${pnl:+.2f}")

        out["cumulative_pnl"] = out["trade_pnl"].cumsum()

        logger.info(f"[Simulator] Done. {len(self.trades)} trades closed.")
        logger.info(f"[Simulator] Max capital deployed: ${self._max_capital_used:,.2f}")
        self._print_summary()
        return out

    def _print_summary(self):
        if not self.trades:
            logger.info("[Simulator] No trades executed.")
            return
        df = pd.DataFrame([t.to_dict() for t in self.trades])
        wins = (df["pnl"] > 0).sum()
        losses = (df["pnl"] <= 0).sum()
        total_pnl = df["pnl"].sum()
        win_rate = wins / len(df) if len(df) > 0 else 0
        logger.info("=" * 60)
        logger.info("[Simulator] TRADE SUMMARY")
        logger.info(f"  Total Trades : {len(df)}")
        logger.info(f"  Wins         : {wins} | Losses: {losses}")
        logger.info(f"  Win Rate     : {win_rate:.1%}")
        logger.info(f"  Total P&L    : ${total_pnl:+,.2f}")
        logger.info(f"  Avg Return   : {df['returns_pct'].mean():+.2%}")
        logger.info(f"  Max Capital  : ${self._max_capital_used:,.2f}")
        logger.info("=" * 60)

    def get_trade_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self._equity_curve, name="equity")

    @property
    def max_capital_used(self) -> float:
        return self._max_capital_used
