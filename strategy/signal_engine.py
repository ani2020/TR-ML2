"""
strategy/signal_engine.py
Trading Signal & Strategy Engine.

Indicator voting system:
  7 indicators → signal only when regime is Bullish AND ≥ min_votes agree.
  Entry / Exit / Idle states.  Max 2 simultaneous open trades.

Plug-and-play: add new indicators by subclassing BaseIndicator.
"""

import abc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("trading.signal")

# ── Signal constants ──────────────────────────────────────────
SIGNAL_ENTRY  = 1
SIGNAL_EXIT   = -1
SIGNAL_IDLE   = 0

BULLISH_REGIMES = {"Bull Run", "Bull"}


# ══════════════════════════════════════════════════════════════
# BASE INDICATOR  (plug-in interface)
# ══════════════════════════════════════════════════════════════
class BaseIndicator(abc.ABC):
    """
    Subclass and implement `vote(row) -> bool`.
    row is a pd.Series representing one trading day.
    """

    @abc.abstractmethod
    def vote(self, row: pd.Series) -> bool:
        """Return True if the indicator votes FOR a long entry."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...


# ══════════════════════════════════════════════════════════════
# BUILT-IN INDICATORS  (7 default)
# ══════════════════════════════════════════════════════════════
class MomentumIndicator(BaseIndicator):
    """Momentum > threshold (default 1%)."""
    def __init__(self, threshold: float = 0.01, period: str = "momentum_10d"):
        self.threshold = threshold
        self.period = period

    @property
    def name(self): return "Momentum"

    def vote(self, row: pd.Series) -> bool:
        val = row.get(self.period, np.nan)
        return bool(not np.isnan(val) and val > self.threshold)


class VolumeIndicator(BaseIndicator):
    """Volume > 20-period SMA."""
    @property
    def name(self): return "Volume>SMA20"

    def vote(self, row: pd.Series) -> bool:
        vol = row.get("Volume", np.nan)
        sma = row.get("volume_sma_20", np.nan)
        return bool(not np.isnan(vol) and not np.isnan(sma) and vol > sma)


class VolatilityIndicator(BaseIndicator):
    """GARCH forecast volatility < max_vol (default 10%)."""
    def __init__(self, max_vol: float = 0.10):
        self.max_vol = max_vol

    @property
    def name(self): return "VolLow"

    def vote(self, row: pd.Series) -> bool:
        gv = row.get("garch_vol_forecast", np.nan)
        if np.isnan(gv):
            gv = row.get("volatility_20d", np.nan)
        return bool(not np.isnan(gv) and gv < self.max_vol)


class ADXIndicator(BaseIndicator):
    """ADX > 25 (strong trend)."""
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold

    @property
    def name(self): return "ADX>25"

    def vote(self, row: pd.Series) -> bool:
        adx = row.get("adx_14", np.nan)
        return bool(not np.isnan(adx) and adx > self.threshold)


class EMAIndicator(BaseIndicator):
    """Price > 50-period EMA."""
    def __init__(self, ema_col: str = "ema_50"):
        self.ema_col = ema_col

    @property
    def name(self): return "Price>EMA50"

    def vote(self, row: pd.Series) -> bool:
        close = row.get("Close", np.nan)
        ema = row.get(self.ema_col, np.nan)
        return bool(not np.isnan(close) and not np.isnan(ema) and close > ema)


class RSIIndicator(BaseIndicator):
    """RSI > rsi_min AND RSI < rsi_max."""
    def __init__(self, rsi_min: float = 60.0, rsi_max: float = 90.0):
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max

    @property
    def name(self): return "RSI(60-90)"

    def vote(self, row: pd.Series) -> bool:
        rsi = row.get("rsi_14", np.nan)
        return bool(not np.isnan(rsi) and self.rsi_min < rsi < self.rsi_max)


class MACDIndicator(BaseIndicator):
    """MACD line > Signal line."""
    @property
    def name(self): return "MACD>Signal"

    def vote(self, row: pd.Series) -> bool:
        macd = row.get("macd", np.nan)
        sig  = row.get("macd_signal", np.nan)
        return bool(not np.isnan(macd) and not np.isnan(sig) and macd > sig)


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════
class SignalEngine:
    """
    Combines indicator votes with regime filter to produce trade signals.

    Output columns added to df:
        vote_count      – number of bullish indicator votes
        vote_details    – dict of {indicator_name: bool}
        signal          – SIGNAL_ENTRY / SIGNAL_EXIT / SIGNAL_IDLE
        signal_label    – "Entry" / "Exit" / "Idle"
        open_trades     – simulated open trade count at that bar
    """

    def __init__(self, config):
        self.cfg = config.signal
        self._indicators: List[BaseIndicator] = []
        self._register_defaults(config)

    def _register_defaults(self, config):
        cfg = config.signal
        self.register(MomentumIndicator(threshold=cfg.momentum_threshold))
        self.register(VolumeIndicator())
        self.register(VolatilityIndicator(max_vol=cfg.volatility_max))
        self.register(ADXIndicator(threshold=cfg.adx_min))
        self.register(EMAIndicator(f"ema_{cfg.ema_period}"))
        self.register(RSIIndicator(rsi_min=cfg.rsi_min, rsi_max=cfg.rsi_max))
        self.register(MACDIndicator())

    def register(self, indicator: BaseIndicator):
        """Plug in a new indicator."""
        self._indicators.append(indicator)
        logger.debug(f"[SignalEngine] Registered indicator: {indicator.name}")

    # ── core signal logic ────────────────────────────────────
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"[SignalEngine] Generating signals on {len(df)} rows …")
        df = df.copy()

        vote_counts  = []
        vote_details = []
        signals      = []
        open_trades  = 0

        for _, row in df.iterrows():
            regime = row.get("hmm_regime", "Unknown")
            is_bullish = regime in BULLISH_REGIMES
            xgb_up = row.get("xgb_direction", 1) == 1   # XGBoost also bullish

            # Collect votes from all indicators
            votes = {}
            for ind in self._indicators:
                try:
                    votes[ind.name] = ind.vote(row)
                except Exception as e:
                    votes[ind.name] = False
                    logger.debug(f"  Indicator {ind.name} raised: {e}")

            n_votes = sum(votes.values())
            vote_counts.append(n_votes)
            vote_details.append(votes)

            # ── Signal decision ───────────────────────────────
            # ENTRY: regime bullish + XGBoost up + ≥ min_votes
            if (is_bullish and xgb_up
                    and n_votes >= self.cfg.min_votes
                    and open_trades < self.cfg.max_open_trades):
                sig = SIGNAL_ENTRY
                open_trades += 1

            # EXIT: regime turns bearish or model flips down
            elif open_trades > 0 and (not is_bullish or not xgb_up):
                sig = SIGNAL_EXIT
                open_trades = max(0, open_trades - 1)

            else:
                sig = SIGNAL_IDLE

            signals.append(sig)

        df["vote_count"]   = vote_counts
        df["vote_details"] = [str(v) for v in vote_details]
        df["signal"]       = signals
        df["signal_label"] = df["signal"].map({
            SIGNAL_ENTRY: "Entry",
            SIGNAL_EXIT:  "Exit",
            SIGNAL_IDLE:  "Idle",
        })

        n_entry = (df["signal"] == SIGNAL_ENTRY).sum()
        n_exit  = (df["signal"] == SIGNAL_EXIT).sum()
        logger.info(f"[SignalEngine] Entry={n_entry}  Exit={n_exit}  "
                    f"Idle={len(df)-n_entry-n_exit}")
        return df
