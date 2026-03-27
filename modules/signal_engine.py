"""
Trading Signal & Strategy Engine
- Plug-and-play indicator architecture
- 7 default indicators with voting system
- Signal generated only when regime is bullish AND ≥ min_votes indicators agree
- States: ENTRY, EXIT, IDLE
- Max 2 simultaneous open trades
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Signal Constants
# ─────────────────────────────────────────────
SIGNAL_IDLE  = 0
SIGNAL_ENTRY = 1
SIGNAL_EXIT  = -1


# ─────────────────────────────────────────────
# Indicator Base + Registry
# ─────────────────────────────────────────────
class Indicator(ABC):
    """Abstract base for all strategy indicators."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series: True = bullish condition met."""
        pass


class IndicatorRegistry:
    """Plug-and-play indicator registry."""
    _registry: Dict[str, Indicator] = {}

    @classmethod
    def register(cls, indicator: Indicator):
        cls._registry[indicator.name] = indicator
        logger.debug(f"[Indicators] Registered: {indicator.name}")

    @classmethod
    def get(cls, name: str) -> Indicator:
        if name not in cls._registry:
            raise KeyError(f"Indicator '{name}' not registered.")
        return cls._registry[name]

    @classmethod
    def available(cls) -> List[str]:
        return list(cls._registry.keys())


# ─────────────────────────────────────────────
# Default Indicators (7)
# ─────────────────────────────────────────────
class MomentumIndicator(Indicator):
    """Momentum > threshold (default 1%)."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    @property
    def name(self): return "momentum"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = "momentum_10" if "momentum_10" in df.columns else "returns"
        return df[col].fillna(0) > self.threshold


class VolumeIndicator(Indicator):
    """Volume > 20-period SMA of volume."""
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    @property
    def name(self): return "volume"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "volume_ratio" in df.columns:
            return df["volume_ratio"].fillna(1) > self.multiplier
        vol_sma = df["Volume"].rolling(20).mean()
        return df["Volume"] > vol_sma * self.multiplier


class VolatilityFilterIndicator(Indicator):
    """Volatility < threshold (default 10% annualized)."""
    def __init__(self, max_vol: float = 0.10):
        self.max_vol = max_vol

    @property
    def name(self): return "low_volatility"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        col = "garch_vol" if "garch_vol" in df.columns else "volatility_20"
        return df[col].fillna(0.2) < self.max_vol


class ADXIndicator(Indicator):
    """ADX > threshold (default 25) → trending market."""
    def __init__(self, threshold: float = 25.0):
        self.threshold = threshold

    @property
    def name(self): return "adx"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "adx" not in df.columns:
            logger.warning("[ADX] adx column missing.")
            return pd.Series(False, index=df.index)
        return df["adx"].fillna(0) > self.threshold


class PriceAboveEMAIndicator(Indicator):
    """Price > 50-period EMA."""
    def __init__(self, ema_period: int = 50):
        self.ema_period = ema_period

    @property
    def name(self): return "price_above_ema"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "ema_50" not in df.columns:
            ema = df["Close"].ewm(span=self.ema_period, adjust=False).mean()
        else:
            ema = df["ema_50"]
        return df["Close"] > ema


class RSIIndicator(Indicator):
    """RSI in bullish zone: rsi_low < RSI < rsi_high (default 60–90)."""
    def __init__(self, rsi_low: float = 60.0, rsi_high: float = 90.0):
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high

    @property
    def name(self): return "rsi"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "rsi_14" not in df.columns:
            logger.warning("[RSI] rsi_14 column missing.")
            return pd.Series(False, index=df.index)
        rsi = df["rsi_14"].fillna(50)
        return (rsi > self.rsi_low) & (rsi < self.rsi_high)


class MACDIndicator(Indicator):
    """MACD > Signal line (bullish crossover zone)."""
    def __init__(self):
        pass

    @property
    def name(self): return "macd"

    def compute(self, df: pd.DataFrame) -> pd.Series:
        if "macd" not in df.columns or "macd_signal" not in df.columns:
            logger.warning("[MACD] macd/macd_signal columns missing.")
            return pd.Series(False, index=df.index)
        return df["macd"].fillna(0) > df["macd_signal"].fillna(0)


# ─── Register all defaults ────────────────────
IndicatorRegistry.register(MomentumIndicator())
IndicatorRegistry.register(VolumeIndicator())
IndicatorRegistry.register(VolatilityFilterIndicator())
IndicatorRegistry.register(ADXIndicator())
IndicatorRegistry.register(PriceAboveEMAIndicator())
IndicatorRegistry.register(RSIIndicator())
IndicatorRegistry.register(MACDIndicator())


# ─────────────────────────────────────────────
# Signal Engine
# ─────────────────────────────────────────────
class SignalEngine:
    """
    Generates trade signals using a voting system.

    Rules:
      - Signal = ENTRY only if:
        1. Current regime is bullish (regime_is_bullish == True)
        2. XGBoost predicts up (xgb_pred == 1) [optional]
        3. At least min_votes out of n indicators are bullish
      - Signal = EXIT when:
        - Regime turns non-bullish, OR
        - Votes drop below exit_votes threshold
      - Max simultaneous open trades enforced externally by simulator.
    """

    def __init__(
        self,
        indicator_names: Optional[List[str]] = None,
        min_votes: int = 5,
        exit_votes: int = 3,
        use_xgb_filter: bool = True,
        max_open_trades: int = 2,
    ):
        self.indicator_names = indicator_names or IndicatorRegistry.available()
        self.min_votes = min_votes
        self.exit_votes = exit_votes
        self.use_xgb_filter = use_xgb_filter
        self.max_open_trades = max_open_trades

        logger.info(f"[SignalEngine] Indicators: {self.indicator_names}")
        logger.info(f"[SignalEngine] min_votes={min_votes}, exit_votes={exit_votes}, "
                    f"xgb_filter={use_xgb_filter}, max_open={max_open_trades}")

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for the entire DataFrame.

        Returns df with new columns:
          - ind_{name}     : bool, each indicator vote
          - vote_count     : int, total bullish votes
          - signal         : int, ENTRY=1, EXIT=-1, IDLE=0
          - signal_label   : str, "ENTRY" / "EXIT" / "IDLE"
          - open_trades    : int, running count of open positions
        """
        out = df.copy()

        # Compute all indicator votes
        vote_cols = []
        for iname in self.indicator_names:
            try:
                ind = IndicatorRegistry.get(iname)
                col = f"ind_{iname}"
                out[col] = ind.compute(out).astype(int)
                vote_cols.append(col)
            except KeyError:
                logger.warning(f"[SignalEngine] Indicator not found: {iname}")

        out["vote_count"] = out[vote_cols].sum(axis=1)

        # Regime filter
        regime_bullish = out.get("regime_is_bullish", pd.Series(True, index=out.index))

        # XGBoost filter
        if self.use_xgb_filter and "xgb_pred" in out.columns:
            xgb_up = (out["xgb_pred"] == 1)
        else:
            xgb_up = pd.Series(True, index=out.index)

        # Confidence from XGBoost
        if "xgb_confidence" in out.columns:
            confidence = out["xgb_confidence"]
        else:
            confidence = pd.Series(0.5, index=out.index)

        # Build raw signal
        entry_cond = regime_bullish & xgb_up & (out["vote_count"] >= self.min_votes)
        exit_cond = (~regime_bullish) | (out["vote_count"] < self.exit_votes)

        # Stateful signal with open trade tracking
        signals = []
        open_trades = 0
        in_trade = False

        for i in range(len(out)):
            ec = bool(entry_cond.iloc[i])
            xc = bool(exit_cond.iloc[i])

            if in_trade and xc:
                sig = SIGNAL_EXIT
                in_trade = False
                open_trades = max(0, open_trades - 1)
            elif (not in_trade) and ec and (open_trades < self.max_open_trades):
                sig = SIGNAL_ENTRY
                in_trade = True
                open_trades += 1
            elif in_trade and (not xc):
                sig = SIGNAL_IDLE   # hold
            else:
                sig = SIGNAL_IDLE

            signals.append(sig)

        out["signal"] = signals
        out["signal_label"] = out["signal"].map(
            {SIGNAL_ENTRY: "ENTRY", SIGNAL_EXIT: "EXIT", SIGNAL_IDLE: "IDLE"}
        )
        out["open_trade_count"] = out["signal"].eq(SIGNAL_ENTRY).cumsum() - out["signal"].eq(SIGNAL_EXIT).cumsum()
        out["xgb_confidence_at_signal"] = confidence.where(out["signal"] != SIGNAL_IDLE, np.nan)

        # Log summary
        n_entry = (out["signal"] == SIGNAL_ENTRY).sum()
        n_exit = (out["signal"] == SIGNAL_EXIT).sum()
        logger.info(
            f"[SignalEngine] Generated {n_entry} ENTRY, {n_exit} EXIT signals "
            f"over {len(out)} bars"
        )

        # Detailed signal log
        entries = out[out["signal"] == SIGNAL_ENTRY][["Close", "vote_count", "regime_label"
                                                       if "regime_label" in out.columns else "vote_count"]]
        for idx, row in entries.iterrows():
            logger.debug(
                f"[ENTRY] {idx.date()} | Close={row['Close']:.2f} | "
                f"votes={int(row['vote_count'])}"
            )

        return out

    def add_indicator(self, indicator: Indicator):
        """Plug in a new indicator at runtime."""
        IndicatorRegistry.register(indicator)
        if indicator.name not in self.indicator_names:
            self.indicator_names.append(indicator.name)
        logger.info(f"[SignalEngine] Added indicator: {indicator.name}")
