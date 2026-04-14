"""
config/settings.py
Central configuration for the ML Trading System.
All parameters can be overridden by the Grid Optimizer.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class DataConfig:
    ticker: str = "^NSEI"
    start_date: str = "2014-01-01"
    end_date: str = "2026-03-30"
    interval: str = "1d"
    cache_dir: str = "cache"
    raw_cache_file: str = "{ticker}_raw.csv"
    feature_cache_file: str = "{ticker}_features.csv"
    force_refresh: bool = False


@dataclass
class HMMConfig:
    n_states: int = 5                          # up to 7 hidden states
    n_iter: int = 200
    covariance_type: str = "full"
    random_state: int = 42
    features: List[str] = field(default_factory=lambda: [
        "log_return",
        "volatility_20d",
        "volume_ratio",
        "momentum_10d",
        "rsi_14",
    ])
    # State labels assigned after training (by mean return ranking)
    state_labels: Optional[Dict[int, str]] = None


@dataclass
class GARCHConfig:
    p: int = 1
    q: int = 1
    vol: str = "Garch"
    dist: str = "normal"
    mean: str = "Constant"


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1
    early_stopping_rounds: int = 30
    eval_metric: str = "logloss"
    # Prediction horizon in days
    horizon: int = 1
    # Target: direction (1=up, 0=down)
    target_threshold: float = 0.0


@dataclass
class SignalConfig:
    # Minimum indicators that must agree (out of 7)
    min_votes: int = 5
    total_indicators: int = 7
    # Indicator thresholds
    momentum_threshold: float = 0.01        # > 1%
    volume_sma_period: int = 20
    volatility_max: float = 0.10            # < 10%
    adx_min: float = 25.0
    ema_period: int = 50
    rsi_min: float = 60.0
    rsi_max: float = 90.0
    # For MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    max_open_trades: int = 2


@dataclass
class SimulationConfig:
    total_capital: float = 100_000.0
    per_trade_max_capital: float = 20_000.0
    commission: float = 0.001               # 0.1% per trade side
    slippage_pct: float = 0.001             # 0.1% slippage
    confidence_sizing: bool = True          # scale position by confidence


@dataclass
class BacktestConfig:
    # List of (train_start, train_end, test_start, test_end)
    windows: List[Tuple[str, str, str, str]] = field(default_factory=lambda: [
        ("2014-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
        ("2014-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("2014-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2014-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2014-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ])
    output_dir: str = "backtest_results"
    log_dir: str = "logs"


@dataclass
class MetricsConfig:
    risk_free_rate: float = 0.04            # annualised
    trading_days: int = 252


@dataclass
class GridConfig:
    output_dir: str = "grid_results"
    summary_file: str = "grid_summary.csv"
    # Parameter grid – lists of values to try
    param_grid: Dict = field(default_factory=lambda: {
        "hmm.n_states": [4, 5, 6],
        "xgb.n_estimators": [200, 300],
        "xgb.max_depth": [4, 5],
        "signal.min_votes": [4, 5],
        "sim.per_trade_max_capital": [15000, 20000],
    })


@dataclass
class MasterConfig:
    data: DataConfig = field(default_factory=DataConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    garch: GARCHConfig = field(default_factory=GARCHConfig)
    xgb: XGBoostConfig = field(default_factory=XGBoostConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    grid: GridConfig = field(default_factory=GridConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "MasterConfig":
        cfg = cls()
        for section, values in d.items():
            if hasattr(cfg, section):
                sec_obj = getattr(cfg, section)
                for k, v in values.items():
                    if hasattr(sec_obj, k):
                        setattr(sec_obj, k, v)
        return cfg


DEFAULT_CONFIG = MasterConfig()
