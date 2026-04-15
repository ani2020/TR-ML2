"""
ML Trading System - Module Package
"""
from modules.data_module import DataModule, DuckDBPlugin, YahooFinancePlugin, CSVPlugin, FeatureEngineer
from modules.regime_module import MarketRegimeModule, REGIME_LABELS, REGIME_COLORS
from modules.volatility_module import GARCHVolatilityModule
from modules.prediction_module import PredictionModule
from modules.signal_engine import SignalEngine, IndicatorRegistry, Indicator
from modules.simulation_module import TradeSimulator, SimulationConfig
from modules.metrics_module import MetricsModule
from modules.backtester_module import Backtester, BacktestConfig, BacktestWindow
from modules.grid_optimizer import GridOptimizer, ParameterGrid, OptimizationObjective
from modules.visualization_module import VisualizationModule
from modules.logger_module import setup_logger, ResultsLogger

__all__ = [
    "DataModule", "DuckDBPlugin", "YahooFinancePlugin", "CSVPlugin", "FeatureEngineer",
    "MarketRegimeModule", "REGIME_LABELS", "REGIME_COLORS",
    "GARCHVolatilityModule",
    "PredictionModule",
    "SignalEngine", "IndicatorRegistry", "Indicator",
    "TradeSimulator", "SimulationConfig",
    "MetricsModule",
    "Backtester", "BacktestConfig", "BacktestWindow",
    "GridOptimizer", "ParameterGrid", "OptimizationObjective",
    "VisualizationModule",
    "setup_logger", "ResultsLogger",
]
