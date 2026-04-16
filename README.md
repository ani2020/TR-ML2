# ML Trading System

A production-grade, modular ML trading system integrating Hidden Markov Models, GARCH volatility forecasting, XGBoost prediction, and an indicator-based voting signal engine — wired through a rolling backtester, trade simulator, metrics calculator, and grid optimizer.

---

## Architecture Overview

```
Data Module (Yahoo Finance / CSV / Plugin)
        │
        ▼
Feature Engineering (14 default features, plug-and-play)
        │
        ▼
Market Regime Module (HMM — up to 7 states)
        │
        ▼
Volatility Module (GARCH 1,1 — next-period vol forecast)
        │
        ▼
Prediction Module (XGBoost classifier — directional)
        │
        ▼
Signal Engine (7 indicators, voting, regime filter)
        │
        ▼
Trade Simulator (capital, slippage, confidence sizing)
        │
        ▼
Metrics Module (Sharpe, Sortino, Calmar, CAGR, …)
        │
        ├── Backtester (rolling windows)
        ├── Grid Optimizer (parameter sweep)
        └── Visualization Module (price + signals + vol + regimes)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run single analysis (SPY, 10 years)

```bash
python main.py
```

### 3. Rolling window backtest

```bash
python main.py --mode backtest --symbol SPY --years 10
```

### 4. Grid optimization

```bash
python main.py --mode grid --symbol QQQ --years 10
```

### 5. Custom symbol / capital

```bash
python main.py --symbol AAPL --years 8 --capital 250000 --states 7 --votes 5
```

---

## CLI Arguments

| Argument    | Default  | Description                            |
|-------------|----------|----------------------------------------|
| `--mode`    | `single` | `single` / `backtest` / `grid`         |
| `--symbol`  | `SPY`    | Ticker symbol (Yahoo Finance)          |
| `--years`   | `10`     | Years of historical data               |
| `--capital` | `100000` | Total trading capital ($)              |
| `--states`  | `5`      | HMM hidden states (2–7)               |
| `--votes`   | `5`      | Min indicator votes for ENTRY signal   |
| `--refresh` | `False`  | Force data re-fetch (bypass cache)     |
| `--no-xgb`  | `False`  | Disable XGBoost filter on signals      |

---

## Module Reference

### 1. Data Module (`modules/data_module.py`)

Plugin architecture — swap data sources without touching any other module.

```python
from modules.data_module import DataModule, YahooFinancePlugin, CSVPlugin

dm = DataModule(cache_dir="cache/")

# Add a custom plugin
dm.register_plugin("my_source", MyCustomPlugin())

# Fetch features (auto-caches to Parquet)
df = dm.get_feature_data("SPY", "2014-01-01", "2024-01-01", source="yahoo")
```

**Adding a new data source:**

```python
from modules.data_module import DataSourcePlugin

class MyBrokerPlugin(DataSourcePlugin):
    @property
    def name(self): return "my_broker"

    def fetch(self, symbol, start, end):
        # return pd.DataFrame with Open/High/Low/Close/Volume + DatetimeIndex
        ...

dm.register_plugin("broker", MyBrokerPlugin())
```

**Adding new features:**

```python
from modules.data_module import FeatureEngineer

@FeatureEngineer.register("my_feature")
def compute_my_feature(df):
    df["my_feature"] = df["Close"].rolling(7).std()
    return df
```

**CSV format required:**

```
Date,Open,High,Low,Close,Volume
2020-01-02,324.5,325.0,322.1,323.8,82341200
```

---

### 2. Market Regime Module (`modules/regime_module.py`)

Gaussian HMM with 2–7 hidden states. States are auto-labelled by mean return and volatility.

```python
from modules.regime_module import MarketRegimeModule

hmm = MarketRegimeModule(
    n_states=5,          # Bull, Bear, Crash, Sideways, Strong_Bull
    n_iter=200,
    features=["log_returns", "volatility_20", "rsi_14", "adx", "momentum_10"],
)
hmm.fit(train_df)
df_regimes = hmm.predict(test_df)
# Added columns: regime_state, regime_label, regime_is_bullish,
#                regime_prob_Bull, regime_prob_Bear, ...
#                is_bull_run, is_bear_run
```

**State Labels (5-state example):**

| State | Label        | Meaning                          |
|-------|--------------|----------------------------------|
| 0     | Crash        | Extreme negative return + high vol |
| 1     | Bear         | Negative trend                   |
| 2     | Sideways     | No clear direction               |
| 3     | Bull         | Positive trend                   |
| 4     | Strong_Bull  | Highest return, momentum regime  |

**Bull Run / Bear Run identification:**
After prediction, the system scans all consecutive regime windows and tags:
- `is_bull_run = True` on the bullish window with the highest cumulative return
- `is_bear_run = True` on the bearish window with the lowest cumulative return

**Adding new HMM features:**

```python
from modules.regime_module import HMMFeatureRegistry

@HMMFeatureRegistry.register("my_hmm_feature")
def feat(df):
    return df["my_col"].fillna(0).values
```

---

### 3. Volatility Module (`modules/volatility_module.py`)

GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Uses `arch` library if available, falls back to scipy MLE automatically.

```python
from modules.volatility_module import GARCHVolatilityModule

garch = GARCHVolatilityModule(annualize=True)
garch.fit(df["returns"])

next_vol = garch.forecast_next()     # annualized next-period vol
df = garch.add_to_dataframe(df)      # adds garch_vol and garch_next_vol columns
print(garch.params)
# {"omega": ..., "alpha": ..., "beta": ..., "persistence": ...}
```

---

### 4. Prediction Module (`modules/prediction_module.py`)

XGBoost binary classifier predicting whether next-day return > 0.
Integrates HMM regime state, GARCH volatility, and all technical indicators as features.

```python
from modules.prediction_module import PredictionModule

xgb = PredictionModule(
    feature_cols=["returns", "rsi_14", "macd", "regime_state", "garch_vol", ...],
    xgb_params={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
)
xgb.fit(train_df)
df = xgb.predict(test_df)
# Added columns: xgb_pred, xgb_prob_up, xgb_confidence
```

---

### 5. Signal Engine (`modules/signal_engine.py`)

**7 default indicators (plug-and-play):**

| Indicator         | Bullish Condition               |
|-------------------|---------------------------------|
| `momentum`        | 10-day momentum > 1%            |
| `volume`          | Volume > 20-period SMA          |
| `low_volatility`  | GARCH vol < 10%                 |
| `adx`             | ADX > 25                        |
| `price_above_ema` | Close > 50-period EMA           |
| `rsi`             | RSI > 60 AND RSI < 90           |
| `macd`            | MACD line > Signal line         |

**Entry rule:** Regime is bullish **AND** xgb_pred=1 **AND** votes ≥ min_votes (default 5/7)
**Exit rule:** Regime turns non-bullish OR votes drop below exit_votes (default 3)
**Max open trades:** 2 simultaneous positions (FIFO close on exit)

```python
from modules.signal_engine import SignalEngine

engine = SignalEngine(
    min_votes=5,
    exit_votes=3,
    max_open_trades=2,
    use_xgb_filter=True,
)
df = engine.generate(df)
# Added columns: ind_*, vote_count, signal (1/0/-1), signal_label (ENTRY/EXIT/IDLE)
```

**Adding a custom indicator:**

```python
from modules.signal_engine import Indicator

class MyIndicator(Indicator):
    @property
    def name(self): return "my_indicator"

    def compute(self, df):
        return df["Close"].pct_change(5) > 0.02   # 5-day return > 2%

engine.add_indicator(MyIndicator())
```

---

### 6. Trade Simulator (`modules/simulation_module.py`)

```python
from modules.simulation_module import TradeSimulator, SimulationConfig

cfg = SimulationConfig(
    total_capital=100_000,
    max_capital_per_trade=10_000,
    trading_expense=10.0,       # $ flat fee per side
    slippage_pct=0.001,         # 0.1% market impact
    confidence_sizing=True,     # scale position by XGB confidence
    min_confidence=0.1,
)
sim = TradeSimulator(cfg)
result_df = sim.simulate(signal_df)
trade_df  = sim.get_trade_dataframe()
equity    = sim.get_equity_curve()
```

**Trade record columns:**
`trade_id, entry_date, entry_price, exit_date, exit_price, shares, capital_used, regime_at_entry, confidence, vote_count, pnl, returns_pct, cumulative_return, is_closed, exit_reason`

---

### 7. Metrics Module (`modules/metrics_module.py`)

```python
from modules.metrics_module import MetricsModule

metrics = MetricsModule(risk_free_rate=0.02).compute(
    trade_df=trade_df,
    equity=equity_series,
    initial_capital=100_000,
    benchmark_returns=df["returns"],
    start_date="2018-01-01",
    end_date="2018-12-31",
)
```

**All computed metrics:**

| Metric                | Description                            |
|-----------------------|----------------------------------------|
| `sharpe_ratio`        | Annualized Sharpe ratio                |
| `sortino_ratio`       | Downside-risk adjusted return          |
| `calmar_ratio`        | CAGR / Max drawdown                    |
| `max_drawdown`        | Peak-to-trough drawdown                |
| `win_rate`            | % of winning trades                    |
| `alpha`               | Jensen's alpha vs benchmark            |
| `n_trades`            | Total closed trades                    |
| `profit_factor`       | Gross profit / Gross loss              |
| `expectancy_per_trade`| Average $ P&L per trade                |
| `cagr`                | Compound annual growth rate            |
| `max_capital_used`    | Peak total capital deployed            |
| `mean_daily_return`   | Average daily portfolio return         |
| `max_win_streak`      | Longest consecutive winning days       |
| `max_loss_streak`     | Longest consecutive losing days        |
| `data_window`         | Test period label                      |

---

### 8. Backtester (`modules/backtester_module.py`)

Rolling expanding-train windows with single-year test periods.

```python
from modules.backtester_module import Backtester, BacktestConfig

config = BacktestConfig(
    hmm_n_states=5, min_votes=5, total_capital=100_000, ...
)
backtester = Backtester(full_df, config, results_dir="results/")

windows = backtester.build_rolling_windows(
    train_start_year=2016,
    first_test_year=2018,
    last_test_year=2022,
)
# Produces:
#   Train: 2016-01-01 → 2017-12-31  |  Test: 2018
#   Train: 2016-01-01 → 2018-12-31  |  Test: 2019
#   Train: 2016-01-01 → 2019-12-31  |  Test: 2020
#   Train: 2016-01-01 → 2020-12-31  |  Test: 2021
#   Train: 2016-01-01 → 2021-12-31  |  Test: 2022

results = backtester.run_all(windows)
```

Each run logs: data slice → HMM train → GARCH fit → XGB train → predict → signals → simulate → metrics → JSON file.

---

### 9. Grid Optimizer (`modules/grid_optimizer.py`)

```python
from modules.grid_optimizer import GridOptimizer

grid = {
    "hmm_n_states":          [3, 5, 7],
    "min_votes":             [4, 5, 6],
    "max_capital_per_trade": [5_000, 10_000, 20_000],
}
# 3 × 3 × 3 = 27 combinations × n_windows backtests

optimizer = GridOptimizer(
    full_feature_df=df,
    windows=windows,
    param_grid=grid,
    results_dir="results/grid/",
)
best = optimizer.run(objective="composite")
# objective: "sharpe" | "cagr" | "calmar" | "sortino" | "composite"

print(best["best_params"])   # {"hmm_n_states": 5, "min_votes": 5, ...}
print(best["best_score"])
```

**Composite score** = 0.4 × Sharpe + 0.3 × CAGR + 0.2 × Calmar - 0.1 × |MaxDD|

**Output files:**
- `results/grid/grid_summary.json` — all runs ranked, full params + metrics
- `results/grid/grid_summary.csv` — flat CSV for spreadsheet comparison
- `results/grid/backtest_grid_*.json` — individual run detail

---

### 10. Visualization Module (`modules/visualization_module.py`)

```python
from modules.visualization_module import VisualizationModule

viz = VisualizationModule(output_dir="plots/")

# Full 5-panel analysis chart
viz.plot_full_analysis(
    df=result_df,
    title="SPY 2018 Backtest",
    chart_type="line",        # or "candle"
    include_equity=True,
)

# Regime distribution pie chart
viz.plot_regime_distribution(df=result_df)

# Grid search heatmap
viz.plot_grid_heatmap(
    results_df, x_param="hmm_n_states", y_param="min_votes",
    metric="metric_sharpe_ratio",
)
```

**Chart panels (top to bottom):**
1. **Price** — close line or OHLC candles with regime-coloured background shading, SMA20, EMA50, ENTRY ▲ / EXIT ▼ scatter markers, Bull/Bear run text annotations
2. **Volume** — bar chart coloured by market regime
3. **GARCH Volatility** — conditional vol line with 10% threshold
4. **RSI** — with overbought/oversold bands and bull-zone fill
5. **Equity** — portfolio value curve (optional)

---

### 11. Logger Module (`modules/logger_module.py`)

```python
from modules.logger_module import setup_logger, ResultsLogger
from modules.logger_module import assert_dataframe, assert_no_nan_in_col, debug_dataframe_snapshot

# Rotating file + console logger
logger = setup_logger("ml_trading", log_dir="logs/", level=logging.DEBUG)

# Structured JSON results
rl = ResultsLogger("results/")
rl.log_backtest(run_id="run_01", params=config, metrics=metrics, trade_summary=df)
rl.log_grid_summary(runs=all_runs, best_run_id="grid_003")
rl.log_csv_summary(runs=all_runs)

# Debugging helpers
assert_dataframe(df, "my_df", required_cols=["Close", "returns"])
assert_no_nan_in_col(df, "garch_vol", name="garch check")
debug_dataframe_snapshot(df, "enriched_df", n=5)
```

---

## Integration Flow

```
Features ──────────────────────────────────────────────────────┐
    │                                                           │
    ▼                                                           │
HMM Regime Detection                                           │
    │  regime_state, regime_label, regime_is_bullish            │
    │  regime_prob_Bull / Bear / Crash / …                      │
    │  is_bull_run ◄── highest cumulative return window         │
    │  is_bear_run ◄── lowest cumulative return window          │
    ▼                                                           │
GARCH(1,1) Volatility                                          │
    │  garch_vol (in-sample conditional volatility)             │
    │  garch_next_vol (one-step-ahead forecast)                 │
    ▼                                                           │
XGBoost Prediction  ◄── uses ALL above as features ◄───────────┘
    │  xgb_pred (0/1), xgb_prob_up, xgb_confidence
    ▼
Signal Engine (regime gate + 7-indicator vote)
    │  ENTRY: regime_is_bullish AND xgb_pred=1 AND votes ≥ 5
    │  EXIT:  regime bearish OR votes < 3
    │  Max 2 simultaneous open positions
    ▼
Trade Simulator → P&L, equity curve
    ▼
Metrics → Sharpe, CAGR, MaxDD, Win Rate, Alpha, …
    ▼
Visualization → 5-panel PNG
```

---

## File / Folder Layout

```
ml_trading/
├── main.py                          # CLI entry point & orchestrator
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── data_module.py               # Plugin data loading + feature engineering
│   ├── regime_module.py             # HMM market regime detection
│   ├── volatility_module.py         # GARCH(1,1) volatility model
│   ├── prediction_module.py         # XGBoost direction classifier
│   ├── signal_engine.py             # Indicator voting + signal generation
│   ├── simulation_module.py         # Trade execution & P&L tracking
│   ├── metrics_module.py            # Performance metrics (15 metrics)
│   ├── backtester_module.py         # Rolling window backtesting
│   ├── grid_optimizer.py            # Parameter grid search
│   ├── visualization_module.py      # Charts & plots
│   └── logger_module.py             # Logging, assertions, result persistence
├── cache/                           # Parquet-cached raw & feature data
├── logs/                            # Rotating .log files
├── results/                         # JSON backtest results
│   └── grid/                        # Grid optimizer outputs
└── plots/                           # Generated PNG charts
```

---

## Extending the System

| Extension point          | How to add                                                   |
|--------------------------|--------------------------------------------------------------|
| New data source          | Subclass `DataSourcePlugin`, call `dm.register_plugin()`     |
| New feature              | `@FeatureEngineer.register("name")` decorator                |
| New HMM input feature    | `@HMMFeatureRegistry.register("name")` decorator             |
| New trading indicator    | Subclass `Indicator`, call `engine.add_indicator()`          |
| New optimizer objective  | Pass `custom_fn` to `OptimizationObjective`                  |
| New metric               | Add method to `MetricsModule.compute()`                      |

---

## Output Files Reference

| File                                  | Contents                                     |
|---------------------------------------|----------------------------------------------|
| `logs/ml_trading_TIMESTAMP.log`       | Full timestamped system log                  |
| `results/backtest_RUNID.json`         | Params + metrics + trade list per run        |
| `results/grid/grid_summary.json`      | All grid runs ranked by score                |
| `results/grid/grid_summary.csv`       | Flat CSV for spreadsheet analysis            |
| `plots/RUNID_analysis.png`            | 5-panel price/signals/vol/RSI/equity chart   |
| `plots/RUNID_regimes.png`             | Regime distribution pie chart                |
| `plots/grid_heatmap_*.png`            | Grid search metric heatmap                   |
| `cache/SYMBOL_START_END_raw.parquet`  | Cached OHLCV data                            |
| `cache/SYMBOL_START_END_features_*.parquet` | Cached feature-engineered data         |
