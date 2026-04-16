"""
grid_validation.py
------------------
Smoke-test / validation run for the GridOptimizer.

Purpose
-------
Validates that the full grid → backtest → metrics → ranking pipeline
runs end-to-end without errors before committing to a long real grid run.

Deliberately kept small:
  - 8 parameter combinations  (2 × 2 × 2 grid)
  - 2 backtest windows        (2022 and 2023 test years)
  - Total: 8 × 2 = 16 backtests  → should complete in 5–10 minutes

Grid axes tested
----------------
  hmm_n_states : [3, 5]        — regime granularity
  min_votes    : [2, 3]        — signal strictness
  hmm_features : [core, full]  — HMM feature set
      core = 5 fast features (no garch_vol) → faster HMM training
      full = DEFAULT_FEATURES including garch_vol

What it validates
-----------------
  ✓ GARCH runs before HMM (garch_vol available as HMM feature)
  ✓ garch_returns_col config is respected (futures vs spot)
  ✓ hmm_features list parameter works in grid (list values handled correctly)
  ✓ Full pipeline: GARCH → HMM → XGBoost → signals → simulation → metrics
  ✓ GridOptimizer aggregates, ranks, saves CSV + JSON
  ✓ Leaderboard printed correctly

Usage
-----
  python grid_validation.py
  python grid_validation.py --symbol NIFTY --source duckdb_futures
  python grid_validation.py --debug
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger_module     import setup_logger
from modules.data_module       import DataModule
from modules.regime_module     import MarketRegimeModule
from modules.backtester_module import Backtester, BacktestConfig, BacktestWindow
from modules.grid_optimizer    import GridOptimizer
from main                      import DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# HMM feature sets to compare in the grid
# ─────────────────────────────────────────────────────────────────────────────
# Note: hmm_features takes a list value, not a scalar.
# ParameterGrid handles this correctly — each list here is one option to try,
# not expanded further. The grid iterates over these two options as-is.

HMM_FEATURES_CORE = [
    # Minimal fast set — no garch_vol dependency
    # Use this to check if simpler HMM is competitive
    "log_returns",
    "volatility_20",
    "momentum_10",
    "rsi_14",
    "vix_level",
]

HMM_FEATURES_FULL = MarketRegimeModule.DEFAULT_FEATURES
# = log_returns, garch_vol, momentum_10, volume_ratio, rsi_14,
#   atr_norm, vwap_deviation, vix_level, vix_change, vix_percentile,
#   fut_zscore, basis_pct, oi_zscore, trend_strength


# ─────────────────────────────────────────────────────────────────────────────
# Validation grid — 2 × 2 × 2 = 8 combinations
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_GRID = {
    "hmm_n_states":  [3, 5],
    "min_votes":     [2, 3],
    "hmm_features":  [HMM_FEATURES_CORE, HMM_FEATURES_FULL],
}


# ─────────────────────────────────────────────────────────────────────────────
# Backtest windows — 2 windows
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_WINDOWS = [
    BacktestWindow(
        train_start="2015-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
    ),
    BacktestWindow(
        train_start="2015-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2023-12-31",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Base BacktestConfig — fixed params shared across all combinations
# ─────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = BacktestConfig(
    hmm_n_iter            = 100,           # fewer iterations → faster for validation
    garch_returns_col     = DEFAULT_CONFIG.get("garch_returns_col", "fut_log_ret"),
    exit_votes            = 2,
    use_xgb_filter        = True,
    max_open_trades       = 2,
    total_capital         = 100_000.0,
    max_capital_per_trade = 10_000.0,
    trading_expense       = 10.0,
    slippage_pct          = 0.001,
    confidence_sizing     = True,
    risk_free_rate        = 0.04,
    indicator_names       = DEFAULT_CONFIG.get("signal_indicators"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="grid_validation.py",
        description="Smoke-test GridOptimizer with a small 8-combination grid.",
    )
    p.add_argument("--symbol",    default=DEFAULT_CONFIG["symbol"])
    p.add_argument("--source",    default=DEFAULT_CONFIG["source"],
                   choices=["duckdb_futures", "duckdb_index", "duckdb_equity", "yahoo"])
    p.add_argument("--objective", default="composite",
                   choices=["sharpe", "cagr", "calmar", "sortino", "composite"])
    p.add_argument("--debug",     action="store_true")
    return p.parse_args()


def _describe_grid(grid: dict) -> None:
    """Log a human-readable grid description."""
    total = 1
    for v in grid.values():
        total *= len(v)
    logger = logging.getLogger("grid_validation")
    logger.info(f"Grid: {total} combinations")
    for k, vals in grid.items():
        if k == "hmm_features":
            logger.info(f"  {k}:")
            for i, v in enumerate(vals):
                label = "CORE" if v == HMM_FEATURES_CORE else "FULL"
                logger.info(f"    [{i}] {label}: {v}")
        else:
            logger.info(f"  {k}: {vals}")


def main():
    args  = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO

    log_dir     = "logs/"
    results_dir = "results/grid_validation/"
    os.makedirs(results_dir, exist_ok=True)

    logger = setup_logger("grid_validation", log_dir=log_dir, level=level)
    logger.info("=" * 70)
    logger.info("GRID VALIDATION RUN")
    logger.info(f"  Symbol       : {args.symbol}")
    logger.info(f"  Source       : {args.source}")
    logger.info(f"  GARCH column : {BASE_CONFIG.garch_returns_col}")
    logger.info(f"  Combos       : {2*2*2}")
    logger.info(f"  Windows      : {len(VALIDATION_WINDOWS)}")
    logger.info(f"  Total runs   : {2*2*2 * len(VALIDATION_WINDOWS)}")
    logger.info(f"  Objective    : {args.objective}")
    logger.info("=" * 70)
    _describe_grid(VALIDATION_GRID)

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("\n[Step 1] Loading feature data...")
    dm = DataModule(cache_dir="cache/")
    df = dm.get_feature_data(
        symbol=args.symbol,
        start=DEFAULT_CONFIG["data_start"],
        end=DEFAULT_CONFIG["data_end"],
        source=args.source,
    )
    logger.info(f"  Loaded: {len(df)} bars, {len(df.columns)} columns")
    logger.info(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # ── 2. Validate windows ───────────────────────────────────────────────
    logger.info("\n[Step 2] Validating windows against data...")
    for i, w in enumerate(VALIDATION_WINDOWS, 1):
        train = df.loc[w.train_start:w.train_end]
        test  = df.loc[w.test_start:w.test_end]
        logger.info(f"  Window {i}: train={len(train)} bars | test={len(test)} bars")
        if len(train) < 100:
            logger.warning(f"  ⚠ Window {i}: train only {len(train)} bars")
        if len(test) < 20:
            logger.warning(f"  ⚠ Window {i}: test only {len(test)} bars")

    # ── 3. Run grid ───────────────────────────────────────────────────────
    logger.info("\n[Step 3] Running GridOptimizer...")
    optimizer = GridOptimizer(
        full_feature_df=df,
        windows=VALIDATION_WINDOWS,
        param_grid=VALIDATION_GRID,
        base_config=BASE_CONFIG,
        results_dir=results_dir,
    )
    best = optimizer.run(objective=args.objective)

    # ── 4. Summary ────────────────────────────────────────────────────────
    results_df = optimizer.get_results_dataframe()

    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    # Show table — replace list values with labels for readability
    display_df = results_df.copy()
    if "param_hmm_features" in display_df.columns:
        display_df["param_hmm_features"] = display_df["param_hmm_features"].apply(
            lambda v: "CORE" if v == HMM_FEATURES_CORE else "FULL"
            if isinstance(v, list) else str(v)
        )

    show_cols = (
        ["run_id", "score"]
        + [c for c in display_df.columns if c.startswith("param_")]
        + [c for c in ("metric_sharpe_ratio", "metric_cagr", "metric_max_drawdown")
           if c in display_df.columns]
    )
    logger.info("\n" + display_df[show_cols].to_string(index=False))

    logger.info(f"\n✓ Best combination:")
    params = best["best_params"].copy()
    if "hmm_features" in params:
        params["hmm_features"] = (
            "CORE" if params["hmm_features"] == HMM_FEATURES_CORE else "FULL"
        )
    logger.info(f"  Params : {params}")
    logger.info(f"  Score  : {best['best_score']:.4f}")
    logger.info(f"  Sharpe : {best['best_metrics'].get('sharpe_ratio', 0):.3f}")
    logger.info(f"  CAGR   : {best['best_metrics'].get('cagr', 0):.2%}")
    logger.info(f"  MaxDD  : {best['best_metrics'].get('max_drawdown', 0):.2%}")

    logger.info(f"\n✓ Output files:")
    logger.info(f"  CSV  : {results_dir}grid_summary.csv")
    logger.info(f"  JSON : {results_dir}grid_summary.json")

    print(f"\n✓ Grid validation complete — {len(results_df)} combinations")
    print(f"  Best params : {params}")
    print(f"  Best score  : {best['best_score']:.4f}")
    print(f"  Results     : {results_dir}grid_summary.csv")

    return best


if __name__ == "__main__":
    main()
