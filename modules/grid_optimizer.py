"""
Grid Optimizer Module
- Batch execution across parameter combinations
- Logs all results, identifies optimal combination
- Generates summary CSV and JSON for manual review
"""

import logging
import itertools
import traceback
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from copy import deepcopy
from dataclasses import asdict

from modules.backtester_module import Backtester, BacktestConfig, BacktestWindow
from modules.logger_module import ResultsLogger

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Parameter Grid
# ─────────────────────────────────────────────
class ParameterGrid:
    """Generates all combinations of parameter dictionaries."""

    def __init__(self, grid: Dict[str, List[Any]]):
        self.grid = grid

    def __iter__(self):
        keys = list(self.grid.keys())
        values = list(self.grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def __len__(self):
        total = 1
        for v in self.grid.values():
            total *= len(v)
        return total


# ─────────────────────────────────────────────
# Optimization Objective
# ─────────────────────────────────────────────
class OptimizationObjective:
    """Defines how to score a set of metrics for ranking."""

    PRESETS = {
        "sharpe": lambda m: m.get("sharpe_ratio", -999),
        "cagr": lambda m: m.get("cagr", -999),
        "calmar": lambda m: m.get("calmar_ratio", -999),
        "sortino": lambda m: m.get("sortino_ratio", -999),
        "composite": lambda m: (
            m.get("sharpe_ratio", 0) * 0.4 +
            m.get("cagr", 0) * 0.3 +
            m.get("calmar_ratio", 0) * 0.2 -
            abs(m.get("max_drawdown", 0)) * 0.1
        ),
    }

    def __init__(self, preset: str = "sharpe", custom_fn: Optional[Callable] = None):
        if custom_fn:
            self.score_fn = custom_fn
        elif preset in self.PRESETS:
            self.score_fn = self.PRESETS[preset]
        else:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS)}")
        self.preset = preset

    def score(self, metrics: Dict) -> float:
        try:
            return float(self.score_fn(metrics))
        except Exception:
            return -999.0


# ─────────────────────────────────────────────
# Grid Optimizer
# ─────────────────────────────────────────────
class GridOptimizer:
    """
    Runs a full parameter grid search over BacktestConfig parameters.

    Usage:
        grid = {
            "hmm_n_states": [3, 5, 7],
            "min_votes": [4, 5],
            "max_capital_per_trade": [5000, 10000],
        }
        optimizer = GridOptimizer(full_df, windows, grid)
        best = optimizer.run(objective="composite")
    """

    def __init__(
        self,
        full_feature_df: pd.DataFrame,
        windows: List[BacktestWindow],
        param_grid: Dict[str, List[Any]],
        base_config: Optional[BacktestConfig] = None,
        results_dir: str = "results/grid/",
    ):
        self.df = full_feature_df
        self.windows = windows
        self.param_grid = ParameterGrid(param_grid)
        self.base_config = base_config or BacktestConfig()
        self.results_dir = results_dir
        self.results_logger = ResultsLogger(results_dir)
        self._all_runs: List[Dict] = []

        logger.info(f"[GridOptimizer] {len(self.param_grid)} parameter combinations × "
                    f"{len(windows)} windows = {len(self.param_grid) * len(windows)} backtests")

    # ── Run Grid ──────────────────────────────
    def run(
        self,
        objective: str = "composite",
        custom_objective: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute full grid search.
        Returns best parameter combination and its aggregated metrics.
        """
        obj = OptimizationObjective(
            preset=objective,
            custom_fn=custom_objective,
        )
        logger.info(f"[GridOptimizer] Starting grid search | objective={objective}")
        logger.info(f"[GridOptimizer] Grid parameters:")
        for k, v in self.param_grid.grid.items():
            logger.info(f"  {k}: {v}")

        run_summaries = []
        total_combos = len(self.param_grid)

        for combo_idx, params in enumerate(self.param_grid):
            logger.info(f"\n[GridOptimizer] Combo {combo_idx+1}/{total_combos}: {params}")

            # Build config from base + overrides
            config = self._apply_params(deepcopy(self.base_config), params)

            # Run all windows for this config
            combo_results = []
            for w_idx, window in enumerate(self.windows):
                run_id = f"grid_{combo_idx+1:03d}_w{w_idx+1:02d}_test{window.test_start[:4]}"
                try:
                    bt = Backtester(self.df, config, results_dir=self.results_dir)
                    result = bt.run_single(window, run_id=run_id)
                    if result:
                        combo_results.append(result)
                except Exception as e:
                    logger.error(f"[GridOptimizer] Failed: {run_id} — {e}")
                    logger.debug(traceback.format_exc())

            # Aggregate metrics across windows
            agg_metrics = self._aggregate_metrics([r.metrics for r in combo_results])
            score = obj.score(agg_metrics)

            summary = {
                "combo_id": combo_idx + 1,
                "run_id": f"grid_{combo_idx+1:03d}",
                "parameters": params,
                "metrics": agg_metrics,
                "score": score,
                "n_windows": len(combo_results),
            }
            run_summaries.append(summary)
            self._all_runs.append(summary)

            logger.info(
                f"[GridOptimizer] Combo {combo_idx+1} score={score:.4f} | "
                f"Sharpe={agg_metrics.get('sharpe_ratio', 0):.2f} | "
                f"CAGR={agg_metrics.get('cagr', 0):.2%} | "
                f"MaxDD={agg_metrics.get('max_drawdown', 0):.2%}"
            )

        # ── Find best ────────────────────────────
        best = max(run_summaries, key=lambda x: x["score"])
        logger.info("\n" + "=" * 70)
        logger.info("[GridOptimizer] OPTIMIZATION COMPLETE")
        logger.info(f"[GridOptimizer] Best combo: {best['run_id']} | score={best['score']:.4f}")
        logger.info(f"[GridOptimizer] Best params: {best['parameters']}")
        logger.info(f"[GridOptimizer] Best metrics:")
        for k, v in best["metrics"].items():
            if isinstance(v, float):
                logger.info(f"  {k:30s}: {v:.4f}")
            else:
                logger.info(f"  {k:30s}: {v}")

        # ── Save results ─────────────────────────
        self.results_logger.log_grid_summary(
            runs=run_summaries,
            best_run_id=best["run_id"],
            summary_file="grid_summary.json",
        )
        csv_path = self.results_logger.log_csv_summary(
            runs=run_summaries,
            csv_file="grid_summary.csv",
        )

        # Full sorted leaderboard
        self._log_leaderboard(run_summaries, objective)

        return {
            "best_params": best["parameters"],
            "best_metrics": best["metrics"],
            "best_score": best["score"],
            "best_run_id": best["run_id"],
            "all_runs": run_summaries,
            "summary_csv": csv_path,
        }

    def _apply_params(self, config: BacktestConfig, params: Dict) -> BacktestConfig:
        """Override BacktestConfig fields with grid params."""
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"[GridOptimizer] Unknown config field: {key}")
        return config

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Average numeric metrics across windows."""
        if not metrics_list:
            return {}
        keys = set().union(*[m.keys() for m in metrics_list])
        agg = {}
        for k in keys:
            vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
            if vals:
                agg[k] = float(np.mean(vals))
        agg["n_windows"] = len(metrics_list)
        return agg

    def _log_leaderboard(self, runs: List[Dict], objective: str):
        sorted_runs = sorted(runs, key=lambda x: x["score"], reverse=True)
        logger.info(f"\n[GridOptimizer] LEADERBOARD (sorted by {objective}):")
        header = f"{'Rank':>4} | {'ComboID':>8} | {'Score':>8} | {'Sharpe':>7} | {'CAGR':>7} | {'MaxDD':>7} | Params"
        logger.info(header)
        logger.info("-" * 100)
        for rank, r in enumerate(sorted_runs[:20], 1):
            m = r["metrics"]
            logger.info(
                f"{rank:>4} | {r['run_id']:>8} | {r['score']:>8.4f} | "
                f"{m.get('sharpe_ratio',0):>7.2f} | "
                f"{m.get('cagr',0):>7.2%} | "
                f"{m.get('max_drawdown',0):>7.2%} | "
                f"{r['parameters']}"
            )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Return all runs as a flat DataFrame for analysis."""
        rows = []
        for r in self._all_runs:
            row = {"run_id": r["run_id"], "score": r["score"]}
            for k, v in r["parameters"].items():
                row[f"param_{k}"] = v
            for k, v in r["metrics"].items():
                if isinstance(v, (int, float)):
                    row[f"metric_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows).sort_values("score", ascending=False)
