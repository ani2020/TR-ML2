"""
optimizer/grid_optimizer.py
Grid search over parameter combinations.
Runs complete backtests in batch, logs all results, identifies optimum.
"""

import copy
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config.settings import MasterConfig
from backtester.backtester import Backtester
from logger.trading_logger import get_logger

logger = get_logger("trading.grid")


class GridOptimizer:
    """
    Batch runner over a parameter grid.

    Parameters are specified as dot-path keys in config.grid.param_grid:
      "hmm.n_states"              → config.hmm.n_states
      "xgb.n_estimators"          → config.xgb.n_estimators
      "signal.min_votes"          → config.signal.min_votes
      etc.

    Results are saved to <output_dir>/grid_summary.csv.
    The optimal combination is determined by maximising the Sharpe ratio
    (configurable via `optimise_by`).
    """

    def __init__(self, base_config: MasterConfig, feature_df: pd.DataFrame,
                 output_dir: Optional[str] = None):
        self.base_config = base_config
        self.feature_df  = feature_df
        self.output_dir  = output_dir or base_config.grid.output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ── parameter grid expansion ─────────────────────────────
    def _expand_grid(self) -> List[Dict]:
        """Return list of {param_path: value} dicts for all combinations."""
        grid = self.base_config.grid.param_grid
        keys   = list(grid.keys())
        values = [grid[k] for k in keys]
        combos = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combos]

    # ── apply one set of params to a config copy ─────────────
    @staticmethod
    def _apply_params(base: MasterConfig, params: Dict) -> MasterConfig:
        cfg_dict = base.to_dict()
        for path, value in params.items():
            parts = path.split(".")
            section = parts[0]
            key     = parts[1]
            if section in cfg_dict and key in cfg_dict[section]:
                cfg_dict[section][key] = value
            else:
                logger.warning(f"[GridOptimizer] Unknown param path: {path}")
        return MasterConfig.from_dict(cfg_dict)

    # ── main run ─────────────────────────────────────────────
    def run(self, optimise_by: str = "sharpe") -> pd.DataFrame:
        combos = self._expand_grid()
        logger.info(f"[GridOptimizer] {len(combos)} parameter combinations to evaluate …")

        all_rows = []
        best_val = float("-inf")
        best_params = None
        best_combo_idx = -1

        for i, params in enumerate(combos):
            logger.info(f"\n[GridOptimizer] Combo {i+1}/{len(combos)}: {params}")
            try:
                cfg = self._apply_params(self.base_config, params)
                cfg.backtest.output_dir = os.path.join(self.output_dir, f"combo_{i+1:04d}")
                cfg.grid.output_dir     = self.output_dir   # keep summary dir unchanged

                bt = Backtester(cfg, self.feature_df, output_dir=cfg.backtest.output_dir)
                results = bt.run_all()

                if not results:
                    logger.warning(f"  No results for combo {i+1}")
                    continue

                summary = bt.summary_table(results)

                # Average key metrics across windows
                numeric_cols = summary.select_dtypes(include="number").columns
                avg_metrics = summary[numeric_cols].mean().to_dict()

                row = {"combo_id": i + 1}
                row.update({f"param_{k.replace('.','_')}": v for k, v in params.items()})
                row.update({f"avg_{k}": round(v, 4) for k, v in avg_metrics.items()})

                # Track best
                metric_val = avg_metrics.get(optimise_by, float("-inf"))
                row["optimise_metric"] = round(metric_val, 4)
                row["is_best"] = False
                if metric_val > best_val:
                    best_val = metric_val
                    best_params = params
                    best_combo_idx = i

                all_rows.append(row)

                # Save incremental summary (so partial runs are not lost)
                pd.DataFrame(all_rows).to_csv(
                    os.path.join(self.output_dir, self.base_config.grid.summary_file),
                    index=False
                )

            except Exception as e:
                logger.error(f"  Combo {i+1} failed: {e}", exc_info=True)
                all_rows.append({"combo_id": i + 1, "error": str(e)})

        if all_rows and best_combo_idx >= 0:
            all_rows[best_combo_idx]["is_best"] = True

        summary_df = pd.DataFrame(all_rows)
        summary_path = os.path.join(self.output_dir, self.base_config.grid.summary_file)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\n[GridOptimizer] Summary saved → {summary_path}")

        if best_params:
            logger.info(f"[GridOptimizer] BEST combination (by {optimise_by}={best_val:.4f}):")
            for k, v in best_params.items():
                logger.info(f"  {k} = {v}")

        return summary_df

    def best_config(self, summary_df: pd.DataFrame) -> Optional[MasterConfig]:
        """Return the MasterConfig for the best parameter combination."""
        best_rows = summary_df[summary_df.get("is_best", False) == True]
        if best_rows.empty:
            return None
        row = best_rows.iloc[0]
        params = {}
        for col in row.index:
            if col.startswith("param_"):
                path = col[len("param_"):].replace("_", ".", 1)
                params[path] = row[col]
        return self._apply_params(self.base_config, params)
