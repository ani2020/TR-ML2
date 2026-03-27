"""
Logger Module - Structured logging for the entire ML trading system.
Logs to both console and rotating file. JSON structured output for
results that need to be parsed programmatically.
"""

import os
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────
# Setup System Logger
# ─────────────────────────────────────────────
def setup_logger(
    name: str = "ml_trading",
    log_dir: str = "logs/",
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure and return the root trading system logger."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (rotating)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    # Propagate to root logger's children
    logging.getLogger().setLevel(level)

    logger.info(f"Logger initialized: {log_file}")
    return logger


# ─────────────────────────────────────────────
# Results Logger (JSON)
# ─────────────────────────────────────────────
class ResultsLogger:
    """
    Logs backtest results, parameters, and metrics to structured JSON files.
    Used by backtester and grid optimizer.
    """

    def __init__(self, results_dir: str = "results/"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self._logger = logging.getLogger("ml_trading.results")

    def log_backtest(
        self,
        run_id: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        trade_summary: Optional[Any] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """
        Write a full backtest result to a JSON file.
        Returns filepath.
        """
        record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": self._serialize(params),
            "metrics": self._serialize(metrics),
            "extra": self._serialize(extra or {}),
        }
        if trade_summary is not None:
            try:
                record["trade_summary"] = trade_summary.to_dict(orient="records")
            except Exception:
                record["trade_summary"] = str(trade_summary)

        filepath = os.path.join(self.results_dir, f"backtest_{run_id}.json")
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        self._logger.info(f"[ResultsLogger] Saved backtest → {filepath}")
        return filepath

    def log_grid_summary(
        self,
        runs: list,
        best_run_id: str,
        summary_file: str = "grid_summary.json",
    ) -> str:
        """Write all grid search results to a summary file."""
        filepath = os.path.join(self.results_dir, summary_file)
        record = {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(runs),
            "best_run_id": best_run_id,
            "runs": [self._serialize(r) for r in runs],
        }
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        self._logger.info(f"[ResultsLogger] Grid summary → {filepath} ({len(runs)} runs)")
        return filepath

    def log_csv_summary(
        self,
        runs: list,
        csv_file: str = "grid_summary.csv",
    ) -> str:
        """Write flat CSV summary of all grid runs for easy comparison."""
        import pandas as pd
        rows = []
        for r in runs:
            row = {"run_id": r.get("run_id", "")}
            row.update(self._flatten(r.get("parameters", {}), prefix="param"))
            row.update(self._flatten(r.get("metrics", {}), prefix="metric"))
            rows.append(row)

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.results_dir, csv_file)
        df.to_csv(filepath, index=False)
        self._logger.info(f"[ResultsLogger] CSV summary → {filepath}")
        return filepath

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Make objects JSON-serializable."""
        if isinstance(obj, dict):
            return {k: ResultsLogger._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ResultsLogger._serialize(v) for v in obj]
        elif hasattr(obj, "item"):    # numpy scalar
            return obj.item()
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    @staticmethod
    def _flatten(d: dict, prefix: str = "", sep: str = "_") -> dict:
        out = {}
        for k, v in d.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(ResultsLogger._flatten(v, new_key, sep))
            else:
                out[new_key] = v
        return out


# ─────────────────────────────────────────────
# Assertion Helpers (for Debugging)
# ─────────────────────────────────────────────
def assert_dataframe(df, name: str, required_cols: list = None):
    logger = logging.getLogger("ml_trading.assertions")
    assert df is not None, f"{name} is None"
    assert len(df) > 0, f"{name} is empty"
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        assert not missing, f"{name} missing columns: {missing}"
    logger.debug(f"[Assert] {name}: OK ({len(df)} rows, {len(df.columns)} cols)")


def assert_no_nan_in_col(df, col: str, name: str = ""):
    logger = logging.getLogger("ml_trading.assertions")
    n_nan = df[col].isna().sum()
    if n_nan > 0:
        logger.warning(f"[Assert] {name or col}: {n_nan} NaN values in '{col}'")
    else:
        logger.debug(f"[Assert] {name or col}: no NaNs in '{col}'")


def debug_dataframe_snapshot(df, name: str, n: int = 5):
    logger = logging.getLogger("ml_trading.debug")
    logger.debug(f"[Snapshot] {name} — shape={df.shape}")
    logger.debug(f"[Snapshot] Columns: {list(df.columns)}")
    logger.debug(f"[Snapshot] Head:\n{df.head(n).to_string()}")
    logger.debug(f"[Snapshot] Tail:\n{df.tail(n).to_string()}")
    logger.debug(f"[Snapshot] NaNs:\n{df.isna().sum().to_string()}")
