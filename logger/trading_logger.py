"""
logger/trading_logger.py
Centralised structured logger for the ML Trading System.
Logs to both console and rotating file. Stores JSON-structured run records.
"""

import logging
import logging.handlers
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────
# Colour formatter for console
# ─────────────────────────────────────────────
class _ColourFormatter(logging.Formatter):
    COLOURS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelno, "")
        msg = super().format(record)
        return f"{colour}{msg}{self.RESET}"


def get_logger(name: str = "trading", log_dir: str = "logs",
               level: int = logging.DEBUG) -> logging.Logger:
    """
    Return (or create) a named logger that writes to both console and file.
    Call once per module with __name__.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(level)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_ColourFormatter(fmt, datefmt=date_fmt))
    logger.addHandler(ch)

    # Rotating file handler (10 MB × 5 files)
    log_file = os.path.join(log_dir, f"{name}.log")
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Structured run-record writer
# ─────────────────────────────────────────────
class RunRecorder:
    """
    Writes JSON run-records to <output_dir>/run_records/.
    Each record captures: parameters, regime summary, trade log, metrics.
    """

    def __init__(self, output_dir: str = "backtest_results", run_id: Optional[str] = None):
        self.output_dir = Path(output_dir) / "run_records"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._record: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "sections": {}
        }

    def add(self, section: str, data: Any):
        """Add a named section to this run record."""
        # Convert numpy / pandas types to plain Python for JSON serialisation
        self._record["sections"][section] = _json_safe(data)

    def save(self) -> str:
        path = self.output_dir / f"{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(self._record, f, indent=2, default=str)
        return str(path)


# ─────────────────────────────────────────────
# Trade event logger (plain-text trade journal)
# ─────────────────────────────────────────────
class TradeJournal:
    """
    Appends one line per trade event to a CSV-style text file.
    """

    HEADER = (
        "run_id,event,date,ticker,direction,price,"
        "shares,capital,commission,slippage,confidence,regime,pnl,cum_pnl\n"
    )

    def __init__(self, output_dir: str = "backtest_results", run_id: str = ""):
        path = Path(output_dir) / "trade_journals"
        path.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._file = open(path / f"{run_id}_trades.csv", "w")
        self._file.write(self.HEADER)

    def log(self, event: str, **kwargs):
        row = {
            "run_id": self.run_id,
            "event": event,
            "date": kwargs.get("date", ""),
            "ticker": kwargs.get("ticker", ""),
            "direction": kwargs.get("direction", ""),
            "price": kwargs.get("price", ""),
            "shares": kwargs.get("shares", ""),
            "capital": kwargs.get("capital", ""),
            "commission": kwargs.get("commission", ""),
            "slippage": kwargs.get("slippage", ""),
            "confidence": kwargs.get("confidence", ""),
            "regime": kwargs.get("regime", ""),
            "pnl": kwargs.get("pnl", ""),
            "cum_pnl": kwargs.get("cum_pnl", ""),
        }
        line = ",".join(str(v) for v in row.values()) + "\n"
        self._file.write(line)
        self._file.flush()

    def close(self):
        self._file.close()


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def _json_safe(obj):
    """Recursively convert numpy/pandas scalars to Python native types."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import pandas as pd
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
    except ImportError:
        pass
    return obj
