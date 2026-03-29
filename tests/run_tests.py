#!/usr/bin/env python3
"""
tests/run_tests.py
==================
Standalone test runner — works without pytest installed.

Usage (from project root):
    PANDAS_FUTURE_INFER_STRING=0 python tests/run_tests.py
    PANDAS_FUTURE_INFER_STRING=0 python tests/run_tests.py --module data
    PANDAS_FUTURE_INFER_STRING=0 python tests/run_tests.py -v

With real pytest installed:
    PANDAS_FUTURE_INFER_STRING=0 pytest tests/ -v
"""

import os
import sys
import math
import time
import inspect
import traceback
import argparse

# ── env var must be set before pandas import ──────────────────────────────────
os.environ.setdefault("PANDAS_FUTURE_INFER_STRING", "0")

# ── project root on path ─────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Load conftest (stubs + fixtures) FIRST ───────────────────────────────────
import tests.conftest as _conftest  # noqa: F401 — side-effect: installs stubs

# Apply the sklearn pyarrow check patch (conftest calls this but
# run_tests.py may import modules before conftest fully executes)
try:
    import sklearn.utils._dataframe as _sdf_patch
    _sdf_patch.is_pyarrow_data = lambda X: False
except Exception:
    pass

# Ensure GARCH uses scipy fallback
try:
    from modules.volatility_module import GARCHVolatilityModule as _GARCH
    _GARCH._check_arch = lambda self: False
except Exception:
    pass

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# Minimal pytest compatibility shim
# ═══════════════════════════════════════════════════════════════════════════

_builtins_abs = abs


class approx:
    def __init__(self, v, rel=1e-6, abs=None):
        self.v = float(v)
        self.rel = rel
        self.abs_ = abs

    def __eq__(self, other):
        try:
            other = float(other)
        except Exception:
            return False
        if not math.isfinite(other):
            return False
        if self.abs_ is not None:
            return _builtins_abs(other - self.v) <= self.abs_
        tol = max(self.rel * _builtins_abs(self.v), 1e-12) if self.v != 0 else self.rel
        return _builtins_abs(other - self.v) <= tol

    def __repr__(self):
        return f"approx({self.v})"


class _RaisesCtx:
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(
                f"Expected {self.exc_type.__name__} to be raised, but nothing was raised"
            )
        if not issubclass(et, self.exc_type):
            return False  # re-raise unexpected exception
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"Expected match '{self.match}' in exception message '{ev}'"
            )
        return True  # suppress the expected exception


class _Mark:
    @staticmethod
    def parametrize(*a, **kw):
        return lambda f: f

    @staticmethod
    def slow(f):
        return f

    @staticmethod
    def integration(f):
        return f

    @staticmethod
    def requires_arch(f):
        return f

    @staticmethod
    def requires_hmmlearn(f):
        return f


class _PytestShim:
    approx = staticmethod(lambda *a, **kw: approx(*a, **kw))
    raises = _RaisesCtx
    mark = _Mark()

    @staticmethod
    def fixture(fn=None, scope=None):
        def wrap(f):
            f._is_fixture = True
            return f

        return wrap(fn) if fn is not None else wrap


sys.modules["pytest"] = _PytestShim


# ═══════════════════════════════════════════════════════════════════════════
# Session-level fixtures
# ═══════════════════════════════════════════════════════════════════════════

from tests.conftest import (  # noqa: E402
    make_ohlcv,
    make_feature_df,
    make_trade_df,
    make_equity_curve,
)

_dates = pd.date_range("2020-01-01", periods=252, freq="B")

SESSION_FIXTURES = {
    "random_ohlcv":        make_ohlcv(500, seed=42),
    "random_feature_df":   make_feature_df(500, seed=42),
    "trending_feature_df": make_feature_df(500, seed=7, trending=True),
    "bullish_feature_df":  make_feature_df(500, seed=99, all_bullish_regime=True),
    "small_trade_df":      make_trade_df(30, seed=42, win_rate=0.5, avg_win=150, avg_loss=-100),
    "winning_trade_df":    make_trade_df(30, seed=42, win_rate=0.70, avg_win=200, avg_loss=-80),
    "losing_trade_df":     make_trade_df(30, seed=42, win_rate=0.30, avg_win=80, avg_loss=-200),
    "flat_equity":         pd.Series(100_000.0, index=_dates, name="equity"),
    "rising_equity":       pd.Series(100_000 * (1 + 0.001) ** np.arange(252), index=_dates),
    "crashing_equity":     pd.Series(100_000 * (1 - 0.002) ** np.arange(252), index=_dates),
}


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.details: list = []

    def _resolve_args(self, fn, fixture_map: dict) -> dict:
        sig = inspect.signature(fn)
        return {
            p: fixture_map[p]
            for p in sig.parameters
            if p != "self" and p in fixture_map
        }

    def run_class(self, cls, extra_fixtures: dict = None):
        obj = cls()
        fm = {**SESSION_FIXTURES, **(extra_fixtures or {})}

        # Collect class-level fixtures (methods decorated as fixtures)
        for attr in dir(cls):
            m = getattr(cls, attr, None)
            if m and callable(m) and getattr(m, "_is_fixture", False):
                try:
                    kw = self._resolve_args(m, fm)
                    val = m(obj, **kw) if kw else m(obj)
                    fm[attr] = val
                except Exception as e:
                    fm[attr] = None
                    if self.verbose:
                        print(f"   \033[33mWARN\033[0m  fixture {cls.__name__}.{attr}: "
                              f"{type(e).__name__}: {e}")

        # Run test methods
        for attr in sorted(dir(cls)):
            if not attr.startswith("test_"):
                continue
            method = getattr(obj, attr)
            try:
                kw = self._resolve_args(method, fm)
                method(**kw)
                self.passed += 1
                if self.verbose:
                    print(f"   \033[32mPASS\033[0m  {cls.__name__}::{attr}")
            except AssertionError as e:
                self.failed += 1
                msg = str(e)[:200]
                self.details.append((cls.__name__, attr, "AssertionError", msg))
                if self.verbose:
                    print(f"   \033[31mFAIL\033[0m  {cls.__name__}::{attr}")
                    print(f"         {msg}")
            except Exception as e:
                self.failed += 1
                tb_lines = traceback.format_exc().splitlines()
                relevant = [
                    l for l in tb_lines
                    if "ml_trading" in l or "test_" in l
                ][-3:]
                msg = f"{type(e).__name__}: {str(e)[:180]}"
                self.details.append((cls.__name__, attr, type(e).__name__, msg))
                if self.verbose:
                    print(f"   \033[33mERR\033[0m   {cls.__name__}::{attr}")
                    print(f"         {msg}")
                    for line in relevant:
                        print(f"         {line.strip()}")

    def print_summary(self, elapsed: float):
        total = self.passed + self.failed
        pct = self.passed / total * 100 if total else 0
        print()
        print("  ╔══════════════════════════════════════════════════════════════╗")
        col = "\033[32m" if self.failed == 0 else "\033[31m"
        print(
            f"  ║  {col}{self.passed:3d} passed\033[0m  ·  "
            f"{self.failed:3d} failed  ·  "
            f"{total:3d} total  ·  "
            f"{pct:.0f}%  ·  "
            f"{elapsed:.1f}s          ║"
        )
        print("  ╚══════════════════════════════════════════════════════════════╝")

        if self.details:
            print()
            print("  FAILURES / ERRORS:")
            print("  " + "─" * 62)
            for cls_name, fn_name, err_type, msg in self.details:
                print(f"  ✗  {cls_name}::{fn_name}  [{err_type}]")
                print(f"     {msg[:160]}")
        return self.failed == 0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ML Trading Test Runner")
    parser.add_argument("--module", "-m", choices=["data", "metrics", "signal", "simulation", "all"],
                        default="all", help="Which test module to run")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet",   "-q", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet
    runner = TestRunner(verbose=verbose)
    t0 = time.time()

    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          ML TRADING SYSTEM — UNIT TEST SUITE                ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")

    run_data       = args.module in ("data",       "all")
    run_metrics    = args.module in ("metrics",    "all")
    run_signal     = args.module in ("signal",     "all")
    run_simulation = args.module in ("simulation", "all")

    if run_data:
        print("\n  ── Data Module ─────────────────────────────────────────────────")
        import tests.test_data_module as tdm
        for cls in [tdm.TestFeatureEngineer, tdm.TestCSVPlugin, tdm.TestDataModuleCaching]:
            runner.run_class(cls)

    if run_metrics:
        print("\n  ── Metrics Module ──────────────────────────────────────────────")
        import tests.test_metrics_module as tmm
        for cls in [
            tmm.TestMetricMath,
            tmm.TestRandomInputPipelineMetrics,
            tmm.TestMetricOrdering,
            tmm.TestMetricsEdgeCases,
        ]:
            runner.run_class(cls)

    if run_signal:
        print("\n  ── Signal Engine ───────────────────────────────────────────────")
        import tests.test_signal_engine as tse
        for cls in [tse.TestIndividualIndicators, tse.TestSignalEngineVoting]:
            runner.run_class(cls)

    if run_simulation:
        print("\n  ── Simulation + Pipeline ───────────────────────────────────────")
        import tests.test_simulation_module as tsm
        for cls in [
            tsm.TestTradeSimulator,
            tsm.TestFullPipeline,
            tsm.TestGARCHModule,
            tsm.TestRegimeModule,
            tsm.TestPredictionModule,
        ]:
            runner.run_class(cls)

    success = runner.print_summary(time.time() - t0)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
