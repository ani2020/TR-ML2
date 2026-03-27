"""
visualization/visualizer.py
Visualization Module for the ML Trading System.

Charts produced:
  1. Candlestick / close price chart
  2. Trade signal scatter (Entry ▲ / Exit ▼)
  3. Predicted GARCH volatility line chart
  4. Market regime shading overlay
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")                     # headless safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from models.regime_model import REGIME_COLOURS

logger = logging.getLogger("trading.viz")

ENTRY_MARKER  = "^"     # up-triangle = entry
EXIT_MARKER   = "v"     # down-triangle = exit
ENTRY_COLOUR  = "#00cc44"
EXIT_COLOUR   = "#ff3333"
PRICE_COLOUR  = "#1a1a2e"
VOL_COLOUR    = "#8b5cf6"


class Visualizer:
    """
    Generates a 4-panel figure per backtest window.
    Can also produce standalone charts.
    """

    def __init__(self, output_dir: str = "charts"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── main combined chart ──────────────────────────────────
    def plot_window(self, df: pd.DataFrame,
                    trade_df: Optional[pd.DataFrame] = None,
                    title: str = "", filename: str = "window.png",
                    use_candles: bool = False):
        """
        4-panel figure:
          Panel 1: Price (candlestick or line) with regime shading
          Panel 2: Trade signals scatter
          Panel 3: GARCH forecast volatility
          Panel 4: Equity curve + drawdown
        """
        fig = plt.figure(figsize=(18, 16))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        gs = fig.add_gridspec(4, 1, hspace=0.45,
                              height_ratios=[3, 1.5, 1.5, 1.5])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3], sharex=ax1)

        self._plot_price(ax1, df, use_candles=use_candles)
        self._shade_regimes(ax1, df)
        self._add_regime_legend(ax1, df)

        self._plot_signals(ax2, df, trade_df)

        self._plot_volatility(ax3, df)

        if "equity_curve" in df.columns:
            self._plot_equity(ax4, df)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_facecolor("#fafafa")

        save_path = str(Path(self.output_dir) / filename)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Visualizer] Chart saved → {save_path}")
        return save_path

    # ── panel 1: price ───────────────────────────────────────
    def _plot_price(self, ax, df: pd.DataFrame, use_candles: bool = False):
        ax.set_title("Price  (with Market Regime Shading)", fontsize=10)
        dates = df.index

        if use_candles and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            self._draw_candles(ax, df)
        else:
            ax.plot(dates, df["Close"], color=PRICE_COLOUR, lw=1.2, label="Close")
            if "ema_50" in df.columns:
                ax.plot(dates, df["ema_50"], color="#f59e0b", lw=0.9,
                        linestyle="--", alpha=0.7, label="EMA50")
            if "ema_200" in df.columns:
                ax.plot(dates, df["ema_200"], color="#3b82f6", lw=0.9,
                        linestyle=":", alpha=0.7, label="EMA200")

        ax.set_ylabel("Price", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    def _draw_candles(self, ax, df: pd.DataFrame):
        """Minimal candlestick rendering (no mplfinance dependency)."""
        for i, (dt, row) in enumerate(df.iterrows()):
            o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
            colour = "#26a69a" if c >= o else "#ef5350"
            ax.plot([dt, dt], [l, h], color=colour, lw=0.6)
            ax.add_patch(
                plt.Rectangle(
                    (mdates.date2num(dt.to_pydatetime()) - 0.3, min(o, c)),
                    0.6, abs(c - o),
                    color=colour, zorder=2
                )
            )
        ax.xaxis_date()

    # ── regime shading ───────────────────────────────────────
    def _shade_regimes(self, ax, df: pd.DataFrame):
        if "hmm_regime" not in df.columns:
            return
        regimes = df["hmm_regime"]
        start = df.index[0]
        current = regimes.iloc[0]

        for i in range(1, len(df)):
            if regimes.iloc[i] != current or i == len(df) - 1:
                end = df.index[i]
                colour = REGIME_COLOURS.get(current, "#888888")
                ax.axvspan(start, end, alpha=0.12, color=colour, lw=0)
                current = regimes.iloc[i]
                start   = df.index[i]

    def _add_regime_legend(self, ax, df: pd.DataFrame):
        if "hmm_regime" not in df.columns:
            return
        unique = df["hmm_regime"].unique()
        patches = [
            mpatches.Patch(color=REGIME_COLOURS.get(r, "#888"), alpha=0.5, label=r)
            for r in unique
        ]
        ax.legend(handles=patches, fontsize=6, loc="upper right",
                  title="Regime", title_fontsize=7, ncol=2)

    # ── panel 2: signals ─────────────────────────────────────
    def _plot_signals(self, ax, df: pd.DataFrame,
                       trade_df: Optional[pd.DataFrame] = None):
        ax.set_title("Trade Signals  (▲ Entry  ▼ Exit)", fontsize=10)
        ax.plot(df.index, df["Close"], color=PRICE_COLOUR, lw=0.8, alpha=0.6)

        entries = df[df.get("signal", pd.Series(0, index=df.index)) == 1]
        exits   = df[df.get("signal", pd.Series(0, index=df.index)) == -1]

        ax.scatter(entries.index, entries["Close"], marker=ENTRY_MARKER,
                   color=ENTRY_COLOUR, s=80, zorder=5, label="Entry")
        ax.scatter(exits.index, exits["Close"], marker=EXIT_MARKER,
                   color=EXIT_COLOUR, s=80, zorder=5, label="Exit")

        # If trade_df available, annotate PnL
        if trade_df is not None and not trade_df.empty:
            closed = trade_df[trade_df["status"].isin(["CLOSED", "FORCE_CLOSED"])]
            for _, t in closed.iterrows():
                try:
                    exit_dt = pd.to_datetime(t["exit_date"])
                    pnl = t["pnl"]
                    colour = ENTRY_COLOUR if pnl >= 0 else EXIT_COLOUR
                    ax.annotate(f"{pnl:+.0f}",
                                xy=(exit_dt, t["exit_price"]),
                                fontsize=5, color=colour, ha="center")
                except Exception:
                    pass

        ax.set_ylabel("Price", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    # ── panel 3: volatility ──────────────────────────────────
    def _plot_volatility(self, ax, df: pd.DataFrame):
        ax.set_title("Forecast Volatility (GARCH)", fontsize=10)
        if "garch_vol_forecast" in df.columns:
            ax.plot(df.index, df["garch_vol_forecast"], color=VOL_COLOUR,
                    lw=1.2, label="GARCH(1,1) Annualised Vol")
        if "volatility_20d" in df.columns:
            ax.plot(df.index, df["volatility_20d"], color="#94a3b8",
                    lw=0.8, linestyle="--", alpha=0.6, label="Realised 20d Vol")
        ax.axhline(0.10, color="#f59e0b", lw=0.8, linestyle=":", label="10% threshold")
        ax.set_ylabel("Volatility", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    # ── panel 4: equity + drawdown ───────────────────────────
    def _plot_equity(self, ax, df: pd.DataFrame):
        ax.set_title("Equity Curve & Drawdown", fontsize=10)
        ax.plot(df.index, df["equity_curve"], color="#3b82f6", lw=1.2, label="Equity")
        ax.set_ylabel("Portfolio Value ($)", fontsize=9)

        if "drawdown" in df.columns:
            ax2 = ax.twinx()
            ax2.fill_between(df.index, df["drawdown"] * 100, 0,
                             alpha=0.25, color="#ef4444", label="Drawdown %")
            ax2.set_ylabel("Drawdown (%)", fontsize=9, color="#ef4444")
            ax2.tick_params(labelcolor="#ef4444")

        ax.legend(fontsize=7, loc="upper left")

    # ── convenience: plot all backtest windows ────────────────
    def plot_all_windows(self, backtest_results, use_candles: bool = False):
        for r in backtest_results:
            title = (f"Train: {r.train_start[:4]}→{r.train_end[:4]}  "
                     f"Test: {r.test_start[:4]}→{r.test_end[:4]}  "
                     f"|  Sharpe: {r.metrics.get('sharpe', 'N/A')}  "
                     f"CAGR: {r.metrics.get('cagr', 'N/A'):.1%}" if isinstance(r.metrics.get('cagr'), float) else "")
            filename = f"{r.run_id}_chart.png"
            self.plot_window(
                r.test_df_signals, r.trade_df,
                title=title, filename=filename,
                use_candles=use_candles
            )

    # ── regime summary bar chart ─────────────────────────────
    def plot_regime_distribution(self, df: pd.DataFrame,
                                  filename: str = "regime_dist.png"):
        if "hmm_regime" not in df.columns:
            return
        counts = df["hmm_regime"].value_counts()
        colours = [REGIME_COLOURS.get(r, "#888") for r in counts.index]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts.index, counts.values, color=colours, alpha=0.85)
        ax.bar_label(bars)
        ax.set_title("Market Regime Distribution", fontsize=13)
        ax.set_ylabel("Number of Trading Days")
        ax.set_xlabel("Regime")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        path = str(Path(self.output_dir) / filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[Visualizer] Regime distribution chart → {path}")
        return path
