"""
Visualization Module — Interactive Plotly Edition
==================================================
Fully interactive HTML dashboards with:
  ✓ Zoom / pan / box-select on every panel
  ✓ Shared x-axis — all panels scroll and zoom together
  ✓ Synchronized crosshair across all rows
  ✓ Range-selector buttons: 1M / 3M / 6M / 1Y / 2Y / All
  ✓ Rangeslider for broad navigation
  ✓ Unified hover tooltip across all rows
  ✓ Candlestick or close-line price panel
  ✓ ENTRY ▲ / EXIT ▼ markers with rich hover detail
  ✓ Bollinger band fill
  ✓ Regime background shading with colour legend
  ✓ Bull-run / Bear-run highlighted bands
  ✓ GARCH conditional volatility + 10% threshold
  ✓ RSI with overbought/oversold zones
  ✓ Volume bars coloured by regime
  ✓ Portfolio equity curve + drawdown overlay

Outputs are self-contained .html files. Open in any browser.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Regime palette ──────────────────────────────────────────────────────────
REGIME_COLORS: Dict[str, str] = {
    "Bear":           "#d32f2f",
    "Sideways_Bear":  "#ef9a9a",
    "Sideways":       "#78909c",
    "Sideways_Bull":  "#a5d6a7",
    "Bull":           "#2e7d32",
    "Strong_Bull":    "#1b5e20",
    "Crash":          "#6a1b9a",
}
BULLISH_REGIMES = {"Bull", "Strong_Bull", "Sideways_Bull"}


def _rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


# ─── Map subplot row → Plotly axis names ────────────────────────────────────
def _xaxis_name(row: int) -> str:
    return "xaxis" if row == 1 else f"xaxis{row}"

def _yaxis_name(row: int) -> str:
    return "yaxis" if row == 1 else f"yaxis{row}"

def _xref(row: int) -> str:
    return "x" if row == 1 else f"x{row}"

def _yref_domain(row: int) -> str:
    """Valid Plotly yref for a shape that spans the full row height."""
    return "y domain" if row == 1 else f"y{row} domain"


# ────────────────────────────────────────────────────────────────────────────
class VisualizationModule:
    """
    Interactive Plotly-based visualization for the ML trading system.
    All outputs are self-contained HTML files.
    """

    # Relative row heights [price, volume, garch, rsi, equity]
    _ROW_HEIGHTS = [0.42, 0.12, 0.15, 0.15, 0.16]

    def __init__(
        self,
        output_dir: str = "plots/",
        theme: str = "plotly_dark",
        save_static: bool = False,
    ):
        self.output_dir = output_dir
        self.theme = theme
        self.save_static = save_static
        os.makedirs(output_dir, exist_ok=True)
        self._check_plotly()

    @staticmethod
    def _check_plotly():
        try:
            import plotly  # noqa: F401
        except ImportError:
            raise ImportError("plotly required:  pip install plotly")

    # ════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════════════

    def plot_full_analysis(
        self,
        df: pd.DataFrame,
        title: str = "ML Trading System — Full Analysis",
        filename: Optional[str] = None,
        chart_type: str = "candle",      # "candle" | "line"
        include_equity: bool = True,
        auto_open: bool = False,
    ) -> str:
        """Build and save the full interactive multi-panel dashboard."""
        from plotly.subplots import make_subplots

        has_equity = include_equity and "equity" in df.columns
        n_rows = 4 + (1 if has_equity else 0)
        row_heights = self._ROW_HEIGHTS[:n_rows]

        subplot_titles = (
            ["Price + Signals", "Volume", "GARCH Volatility", "RSI (14)"]
            + (["Portfolio Equity"] if has_equity else [])
        )

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.022,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        # ── Price panel ──────────────────────────────────────────────────
        if chart_type == "candle":
            self._add_candlestick(fig, df, row=1)
        else:
            self._add_close_line(fig, df, row=1)

        self._add_moving_averages(fig, df, row=1)
        self._add_bollinger(fig, df, row=1)
        self._add_signals(fig, df, row=1)

        # ── Other panels ─────────────────────────────────────────────────
        self._add_volume(fig, df, row=2)
        self._add_garch(fig, df, row=3)
        self._add_rsi(fig, df, row=4)
        if has_equity:
            self._add_equity(fig, df, row=5)

        # ── Backgrounds + annotations (added AFTER traces so shapes list
        #    is set once, not overwritten) ──────────────────────────────
        self._add_regime_shapes(fig, df, n_rows)
        self._add_bull_bear_vrects(fig, df)

        # ── Layout (axes, hover, range-selector, rangeslider) ────────────
        self._apply_layout(fig, title, n_rows)

        # ── Save ─────────────────────────────────────────────────────────
        if filename is None:
            filename = f"analysis_{title.replace(' ', '_')[:40]}.html"
        if not filename.endswith(".html"):
            filename = os.path.splitext(filename)[0] + ".html"
        filepath = os.path.join(self.output_dir, filename)

        fig.write_html(
            filepath,
            include_plotlyjs="cdn",
            full_html=True,
            auto_open=auto_open,
            config={
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": os.path.splitext(os.path.basename(filename))[0],
                    "height": 900, "width": 1800, "scale": 2,
                },
            },
        )
        logger.info(f"[Viz] Dashboard saved → {filepath}")
        if self.save_static:
            self._export_static(fig, filepath.replace(".html", ".png"))
        return filepath

    # ─────────────────────────────────────────────────────────────────────
    def plot_regime_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Market Regime Distribution",
        filename: Optional[str] = None,
        auto_open: bool = False,
    ) -> str:
        """Interactive donut chart of regime proportions."""
        import plotly.graph_objects as go

        if "regime_label" not in df.columns:
            logger.warning("[Viz] No regime_label — skipping distribution chart.")
            return ""

        counts = df["regime_label"].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()
        colors = [REGIME_COLORS.get(l, "#607d8b") for l in labels]

        hover_parts = []
        for lbl in labels:
            mask = df["regime_label"] == lbl
            n = mask.sum()
            share = n / len(df) * 100
            text = f"<b>{lbl}</b><br>Days: {n:,}<br>Share: {share:.1f}%"
            if "returns" in df.columns:
                ann_ret = df.loc[mask, "returns"].mean() * 252
                text += f"<br>Ann. Return: {ann_ret:+.1%}"
            if "volatility_20" in df.columns:
                ann_vol = df.loc[mask, "volatility_20"].mean()
                text += f"<br>Ann. Vol: {ann_vol:.1%}"
            hover_parts.append(text)

        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.45,
            marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.3)", width=1)),
            hovertext=hover_parts, hoverinfo="text",
            textinfo="label+percent", textfont=dict(size=12),
            direction="clockwise", sort=False,
        ))
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=18), x=0.5, xanchor="center"),
            template=self.theme, showlegend=True,
            legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=12)),
            annotations=[dict(text="Regime<br>Split", x=0.5, y=0.5,
                              font=dict(size=13), showarrow=False)],
            height=480, margin=dict(l=40, r=160, t=70, b=40),
        )

        filename = filename or "regime_distribution.html"
        if not filename.endswith(".html"):
            filename = os.path.splitext(filename)[0] + ".html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath, include_plotlyjs="cdn", auto_open=auto_open)
        logger.info(f"[Viz] Regime distribution → {filepath}")
        return filepath

    # ─────────────────────────────────────────────────────────────────────
    def plot_grid_heatmap(
        self,
        results_df: pd.DataFrame,
        x_param: str,
        y_param: str,
        metric: str = "metric_sharpe_ratio",
        filename: Optional[str] = None,
        auto_open: bool = False,
    ) -> str:
        """Interactive heatmap of a grid-search metric across two parameters."""
        import plotly.graph_objects as go

        xcol, ycol = f"param_{x_param}", f"param_{y_param}"
        if xcol not in results_df.columns or ycol not in results_df.columns:
            logger.warning(f"[Viz] Heatmap: columns '{xcol}'/'{ycol}' missing.")
            return ""

        try:
            pivot = results_df.pivot_table(values=metric, index=ycol, columns=xcol, aggfunc="mean")
        except Exception as e:
            logger.warning(f"[Viz] Heatmap pivot failed: {e}")
            return ""

        metric_label = metric.replace("metric_", "").replace("_", " ").title()
        text_vals = [[f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row] for row in pivot.values]

        fig = go.Figure(go.Heatmap(
            z=pivot.values.tolist(),
            x=[str(v) for v in pivot.columns],
            y=[str(v) for v in pivot.index],
            colorscale="RdYlGn",
            text=text_vals, texttemplate="%{text}", textfont=dict(size=12),
            hovertemplate=(f"<b>{x_param}</b>=%{{x}}<br><b>{y_param}</b>=%{{y}}<br>"
                           f"<b>{metric_label}</b>=%{{z:.4f}}<extra></extra>"),
            colorbar=dict(title=dict(text=metric_label, font=dict(size=12)), thickness=16),
        ))
        fig.update_layout(
            title=dict(text=f"<b>Grid Search — {metric_label}</b>",
                       font=dict(size=16), x=0.5, xanchor="center"),
            xaxis=dict(title=x_param, type="category"),
            yaxis=dict(title=y_param, type="category"),
            template=self.theme, height=480,
            margin=dict(l=80, r=100, t=70, b=60),
        )

        filename = filename or f"grid_heatmap_{x_param}_{y_param}.html"
        if not filename.endswith(".html"):
            filename = os.path.splitext(filename)[0] + ".html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath, include_plotlyjs="cdn", auto_open=auto_open)
        logger.info(f"[Viz] Grid heatmap → {filepath}")
        return filepath

    # ════════════════════════════════════════════════════════════════════════
    # PRIVATE — TRACE BUILDERS
    # ════════════════════════════════════════════════════════════════════════

    def _add_candlestick(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            self._add_close_line(fig, df, row)
            return

        regime_vals = df.get("regime_label", pd.Series([""] * len(df), index=df.index))
        hover = [
            (f"<b>{idx.strftime('%Y-%m-%d')}</b><br>"
             f"O: ${o:.2f}  H: ${h:.2f}  L: ${l:.2f}  C: ${c:.2f}<br>"
             f"Chg: {((c-o)/o*100):+.2f}%"
             + (f"<br>Regime: {rg}" if rg else ""))
            for idx, o, h, l, c, rg in zip(
                df.index, df["Open"], df["High"], df["Low"], df["Close"], regime_vals)
        ]
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="OHLC",
            increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
            hovertext=hover, hoverinfo="text",
            showlegend=True, whiskerwidth=0,
        ), row=row, col=1)

    def _add_close_line(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name="Close",
            line=dict(color="#00e5ff", width=1.5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:.2f}<extra></extra>",
        ), row=row, col=1)

    def _add_moving_averages(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        for col, name, color, dash in [
            ("sma_20", "SMA 20", "#ff9800", "dot"),
            ("ema_50", "EMA 50", "#ce93d8", "dash"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], mode="lines", name=name,
                    line=dict(color=color, width=1.0, dash=dash), opacity=0.85,
                    hovertemplate=f"{name}: $%{{y:.2f}}<extra></extra>",
                ), row=row, col=1)

    def _add_bollinger(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        if "bb_upper" not in df.columns or "bb_lower" not in df.columns:
            return
        bb_color = "#546e7a"
        # Upper — reference line for fill
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], mode="lines",
            line=dict(color=_rgba(bb_color, 0.5), width=0.8, dash="dot"),
            name="BB Upper", showlegend=False, hoverinfo="skip",
        ), row=row, col=1)
        # Lower — fills back to upper
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], mode="lines",
            line=dict(color=_rgba(bb_color, 0.5), width=0.8, dash="dot"),
            fill="tonexty", fillcolor=_rgba(bb_color, 0.07),
            name="Bollinger", showlegend=True, hoverinfo="skip",
        ), row=row, col=1)

    def _add_signals(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        if "signal" not in df.columns:
            return

        entries = df[df["signal"] == 1]
        exits   = df[df["signal"] == -1]

        def _hover(subset: pd.DataFrame, kind: str) -> List[str]:
            out = []
            for idx, r in subset.iterrows():
                lines = [f"<b>{kind} — {idx.strftime('%Y-%m-%d')}</b>",
                         f"Price: ${r['Close']:.2f}"]
                vc = r.get("vote_count")
                if vc is not None and not pd.isna(vc):
                    lines.append(f"Votes: {int(vc)}/7")
                rl = r.get("regime_label")
                if rl:
                    lines.append(f"Regime: {rl}")
                conf = r.get("xgb_confidence_at_signal")
                if conf is not None and not pd.isna(conf):
                    lines.append(f"XGB Conf: {conf:.1%}")
                out.append("<br>".join(lines))
            return out

        if len(entries) > 0:
            fig.add_trace(go.Scatter(
                x=entries.index,
                y=entries["Close"] * 0.984,
                mode="markers", name=f"ENTRY ({len(entries)})",
                marker=dict(symbol="triangle-up", color="#00e676", size=12,
                            line=dict(color="white", width=1)),
                hovertext=_hover(entries, "ENTRY"), hoverinfo="text",
                legendgroup="signals",
            ), row=row, col=1)

        if len(exits) > 0:
            fig.add_trace(go.Scatter(
                x=exits.index,
                y=exits["Close"] * 1.016,
                mode="markers", name=f"EXIT ({len(exits)})",
                marker=dict(symbol="triangle-down", color="#ff1744", size=12,
                            line=dict(color="white", width=1)),
                hovertext=_hover(exits, "EXIT"), hoverinfo="text",
                legendgroup="signals",
            ), row=row, col=1)

    def _add_volume(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        if "regime_label" in df.columns:
            bar_colors = [REGIME_COLORS.get(lbl, "#607d8b") for lbl in df["regime_label"]]
        else:
            prev = df["Close"].shift(1).fillna(df["Close"])
            bar_colors = ["#26a69a" if c >= p else "#ef5350"
                          for c, p in zip(df["Close"], prev)]

        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=bar_colors, marker_opacity=0.65,
            name="Volume", showlegend=False,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ), row=row, col=1)

        if "volume_sma_20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["volume_sma_20"], mode="lines",
                name="Vol SMA 20",
                line=dict(color="rgba(255,255,255,0.7)", width=1, dash="dot"),
                showlegend=False,
                hovertemplate="Vol SMA20: %{y:,.0f}<extra></extra>",
            ), row=row, col=1)

    def _add_garch(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        col   = "garch_vol" if "garch_vol" in df.columns else "volatility_20"
        label = "GARCH(1,1) Vol" if col == "garch_vol" else "Rolling Vol (20d)"
        if col not in df.columns:
            return

        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], mode="lines", name=label,
            line=dict(color="#ffca28", width=1.3),
            fill="tozeroy", fillcolor=_rgba("#ffca28", 0.10),
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label}: %{{y:.2%}}<extra></extra>",
        ), row=row, col=1)

        fig.add_hline(
            y=0.10, row=row, col=1,
            line=dict(color="#ff7043", width=1.0, dash="dash"),
            annotation_text="10% threshold",
            annotation_font=dict(color="#ff7043", size=10),
            annotation_position="top right",
        )
        fig.update_yaxes(tickformat=".0%", row=row, col=1)

    def _add_rsi(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        if "rsi_14" not in df.columns:
            return

        rsi = df["rsi_14"].fillna(50)

        fig.add_trace(go.Scatter(
            x=df.index, y=rsi, mode="lines", name="RSI (14)",
            line=dict(color="#7986cb", width=1.3),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>RSI: %{y:.1f}<extra></extra>",
        ), row=row, col=1)

        # Overbought fill (70-100)
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi.clip(upper=100),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi.clip(lower=70),
            fill="tonexty", fillcolor=_rgba("#ef5350", 0.15),
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            name="Overbought",
        ), row=row, col=1)

        fig.add_hrect(y0=60, y1=90, row=row, col=1,
                      fillcolor=_rgba("#66bb6a", 0.07), line_width=0)
        fig.add_hrect(y0=0,  y1=30, row=row, col=1,
                      fillcolor=_rgba("#ef5350", 0.07), line_width=0)

        for level, color, ann in [
            (70, "#ef5350", "OB 70"),
            (60, "#66bb6a", "Bull 60"),
            (30, "#29b6f6", "OS 30"),
        ]:
            fig.add_hline(
                y=level, row=row, col=1,
                line=dict(color=color, width=0.8, dash="dot"),
                annotation_text=ann,
                annotation_font=dict(color=color, size=9),
                annotation_position="top right",
            )

        fig.update_yaxes(range=[0, 100], row=row, col=1)

    def _add_equity(self, fig, df: pd.DataFrame, row: int):
        import plotly.graph_objects as go
        eq = df["equity"].dropna()
        if len(eq) == 0:
            return

        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, mode="lines", name="Equity",
            line=dict(color="#69f0ae", width=1.5),
            fill="tozeroy", fillcolor=_rgba("#69f0ae", 0.08),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Equity: $%{y:,.0f}<extra></extra>",
        ), row=row, col=1)

        peak   = eq.cummax()
        dd_pct = ((eq - peak) / peak).fillna(0)
        fig.add_trace(go.Scatter(
            x=eq.index, y=dd_pct.values, mode="lines", name="Drawdown %",
            line=dict(color="#ff1744", width=1.0),
            fill="tozeroy", fillcolor=_rgba("#ff1744", 0.12),
            visible="legendonly",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2%}<extra></extra>",
        ), row=row, col=1)

        fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=row, col=1)

    # ════════════════════════════════════════════════════════════════════════
    # BACKGROUNDS — applied ONCE after all traces are added
    # ════════════════════════════════════════════════════════════════════════

    def _get_regime_spans(self, df: pd.DataFrame) -> List[Tuple]:
        spans: List[Tuple] = []
        if "regime_label" not in df.columns or len(df) == 0:
            return spans
        prev_lbl  = df["regime_label"].iloc[0]
        prev_date = df.index[0]
        for date, row in df.iloc[1:].iterrows():
            lbl = row["regime_label"]
            if lbl != prev_lbl:
                spans.append((prev_date, date, prev_lbl))
                prev_lbl  = lbl
                prev_date = date
        spans.append((prev_date, df.index[-1], prev_lbl))
        return spans

    def _add_regime_shapes(self, fig, df: pd.DataFrame, n_rows: int):
        """
        Add coloured background rectangles for each regime period across every
        subplot row.  Uses correct Plotly xref/yref naming for multi-row figures.
        Adds invisible legend entries so colours appear in the legend.
        """
        import plotly.graph_objects as go

        spans = self._get_regime_spans(df)
        if not spans:
            return

        shapes = []
        added_legend: set = set()

        for x0, x1, label in spans:
            color    = REGIME_COLORS.get(label, "#607d8b")
            fill_col = _rgba(color, 0.10)

            for row in range(1, n_rows + 1):
                # xref must match the shared x-axis (always "x" since axes are shared)
                # yref uses the per-row y domain syntax
                shapes.append(dict(
                    type="rect",
                    xref="x",                   # shared x-axis is always "x"
                    yref=_yref_domain(row),      # e.g. "y domain", "y2 domain" …
                    x0=x0, x1=x1,
                    y0=0,  y1=1,
                    fillcolor=fill_col,
                    line=dict(width=0),
                    layer="below",
                ))

            # Legend swatch — one per unique label
            if label not in added_legend:
                added_legend.add(label)
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(color=color, size=12, symbol="square"),
                    name=f"◼ {label}",
                    legendgroup=f"regime_{label}",
                    showlegend=True,
                    hoverinfo="skip",
                ), row=1, col=1)

        # Set shapes in one call — do NOT use update_layout() to set shapes
        # because it would overwrite shapes already added by add_hline / add_hrect.
        # Instead use fig.layout.shapes which is additive via tuple concatenation.
        fig.layout.shapes = list(fig.layout.shapes or []) + shapes

    def _add_bull_bear_vrects(self, fig, df: pd.DataFrame):
        """Highlight the single best Bull Run and Bear Run window."""
        for col, color, label in [
            ("is_bull_run", "#00e676", "🐂 Bull Run"),
            ("is_bear_run", "#ff1744", "🐻 Bear Run"),
        ]:
            if col not in df.columns:
                continue
            mask = df[col].fillna(False)
            if not mask.any():
                continue
            run_df = df[mask]
            x0, x1 = run_df.index[0], run_df.index[-1]
            # add_vrect spans all rows automatically
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=_rgba(color, 0.15),
                line=dict(color=color, width=1.2, dash="dot"),
                annotation_text=label,
                annotation_position="top left",
                annotation_font=dict(color=color, size=12),
            )

    # ════════════════════════════════════════════════════════════════════════
    # LAYOUT
    # ════════════════════════════════════════════════════════════════════════

    def _apply_layout(self, fig, title: str, n_rows: int):
        """
        Configure axes, hover, range-selector, rangeslider and visual theme.

        Key rules that fix the blank price panel:
          • y-axes for rows 2..n use "right" side; row 1 (price) uses LEFT
            so Plotly does not collapse the price subplot.
          • x-axis tick labels are hidden for all rows except the last.
          • The rangeslider is placed on the LAST row's x-axis (xaxis{n}).
          • Spike lines are enabled on every axis for the crosshair.
        """
        range_buttons = [
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(count=2,  label="2Y", step="year",  stepmode="backward"),
            dict(step="all", label="All"),
        ]

        y_titles = {
            1: "Price ($)", 2: "Volume", 3: "Volatility",
            4: "RSI", 5: "Equity ($)"
        }

        # ── Build axis update dicts ───────────────────────────────────────
        layout_updates: dict = {}

        for r in range(1, n_rows + 1):
            xax = _xaxis_name(r)
            yax = _yaxis_name(r)
            is_bottom = (r == n_rows)

            # x-axis
            xax_cfg = dict(
                showticklabels=is_bottom,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                showline=True,
                linecolor="rgba(255,255,255,0.08)",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
                spikecolor="rgba(255,255,255,0.3)",
                spikedash="solid",
                zeroline=False,
            )
            if is_bottom:
                xax_cfg.update(dict(
                    type="date",
                    rangeselector=dict(
                        buttons=range_buttons,
                        bgcolor="#1e2d40",
                        activecolor="#0284c7",
                        bordercolor="rgba(255,255,255,0.15)",
                        font=dict(size=11, color="white"),
                        x=0, y=1.04, xanchor="left",
                    ),
                    rangeslider=dict(
                        visible=True,
                        thickness=0.04,
                        bgcolor="#0f172a",
                        bordercolor="rgba(255,255,255,0.1)",
                    ),
                ))
            layout_updates[xax] = xax_cfg

            # y-axis
            # FIX: price panel (row 1) stays on the LEFT; others go right.
            # Putting all on "right" used to collapse the price subplot area.
            layout_updates[yax] = dict(
                title=dict(text=y_titles.get(r, ""), font=dict(size=10)),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=False,
                showline=True,
                linecolor="rgba(255,255,255,0.08)",
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikethickness=1,
                spikecolor="rgba(255,255,255,0.25)",
                side="left" if r == 1 else "right",   # ← KEY FIX
                autorange=True,                        # ← always auto-scale
                fixedrange=False,
            )

        fig.update_layout(
            **layout_updates,
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=15, color="white"),
                x=0.5, xanchor="center",
            ),
            template=self.theme,
            height=860 + max(0, n_rows - 4) * 110,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="rgba(15,23,42,0.92)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(size=11, color="white"),
            ),
            legend=dict(
                orientation="v",
                x=1.01, y=1,
                xanchor="left",
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="rgba(255,255,255,0.15)",
                borderwidth=1,
                font=dict(size=10),
                tracegroupgap=4,
            ),
            margin=dict(l=60, r=160, t=90, b=60),
            plot_bgcolor="rgba(10,15,25,1)",
            paper_bgcolor="rgba(10,15,25,1)",
            # Global spike config
            spikedistance=999,
            hoverdistance=50,
        )

    # ── Static PNG fallback ───────────────────────────────────────────────
    def _export_static(self, fig, png_path: str):
        try:
            fig.write_image(png_path, width=1800, height=900, scale=1.5)
            logger.info(f"[Viz] Static PNG → {png_path}")
        except Exception as e:
            logger.warning(f"[Viz] PNG export skipped (pip install kaleido): {e}")
