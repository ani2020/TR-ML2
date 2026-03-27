"""
models/volatility_model.py
GARCH(1,1) volatility model for next-period volatility forecasting.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("trading.volatility")


class VolatilityModel:
    """
    GARCH(1,1) wrapper using the `arch` library.

    Usage
    -----
    vm = VolatilityModel(config)
    vm.fit(train_df)
    forecast_vol = vm.forecast(test_df)  # Series of next-period annualised vol
    """

    def __init__(self, config):
        self.cfg = config.garch
        self._fit_result = None
        self._last_train_returns: Optional[pd.Series] = None

    # ── training ─────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "VolatilityModel":
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("arch not installed. Run: pip install arch")

        if "log_return" not in df.columns:
            raise ValueError("DataFrame must contain 'log_return' column.")

        returns = df["log_return"].dropna() * 100   # arch expects %-scaled returns

        logger.info(f"[VolatilityModel] Fitting GARCH({self.cfg.p},{self.cfg.q}) "
                    f"on {len(returns)} observations …")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                returns,
                vol=self.cfg.vol,
                p=self.cfg.p,
                q=self.cfg.q,
                dist=self.cfg.dist,
                mean=self.cfg.mean,
            )
            self._fit_result = model.fit(disp="off", show_warning=False)

        logger.info("[VolatilityModel] GARCH fit complete.")
        logger.debug("\n" + str(self._fit_result.summary()))
        self._last_train_returns = returns
        return self

    # ── forecasting ──────────────────────────────────────────
    def forecast_series(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a Series of next-period annualised volatility forecasts
        aligned to df.index, one step ahead, using a rolling re-forecast
        approach (computationally feasible for daily data).
        """
        if self._fit_result is None:
            raise RuntimeError("VolatilityModel not fitted. Call fit() first.")

        from arch import arch_model

        all_returns = pd.concat([
            self._last_train_returns,
            df["log_return"].dropna() * 100
        ])

        forecasts = []
        params = self._fit_result.params

        # Walk-forward: for each test date refit is expensive;
        # use the trained parameters to get conditional variances instead.
        try:
            res = self._fit_result.model.fit(
                starting_values=params.values,
                first_obs=0,
                last_obs=len(all_returns),
                disp="off",
                show_warning=False,
            )
            cond_vol = res.conditional_volatility
            # Annualise: daily vol * sqrt(252)
            ann_vol = (cond_vol / 100) * np.sqrt(252)
            full_index = all_returns.index
            ann_series = pd.Series(ann_vol.values, index=full_index)
            forecast_vol = ann_series.loc[df.index].shift(1)   # next-period forecast
        except Exception as e:
            logger.warning(f"[VolatilityModel] Rolling forecast failed ({e}); "
                           "falling back to constant forecast.")
            fcast = self._fit_result.forecast(horizon=1, reindex=False)
            last_var = fcast.variance.values[-1, 0]
            last_vol_ann = (np.sqrt(last_var) / 100) * np.sqrt(252)
            forecast_vol = pd.Series(last_vol_ann, index=df.index)

        forecast_vol.name = "garch_vol_forecast"
        logger.info(f"[VolatilityModel] Forecast vol: "
                    f"mean={forecast_vol.mean():.4f}  "
                    f"min={forecast_vol.min():.4f}  "
                    f"max={forecast_vol.max():.4f}")
        return forecast_vol

    def add_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'garch_vol_forecast' column to df and return."""
        df = df.copy()
        df["garch_vol_forecast"] = self.forecast_series(df)
        return df
