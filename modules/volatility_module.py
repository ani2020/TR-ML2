"""
Volatility Module - GARCH(1,1) model for forecasting next-period volatility.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class GARCHVolatilityModule:
    """
    GARCH(1,1) volatility model.

    Model: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Forecasts one-step-ahead conditional variance.

    Uses arch library if available, otherwise falls back to a
    pure-numpy rolling GARCH(1,1) MLE implementation.
    """

    def __init__(self, p: int = 1, q: int = 1, annualize: bool = True,
                 trading_days: int = 252):
        self.p = p
        self.q = q
        self.annualize = annualize
        self.trading_days = trading_days
        self._fitted_model = None
        self._omega: Optional[float] = None
        self._alpha: Optional[float] = None
        self._beta: Optional[float] = None
        self._last_sigma2: Optional[float] = None
        self._last_resid: Optional[float] = None
        self._is_fitted = False
        self._use_arch = self._check_arch()

    def _check_arch(self) -> bool:
        try:
            import arch
            return True
        except ImportError:
            logger.warning("[GARCH] 'arch' library not found. Using numpy fallback.")
            return False

    # ── Fitting ───────────────────────────────
    def fit(self, returns: pd.Series) -> "GARCHVolatilityModule":
        """Fit GARCH(1,1) on a return series (fractional, e.g. 0.01 = 1%)."""
        ret = returns.dropna() * 100  # convert to % for numerical stability

        if self._use_arch:
            self._fit_arch(ret)
        else:
            self._fit_numpy(ret)

        self._is_fitted = True
        logger.info(
            f"[GARCH] Fitted — ω={self._omega:.6f}, α={self._alpha:.4f}, β={self._beta:.4f} "
            f"(α+β={self._alpha+self._beta:.4f})"
        )
        if self._alpha + self._beta >= 1.0:
            logger.warning("[GARCH] α+β ≥ 1 — process is non-stationary (IGARCH territory)")
        return self

    def _fit_arch(self, ret: pd.Series):
        from arch import arch_model
        am = arch_model(ret, vol="Garch", p=self.p, q=self.q, dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        self._fitted_model = res
        params = res.params
        self._omega = float(params.get("omega", params.iloc[1]))
        self._alpha = float(params.get("alpha[1]", params.iloc[2]))
        self._beta = float(params.get("beta[1]", params.iloc[3]))
        # Last conditional variance
        self._last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
        self._last_resid = float(ret.iloc[-1] - res.params.get("mu", 0))

    def _fit_numpy(self, ret: pd.Series):
        """Simple MLE GARCH(1,1) via scipy optimize."""
        from scipy.optimize import minimize

        r = ret.values
        n = len(r)
        var_target = np.var(r)

        def neg_loglik(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            sigma2 = np.full(n, var_target)
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r**2 / sigma2)
            return -ll

        x0 = [var_target * 0.1, 0.1, 0.8]
        bounds = [(1e-8, None), (1e-8, 1-1e-8), (1e-8, 1-1e-8)]
        res = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=bounds)
        self._omega, self._alpha, self._beta = res.x

        # Compute last sigma2
        sigma2 = np.full(n, var_target)
        for t in range(1, n):
            sigma2[t] = self._omega + self._alpha * r[t-1]**2 + self._beta * sigma2[t-1]
        self._last_sigma2 = sigma2[-1]
        self._last_resid = r[-1]

    # ── Forecasting ───────────────────────────
    def forecast_next(self) -> float:
        """Forecast next period conditional volatility (annualized if configured)."""
        if not self._is_fitted:
            raise RuntimeError("GARCH model not fitted.")
        next_sigma2 = (
            self._omega
            + self._alpha * self._last_resid**2
            + self._beta * self._last_sigma2
        )
        # Convert from % units back to fractional
        next_vol = np.sqrt(next_sigma2) / 100.0
        if self.annualize:
            next_vol *= np.sqrt(self.trading_days)
        logger.debug(f"[GARCH] Next-period forecast vol: {next_vol:.4f}")
        return next_vol

    def predict_series(self, returns: pd.Series) -> pd.Series:
        """
        Generate in-sample conditional volatility series.
        Returns annualized volatility for each bar.
        """
        if not self._is_fitted:
            raise RuntimeError("GARCH model not fitted.")

        ret = returns.dropna() * 100
        r = ret.values
        n = len(r)
        var_target = np.var(r)

        if self._use_arch and self._fitted_model is not None:
            cv = self._fitted_model.conditional_volatility.values / 100.0
        else:
            sigma2 = np.full(n, var_target)
            for t in range(1, n):
                sigma2[t] = self._omega + self._alpha * r[t-1]**2 + self._beta * sigma2[t-1]
            cv = np.sqrt(sigma2) / 100.0

        if self.annualize:
            cv = cv * np.sqrt(self.trading_days)

        return pd.Series(cv, index=ret.index, name="garch_vol")

    def add_to_dataframe(self, df: pd.DataFrame,
                         returns_col: str = "returns") -> pd.DataFrame:
        """
        Fit GARCH on a return series and attach garch_vol + garch_next_vol.

        Parameters
        ----------
        df          : Feature DataFrame.
        returns_col : Column to fit GARCH on. Use 'fut_log_ret' when working
                      with the continuous futures series (recommended — futures
                      returns reflect actual traded volatility without roll gaps).
                      Defaults to 'returns' for backward compatibility.
        """
        df = df.copy()

        # Resolve column: fall back gracefully with a warning if not found
        if returns_col not in df.columns:
            fallback = next(
                (c for c in ("returns", "log_returns", "fut_log_ret") if c in df.columns),
                None
            )
            if fallback is None:
                raise ValueError(
                    f"[GARCH] '{returns_col}' not found and no fallback available. "
                    f"DataFrame columns: {list(df.columns)}"
                )
            logger.warning(
                f"[GARCH] '{returns_col}' not found — falling back to '{fallback}'. "
                f"Ensure futures_features has been computed before GARCH."
            )
            returns_col = fallback

        logger.info(f"[GARCH] Fitting on column: '{returns_col}'")
        self.fit(df[returns_col])
        garch_series = self.predict_series(df[returns_col])

        # Drop existing garch columns so join never raises overlap error
        for _col in ["garch_vol", "garch_next_vol"]:
            if _col in df.columns:
                df.drop(columns=[_col], inplace=True)
        df = df.join(garch_series, how="left")
        df["garch_next_vol"] = self.forecast_next()
        logger.info(
            f"[GARCH] Added garch_vol (fitted on '{returns_col}'). "
            f"Next-period vol: {df['garch_next_vol'].iloc[-1]:.4f}"
        )
        return df

    @property
    def params(self) -> dict:
        return {
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "persistence": (self._alpha or 0) + (self._beta or 0),
        }

    @property
    def is_fitted(self):
        return self._is_fitted
