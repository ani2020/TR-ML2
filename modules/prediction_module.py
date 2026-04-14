"""
Prediction Module - XGBoost-based return prediction.
Integrates HMM regime labels and GARCH volatility as features.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

# Default feature columns used for XGBoost
DEFAULT_FEATURE_COLS = [
    "returns",
    "log_returns",
    "volatility_20",
    "momentum_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx",
    "volume_ratio",
    "bb_width",
    "price_above_ema50",
    "regime_state",
    "garch_vol",
    # ── NEW FEATURES ──────────────────── ← these four lines
    "atr_14",            # raw ATR in price units
    "atr_pct",           # normalised ATR (cross-asset comparable)
    "vwap_20",           # VWAP price level
    "vwap_deviation",    # % deviation from VWAP (mean-reversion signal)    
]


class PredictionModule:
    """
    XGBoost classifier: predicts whether next-day return is positive (1) or negative (0).

    Integrates:
      - HMM regime features (regime_state, regime_prob_*)
      - GARCH volatility (garch_vol)
      - Technical indicators

    Output columns added to DataFrame:
      - xgb_pred        : 0 or 1
      - xgb_prob_up     : probability of positive return
      - xgb_confidence  : abs(prob_up - 0.5) * 2  → [0,1]
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        target_lookahead: int = 1,
    ):
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self.target_lookahead = target_lookahead
        self._model = None
        self._used_features: List[str] = []
        self._is_fitted = False

    # ── Feature Selection ─────────────────────
    def _select_features(self, df: pd.DataFrame) -> List[str]:
        available = [c for c in self.feature_cols if c in df.columns]
        # Also include any regime_prob_ columns
        prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
        all_feats = list(dict.fromkeys(available + prob_cols))
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            logger.debug(f"[XGB] Feature cols not in DataFrame: {missing}")
        logger.info(f"[XGB] Using {len(all_feats)} features: {all_feats}")
        return all_feats

    # ── Build Target ──────────────────────────
    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        """Binary target: 1 if future return > 0 else 0."""
        future_ret = df["returns"].shift(-self.target_lookahead)
        target = (future_ret > 0).astype(int)
        return target

    # ── Training ──────────────────────────────
    def fit(self, df: pd.DataFrame) -> "PredictionModule":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        self._used_features = self._select_features(df)
        target = self._build_target(df)

        # Drop rows where target is NaN (last lookahead rows)
        valid_idx = target.dropna().index
        X = df.loc[valid_idx, self._used_features].fillna(0)
        y = target.loc[valid_idx].astype(int)

        logger.info(f"[XGB] Training on {len(X)} samples | target balance: {y.mean():.2%} positive")

        params = {k: v for k, v in self.xgb_params.items() if k != "use_label_encoder"}
        self._model = XGBClassifier(**params)
        self._model.fit(X, y)

        # Feature importance log
        importances = pd.Series(
            self._model.feature_importances_, index=self._used_features
        ).sort_values(ascending=False)
        logger.info("[XGB] Top 10 feature importances:")
        for feat, imp in importances.head(10).items():
            logger.info(f"  {feat:30s}: {imp:.4f}")

        self._is_fitted = True
        return self

    # ── Prediction ────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = df[self._used_features].fillna(0)
        preds = self._model.predict(X)
        probs = self._model.predict_proba(X)[:, 1]

        out = df.copy()
        out["xgb_pred"] = preds
        out["xgb_prob_up"] = probs
        out["xgb_confidence"] = (np.abs(probs - 0.5) * 2).clip(0, 1)

        up_pct = (preds == 1).mean()
        logger.info(f"[XGB] Predictions: {up_pct:.1%} bullish, mean confidence: {out['xgb_confidence'].mean():.3f}")
        return out

    @property
    def feature_importances(self) -> Optional[pd.Series]:
        if self._model is None:
            return None
        return pd.Series(
            self._model.feature_importances_, index=self._used_features
        ).sort_values(ascending=False)

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def params(self):
        return self.xgb_params
