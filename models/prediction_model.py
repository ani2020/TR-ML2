"""
models/prediction_model.py
XGBoost directional prediction model.

Integration chain:
  FeatureEngine → RegimeModel → VolatilityModel → PredictionModel
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

logger = logging.getLogger("trading.prediction")


# ── Feature selector for XGBoost input ───────────────────────
PREDICTION_FEATURES = [
    # Returns & momentum
    "log_return", "momentum_5d", "momentum_10d", "momentum_20d",
    # Volatility
    "volatility_5d", "volatility_10d", "volatility_20d", "garch_vol_forecast",
    # Volume
    "volume_ratio", "volume_zscore",
    # Technical
    "rsi_14", "macd", "macd_hist", "adx_14",
    "bb_pct_b", "bb_width",
    # Price structure
    "high_low_range", "close_open_range",
    # EMA ratios
    "ema_20", "ema_50",
    # HMM regime (one-hot encoded later)
    "hmm_state", "regime_confidence",
]


class PredictionModel:
    """
    XGBoost binary classifier: predicts whether the close price
    will be higher in `horizon` trading days.
    """

    def __init__(self, config):
        self.cfg = config.xgb
        self._model = None
        self._scaler = StandardScaler()
        self._feature_cols: List[str] = []
        self._importance: Optional[pd.Series] = None

    # ── feature preparation ──────────────────────────────────
    def _prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Select, encode, and scale feature matrix."""
        available = [c for c in PREDICTION_FEATURES if c in df.columns]

        # Add regime one-hot columns if present
        regime_dummies = pd.DataFrame(index=df.index)
        if "hmm_regime" in df.columns:
            dummies = pd.get_dummies(df["hmm_regime"], prefix="regime")
            regime_dummies = dummies

        feat_df = pd.concat([df[available], regime_dummies], axis=1)
        feat_df.fillna(method="ffill", inplace=True)
        feat_df.fillna(0, inplace=True)

        if fit_scaler:
            self._feature_cols = list(feat_df.columns)
            X = self._scaler.fit_transform(feat_df.values)
        else:
            # Align columns to training set
            for col in self._feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df[self._feature_cols]
            X = self._scaler.transform(feat_df.values)
        return X

    def _build_target(self, df: pd.DataFrame) -> pd.Series:
        """1 if price rises in `horizon` days, else 0."""
        future_return = df["Close"].shift(-self.cfg.horizon) / df["Close"] - 1
        target = (future_return > self.cfg.target_threshold).astype(int)
        return target

    # ── training ─────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "PredictionModel":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        logger.info(f"[PredictionModel] Preparing training data ({len(df)} rows) …")
        y = self._build_target(df)
        X = self._prepare_features(df, fit_scaler=True)

        # Align and drop NaN rows
        valid = ~np.isnan(X).any(axis=1) & ~y.isna().values
        valid[-self.cfg.horizon:] = False   # drop last horizon rows (no target)
        X_train = X[valid]
        y_train = y.values[valid]

        logger.info(f"[PredictionModel] Training XGBoost on {X_train.shape[0]} samples, "
                    f"{X_train.shape[1]} features …")

        self._model = XGBClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            subsample=self.cfg.subsample,
            colsample_bytree=self.cfg.colsample_bytree,
            min_child_weight=self.cfg.min_child_weight,
            gamma=self.cfg.gamma,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            eval_metric=self.cfg.eval_metric,
            use_label_encoder=False,
        )

        # Time-series cross-validation for early stopping
        tscv = TimeSeriesSplit(n_splits=3)
        splits = list(tscv.split(X_train))
        tr_idx, val_idx = splits[-1]
        eval_set = [(X_train[val_idx], y_train[val_idx])]

        self._model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=eval_set,
            verbose=False,
        )

        # Feature importance
        self._importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)

        # In-sample metrics
        y_pred = self._model.predict(X_train)
        y_prob = self._model.predict_proba(X_train)[:, 1]
        acc = accuracy_score(y_train, y_pred)
        try:
            auc = roc_auc_score(y_train, y_prob)
        except Exception:
            auc = float("nan")

        logger.info(f"[PredictionModel] Train Accuracy={acc:.4f}  AUC={auc:.4f}")
        logger.info("[PredictionModel] Top-10 features:\n" +
                    self._importance.head(10).to_string())
        return self

    # ── prediction ───────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("PredictionModel not trained.")

        X = self._prepare_features(df, fit_scaler=False)
        valid = ~np.isnan(X).any(axis=1)

        proba = np.full(len(df), np.nan)
        direction = np.full(len(df), np.nan)

        proba[valid] = self._model.predict_proba(X[valid])[:, 1]
        direction[valid] = self._model.predict(X[valid])

        df = df.copy()
        df["xgb_prob_up"] = proba
        df["xgb_direction"] = direction
        df["xgb_confidence"] = np.abs(proba - 0.5) * 2   # 0 = uncertain, 1 = confident

        logger.info(f"[PredictionModel] Predictions: "
                    f"up={np.nansum(direction==1):.0f}  "
                    f"down={np.nansum(direction==0):.0f}  "
                    f"mean_prob={np.nanmean(proba):.4f}")
        return df

    @property
    def feature_importance(self) -> Optional[pd.Series]:
        return self._importance
