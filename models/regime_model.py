"""
models/regime_model.py
Market Regime Detection via Hidden Markov Model (HMM).

Identifies up to 7 hidden states and labels them:
  Bull Run, Bull, Sideways, Bear, Bear Run, High-Vol, Crash
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger("trading.regime")

# ── Regime label constants ────────────────────────────────────
REGIME_LABELS = {
    "bull_run":   "Bull Run",
    "bull":       "Bull",
    "sideways":   "Sideways",
    "bear":       "Bear",
    "bear_run":   "Bear Run",
    "high_vol":   "High Volatility",
    "crash":      "Crash",
}

# Colour map for visualisation
REGIME_COLOURS = {
    "Bull Run":        "#00b300",
    "Bull":            "#66cc66",
    "Sideways":        "#cccc00",
    "Bear":            "#ff8000",
    "Bear Run":        "#ff4400",
    "High Volatility": "#cc00cc",
    "Crash":           "#cc0000",
    "Unknown":         "#888888",
}


class RegimeModel:
    """
    Gaussian HMM-based market regime detector.

    Pipeline
    --------
    fit(df)  →  predict(df)  →  label_regimes(df)

    State assignment after training
    --------------------------------
    States are sorted by their mean *log_return* component.
    The top state becomes "Bull Run", the bottom "Bear Run" (or "Crash").
    """

    def __init__(self, config):
        self.cfg = config.hmm
        self._model: Optional[GaussianHMM] = None
        self._state_label_map: Dict[int, str] = {}
        self._state_means: Optional[np.ndarray] = None

    # ── feature extraction ───────────────────────────────────
    def _extract_X(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        missing = [f for f in self.cfg.features if f not in df.columns]
        if missing:
            raise ValueError(f"[RegimeModel] Missing features: {missing}")
        X = df[self.cfg.features].values.astype(float)
        # Winsorise at 1 / 99 percentile to reduce outlier sensitivity
        for col_idx in range(X.shape[1]):
            lo, hi = np.nanpercentile(X[:, col_idx], [1, 99])
            X[:, col_idx] = np.clip(X[:, col_idx], lo, hi)
        # Z-score normalise
        means = np.nanmean(X, axis=0)
        stds = np.nanstd(X, axis=0) + 1e-8
        X = (X - means) / stds
        return X, df.index

    # ── training ─────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "RegimeModel":
        logger.info(f"[RegimeModel] Training HMM: n_states={self.cfg.n_states}, "
                    f"features={self.cfg.features}")
        X, _ = self._extract_X(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = GaussianHMM(
                n_components=self.cfg.n_states,
                covariance_type=self.cfg.covariance_type,
                n_iter=self.cfg.n_iter,
                random_state=self.cfg.random_state,
            )
            self._model.fit(X)

        # ── auto-label states by mean log_return ─────────────
        return_idx = self.cfg.features.index("log_return") if "log_return" in self.cfg.features else 0
        # De-normalise mean returns for labelling
        raw_X, _ = self._extract_X(df)   # already normalised above; recompute unnormalised
        raw_unnorm = df[self.cfg.features].values.astype(float)
        state_returns = []
        states_all = self._model.predict(X)
        for s in range(self.cfg.n_states):
            mask = states_all == s
            if mask.sum() > 0:
                mean_ret = raw_unnorm[mask, return_idx].mean()
            else:
                mean_ret = 0.0
            state_returns.append((s, mean_ret))

        sorted_states = sorted(state_returns, key=lambda x: x[1], reverse=True)
        n = len(sorted_states)

        # Assign labels based on rank
        label_slots = self._get_label_slots(n)
        self._state_label_map = {}
        for rank, (state_id, mean_ret) in enumerate(sorted_states):
            label = label_slots[rank]
            self._state_label_map[state_id] = label
            logger.info(f"  State {state_id:2d} → '{label:18s}'  mean_log_return={mean_ret:.5f}")

        logger.info("[RegimeModel] Training complete.")
        logger.info("[RegimeModel] Transition matrix:\n" +
                    pd.DataFrame(self._model.transmat_).to_string())
        return self

    def _get_label_slots(self, n: int) -> List[str]:
        """Map rank positions to regime labels depending on n_states."""
        if n == 1:
            return ["Sideways"]
        if n == 2:
            return ["Bull", "Bear"]
        if n == 3:
            return ["Bull", "Sideways", "Bear"]
        if n == 4:
            return ["Bull Run", "Bull", "Bear", "Bear Run"]
        if n == 5:
            return ["Bull Run", "Bull", "Sideways", "Bear", "Bear Run"]
        if n == 6:
            return ["Bull Run", "Bull", "Sideways", "Bear", "Bear Run", "Crash"]
        # 7
        return ["Bull Run", "Bull", "Sideways", "High Volatility", "Bear", "Bear Run", "Crash"]

    # ── prediction ───────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("RegimeModel not trained. Call fit() first.")

        X, idx = self._extract_X(df)
        raw_states = self._model.predict(X)
        state_probs = self._model.predict_proba(X)

        df = df.copy()
        df["hmm_state"] = raw_states
        df["hmm_regime"] = [self._state_label_map.get(s, "Unknown") for s in raw_states]

        for s in range(self.cfg.n_states):
            label = self._state_label_map.get(s, f"state_{s}")
            col = f"prob_{label.replace(' ', '_').lower()}"
            df[col] = state_probs[:, s]

        df["regime_confidence"] = state_probs.max(axis=1)

        # ── identify Bull Run / Bear Run windows ──────────────
        df = self._annotate_runs(df)

        # ── log regime distribution ───────────────────────────
        dist = df["hmm_regime"].value_counts()
        logger.info("[RegimeModel] Regime distribution on prediction set:\n" + dist.to_string())

        return df

    def _annotate_runs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag the single longest contiguous Bull Run and Bear Run windows."""
        df["is_bull_run"] = df["hmm_regime"] == "Bull Run"
        df["is_bear_run"] = (df["hmm_regime"] == "Bear Run") | (df["hmm_regime"] == "Crash")
        return df

    def state_label_map(self) -> Dict[int, str]:
        return dict(self._state_label_map)

    def log_state_probabilities(self, df: pd.DataFrame):
        """Log detailed state probability table for manual review."""
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        tbl = df[["hmm_regime", "regime_confidence"] + prob_cols].tail(20)
        logger.info("[RegimeModel] Last 20 state probabilities:\n" + tbl.to_string())
