"""
Market Regime Module - Hidden Markov Model based regime detection.
Identifies up to 7 hidden states (Bull, Bear, Crash, Sideways, etc.)
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Regime Labels
# ─────────────────────────────────────────────
REGIME_LABELS = {
    0: "Bear",
    1: "Sideways_Bear",
    2: "Sideways",
    3: "Sideways_Bull",
    4: "Bull",
    5: "Strong_Bull",
    6: "Crash",
}

REGIME_COLORS = {
    "Bear":           "#d32f2f",
    "Sideways_Bear":  "#ef9a9a",
    "Sideways":       "#b0bec5",
    "Sideways_Bull":  "#a5d6a7",
    "Bull":           "#2e7d32",
    "Strong_Bull":    "#1b5e20",
    "Crash":          "#4a148c",
}

BULLISH_REGIMES = {"Bull", "Strong_Bull", "Sideways_Bull"}
BEARISH_REGIMES = {"Bear", "Crash"}


# ─────────────────────────────────────────────
# Feature Plugin Registry for HMM
# ─────────────────────────────────────────────
class HMMFeatureRegistry:
    """Plug-and-play feature registry for HMM training."""
    _registry: Dict[str, callable] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def get_features(cls, df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        arrays = []
        for name in feature_names:
            if name in cls._registry:
                arr = cls._registry[name](df)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                arrays.append(arr)
            elif name in df.columns:
                arrays.append(df[name].values.reshape(-1, 1))
            else:
                logger.warning(f"[HMM] Feature not found: {name}")
        if not arrays:
            raise ValueError("No valid features found for HMM training.")
        return np.hstack(arrays)

    @classmethod
    def available(cls):
        return list(cls._registry.keys())


@HMMFeatureRegistry.register("log_returns")
def _feat_log_ret(df):
    return df["log_returns"].fillna(0).values

@HMMFeatureRegistry.register("volatility_20")
def _feat_vol(df):
    return df["volatility_20"].fillna(0).values

@HMMFeatureRegistry.register("volume_ratio")
def _feat_volratio(df):
    return df["volume_ratio"].fillna(1).values

@HMMFeatureRegistry.register("momentum_10")
def _feat_mom(df):
    return df["momentum_10"].fillna(0).values

@HMMFeatureRegistry.register("rsi_14")
def _feat_rsi(df):
    return (df["rsi_14"].fillna(50).values / 100.0)

@HMMFeatureRegistry.register("adx")
def _feat_adx(df):
    return (df["adx"].fillna(25).values / 100.0)


# ─────────────────────────────────────────────
# Market Regime Module
# ─────────────────────────────────────────────
class MarketRegimeModule:
    """
    Detects market regimes using a Gaussian Hidden Markov Model.

    Pipeline:
      1. Extract features from DataFrame
      2. Fit GaussianHMM with n_states
      3. Predict hidden state sequence
      4. Label states by mean return (Bull = highest, Crash = lowest, etc.)
      5. Calculate per-bar state probabilities
    """

    DEFAULT_FEATURES = [
        "log_returns",
        "volatility_20",
        "volume_ratio",
        "momentum_10",
        "rsi_14",
    ]

    def __init__(
        self,
        n_states: int = 5,
        n_iter: int = 200,
        covariance_type: str = "full",
        features: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        assert 2 <= n_states <= 7, "n_states must be between 2 and 7"
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.features = features or self.DEFAULT_FEATURES
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.scaler = StandardScaler()
        self.state_label_map: Dict[int, str] = {}
        self.label_state_map: Dict[str, int] = {}
        self._is_fitted = False

    # ── Training ──────────────────────────────
    def fit(self, df: pd.DataFrame) -> "MarketRegimeModule":
        logger.info(f"[HMM] Training with {self.n_states} states on {len(df)} bars")
        logger.info(f"[HMM] Features: {self.features}")

        X = HMMFeatureRegistry.get_features(df, self.features)
        X_scaled = self.scaler.fit_transform(X)

        # lengths parameter: single continuous sequence
        lengths = [len(X_scaled)]

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        self.model.fit(X_scaled, lengths)
        logger.info(f"[HMM] Converged: {self.model.monitor_.converged}")

        # Label states
        self._label_states(df, X_scaled)
        self._is_fitted = True

        # Log transition matrix
        self._log_transition_matrix()
        return self

    def _label_states(self, df: pd.DataFrame, X_scaled: np.ndarray):
        """
        Label hidden states by mean log return.
        States are sorted: lowest return → highest return.
        Special: state with lowest return AND high volatility → Crash.
        """
        hidden_states = self.model.predict(X_scaled)
        log_ret = df["log_returns"].fillna(0).values
        vol = df["volatility_20"].fillna(0).values if "volatility_20" in df.columns else np.zeros(len(df))

        state_means = {}
        state_vols = {}
        for s in range(self.n_states):
            mask = hidden_states == s
            state_means[s] = log_ret[mask].mean() if mask.sum() > 0 else 0.0
            state_vols[s] = vol[mask].mean() if mask.sum() > 0 else 0.0

        # Sort states by mean return
        sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])

        # Build labels based on position in sorted order
        n = self.n_states
        label_sequence = []
        if n == 2:
            label_sequence = ["Bear", "Bull"]
        elif n == 3:
            label_sequence = ["Bear", "Sideways", "Bull"]
        elif n == 4:
            label_sequence = ["Bear", "Sideways_Bear", "Sideways_Bull", "Bull"]
        elif n == 5:
            label_sequence = ["Crash", "Bear", "Sideways", "Bull", "Strong_Bull"]
        elif n == 6:
            label_sequence = ["Crash", "Bear", "Sideways_Bear", "Sideways_Bull", "Bull", "Strong_Bull"]
        elif n == 7:
            label_sequence = ["Crash", "Bear", "Sideways_Bear", "Sideways", "Sideways_Bull", "Bull", "Strong_Bull"]

        # Override: lowest-return state with above-median volatility → Crash
        lowest_state = sorted_states[0]
        median_vol = np.median(list(state_vols.values()))
        if state_vols[lowest_state] < median_vol and n >= 5:
            # Demote crash label to bear if not really volatile
            label_sequence[0] = "Bear"

        self.state_label_map = {}
        for idx, state in enumerate(sorted_states):
            self.state_label_map[state] = label_sequence[idx]

        self.label_state_map = {v: k for k, v in self.state_label_map.items()}

        logger.info("[HMM] State label mapping:")
        for s, label in self.state_label_map.items():
            logger.info(
                f"  State {s} → {label:15s} | mean_return={state_means[s]:+.4f} | mean_vol={state_vols[s]:.4f}"
            )

    def _log_transition_matrix(self):
        logger.info("[HMM] Transition Matrix (rows=from, cols=to):")
        header = "       " + "  ".join(f"S{i}" for i in range(self.n_states))
        logger.info(header)
        for i, row in enumerate(self.model.transmat_):
            vals = "  ".join(f"{v:.3f}" for v in row)
            logger.info(f"  S{i} →  {vals}  ({self.state_label_map.get(i, '?')})")

    # ── Prediction ────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict regime for each bar.
        Returns input df with additional columns:
          - regime_state (int)
          - regime_label (str)
          - regime_is_bullish (bool)
          - regime_prob_* (float, one per state)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = HMMFeatureRegistry.get_features(df, self.features)
        X_scaled = self.scaler.transform(X)

        hidden_states = self.model.predict(X_scaled)
        state_probs = self.model.predict_proba(X_scaled)

        out = df.copy()
        out["regime_state"] = hidden_states
        out["regime_label"] = [self.state_label_map.get(s, "Unknown") for s in hidden_states]
        out["regime_is_bullish"] = out["regime_label"].isin(BULLISH_REGIMES)
        out["regime_is_bearish"] = out["regime_label"].isin(BEARISH_REGIMES)

        # Per-state probabilities
        for s in range(self.n_states):
            label = self.state_label_map.get(s, f"state_{s}")
            out[f"regime_prob_{label}"] = state_probs[:, s]

        # Identify Bull Run and Bear Run windows
        out = self._tag_runs(out)

        self._log_regime_summary(out)
        return out

    def _tag_runs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify the Bull Run (highest cumulative positive return window)
        and Bear Run (lowest return window) in the predicted data.
        """
        if "returns" not in df.columns:
            return df

        df = df.copy()
        df["is_bull_run"] = False
        df["is_bear_run"] = False

        # Group consecutive bull/bear regimes
        df["_regime_group"] = (df["regime_label"] != df["regime_label"].shift()).cumsum()
        best_bull_return = -np.inf
        best_bear_return = np.inf
        best_bull_group = None
        best_bear_group = None

        for grp_id, grp in df.groupby("_regime_group"):
            label = grp["regime_label"].iloc[0]
            cum_ret = grp["returns"].sum()
            if label in BULLISH_REGIMES and cum_ret > best_bull_return:
                best_bull_return = cum_ret
                best_bull_group = grp_id
            if label in BEARISH_REGIMES and cum_ret < best_bear_return:
                best_bear_return = cum_ret
                best_bear_group = grp_id

        if best_bull_group is not None:
            df.loc[df["_regime_group"] == best_bull_group, "is_bull_run"] = True
            start = df[df["_regime_group"] == best_bull_group].index[0]
            end = df[df["_regime_group"] == best_bull_group].index[-1]
            logger.info(f"[HMM] 🐂 Bull Run: {start.date()} → {end.date()} | cumulative return: {best_bull_return:+.2%}")

        if best_bear_group is not None:
            df.loc[df["_regime_group"] == best_bear_group, "is_bear_run"] = True
            start = df[df["_regime_group"] == best_bear_group].index[0]
            end = df[df["_regime_group"] == best_bear_group].index[-1]
            logger.info(f"[HMM] 🐻 Bear Run: {start.date()} → {end.date()} | cumulative return: {best_bear_return:+.2%}")

        df.drop(columns=["_regime_group"], inplace=True)
        return df

    def _log_regime_summary(self, df: pd.DataFrame):
        counts = df["regime_label"].value_counts()
        total = len(df)
        logger.info(f"[HMM] Regime distribution ({total} bars):")
        for label, cnt in counts.items():
            pct = cnt / total * 100
            logger.info(f"  {label:18s}: {cnt:5d} bars ({pct:.1f}%)")

    def get_state_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame of state probabilities per bar."""
        X = HMMFeatureRegistry.get_features(df, self.features)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        cols = [f"prob_{self.state_label_map.get(s, s)}" for s in range(self.n_states)]
        return pd.DataFrame(probs, index=df.index, columns=cols)

    @property
    def is_fitted(self):
        return self._is_fitted
