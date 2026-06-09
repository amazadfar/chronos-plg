"""
Meta-model stacking Chronos quantile forecasts with tabular features.

Phase 7 behavior:
- Stage-2 training uses out-of-fold Chronos predictions (OOF), not in-sample fits.
- Final Chronos model is fitted on full training data only after OOF generation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.baselines.random_walk import BaselineModel
from src.models.chronos2_runner import Chronos2ForReturns, Chronos2Runner

logger = logging.getLogger(__name__)


class MetaModel(BaselineModel):
    """Two-stage model: Chronos quantiles + LightGBM quantile regressors."""

    def __init__(
        self,
        chronos_model: Chronos2Runner | None = None,
        feature_columns: list[str] | None = None,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        lgb_params: dict[str, Any] | None = None,
        n_estimators: int = 300,
        early_stopping_rounds: int = 50,
        use_uncertainty_features: bool = True,
        oof_splits: int = 5,
        oof_min_train_samples: int = 320,
    ):
        self.chronos_model = chronos_model
        self.feature_columns = feature_columns
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.use_uncertainty_features = use_uncertainty_features
        self.oof_splits = oof_splits
        self.oof_min_train_samples = oof_min_train_samples

        self.lgb_params = lgb_params or {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1,
        }

        self._lgb_models: dict[str, lgb.Booster] = {}
        self._meta_feature_names: list[str] = []
        self._chronos_oof_coverage: float = 0.0
        self._fitted = False
        self._chronos_fitted = False

    @property
    def name(self) -> str:
        return "MetaModel(Chronos2+LightGBM)"

    def _ensure_chronos_template(self) -> Chronos2Runner:
        if self.chronos_model is None:
            self.chronos_model = Chronos2ForReturns(
                context_length=256,
                device="auto",
            )
        return self.chronos_model

    def _clone_chronos_unfitted(self) -> Chronos2Runner:
        model = self._ensure_chronos_template()
        if hasattr(model, "clone_unfitted"):
            return model.clone_unfitted()
        return copy.deepcopy(model)

    def _get_chronos_features(
        self,
        chronos_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        features = pd.DataFrame(index=chronos_preds.index)

        for q in self.quantiles:
            col = f"q{int(q * 100)}"
            if col in chronos_preds.columns:
                features[f"chronos_{col}"] = chronos_preds[col]

        if self.use_uncertainty_features:
            if "q90" in chronos_preds.columns and "q10" in chronos_preds.columns:
                width = chronos_preds["q90"] - chronos_preds["q10"]
                features["chronos_interval_width"] = width
                if "q50" in chronos_preds.columns:
                    features["chronos_interval_width_pct"] = width / (chronos_preds["q50"].abs() + 1e-8)

            if {"q10", "q50", "q90"}.issubset(chronos_preds.columns):
                q10 = chronos_preds["q10"]
                q50 = chronos_preds["q50"]
                q90 = chronos_preds["q90"]
                features["chronos_skew"] = (q90 - q50) - (q50 - q10)
                features["chronos_signal_strength"] = q50.abs()
                features["chronos_confidence"] = (
                    features["chronos_signal_strength"] / (features.get("chronos_interval_width", 0.0) + 1e-8)
                )

        return features

    def _prepare_raw_features(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns:
            cols = [col for col in self.feature_columns if col in x.columns]
            return x[cols].copy()

        exclude_tokens = ("forward_", "regime", "hist_q", "timestamp")
        exclude_exact = {"open", "high", "low", "close", "volume"}

        cols = [
            col
            for col in x.columns
            if pd.api.types.is_numeric_dtype(x[col])
            and col not in exclude_exact
            and not any(token in col for token in exclude_tokens)
        ]
        return x[cols].copy()

    def _prepare_meta_features(
        self,
        x: pd.DataFrame,
        chronos_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        raw = self._prepare_raw_features(x)
        chronos_features = self._get_chronos_features(chronos_preds)
        return pd.concat([raw, chronos_features], axis=1)

    def _build_oof_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if n_samples <= self.oof_min_train_samples + 8:
            return []

        n_splits = max(2, self.oof_splits)
        remaining = n_samples - self.oof_min_train_samples
        fold_size = max(8, remaining // n_splits)

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        train_end = self.oof_min_train_samples
        while train_end < n_samples:
            val_end = min(n_samples, train_end + fold_size)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))
            train_end = val_end
        return splits

    def _generate_oof_chronos_predictions(
        self,
        x: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        cols = [f"q{int(q * 100)}" for q in self.quantiles]
        oof = pd.DataFrame(index=x.index, columns=cols, dtype=float)

        splits = self._build_oof_splits(len(x))
        if not splits:
            logger.warning(
                "Insufficient samples for OOF Chronos splits. Falling back to single holdout-style split."
            )
            split_at = max(32, len(x) // 2)
            splits = [(np.arange(0, split_at), np.arange(split_at, len(x)))]

        for fold_id, (train_idx, val_idx) in enumerate(splits):
            x_tr = x.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            x_val = x.iloc[val_idx]
            if x_val.empty:
                continue

            model = self._clone_chronos_unfitted()
            model.fit(x_tr, y_tr)
            preds = model.predict(x_val)
            oof.loc[x_val.index, cols] = preds.reindex(x_val.index)[cols]
            logger.info(
                "OOF Chronos fold=%s train=%s val=%s",
                fold_id,
                len(x_tr),
                len(x_val),
            )

        self._chronos_oof_coverage = float(oof["q50"].notna().mean()) if "q50" in oof.columns else 0.0
        return oof

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> MetaModel:
        if len(x) != len(y):
            raise ValueError("X and y must have matching lengths")

        oof_chronos = self._generate_oof_chronos_predictions(x, y)
        meta_features = self._prepare_meta_features(x, oof_chronos)

        valid_mask = ~(meta_features.isna().any(axis=1) | y.isna())
        x_meta = meta_features.loc[valid_mask]
        y_meta = y.loc[valid_mask]
        if x_meta.empty:
            raise ValueError("No valid meta-model training samples after OOF filtering")

        self._meta_feature_names = list(x_meta.columns)
        logger.info(
            "Meta-model OOF coverage=%.1f%% samples=%s features=%s",
            100.0 * self._chronos_oof_coverage,
            len(x_meta),
            len(self._meta_feature_names),
        )

        # Fit final Chronos model on full training data for inference-time usage.
        self.chronos_model = self._clone_chronos_unfitted()
        self.chronos_model.fit(x, y)
        self._chronos_fitted = True

        if x_val is not None and y_val is not None:
            chronos_val = self.chronos_model.predict(x_val)
            x_v = self._prepare_meta_features(x_val, chronos_val)
            valid_val = ~(x_v.isna().any(axis=1) | y_val.isna())
            x_v = x_v.loc[valid_val]
            y_v = y_val.loc[valid_val]
            x_train = x_meta
            y_train = y_meta
        else:
            split = max(1, int(len(x_meta) * 0.8))
            x_train = x_meta.iloc[:split]
            y_train = y_meta.iloc[:split]
            x_v = x_meta.iloc[split:]
            y_v = y_meta.iloc[split:]

        has_validation = len(x_v) > 32 and len(y_v) > 32

        for q in self.quantiles:
            col = f"q{int(q * 100)}"
            params = self.lgb_params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q

            train_data = lgb.Dataset(x_train, label=y_train)
            callbacks = [lgb.log_evaluation(period=0)]
            valid_sets = None

            if has_validation:
                val_data = lgb.Dataset(x_v, label=y_v, reference=train_data)
                valid_sets = [val_data]
                callbacks.insert(0, lgb.early_stopping(self.early_stopping_rounds, verbose=False))

            model = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
            self._lgb_models[col] = model

        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or not self._chronos_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        chronos_preds = self.chronos_model.predict(x)
        meta = self._prepare_meta_features(x, chronos_preds)

        for col in self._meta_feature_names:
            if col not in meta.columns:
                meta[col] = 0.0
        meta = meta[self._meta_feature_names].fillna(0.0)

        out = pd.DataFrame(index=x.index)
        for q in self.quantiles:
            col = f"q{int(q * 100)}"
            if col in self._lgb_models:
                out[col] = self._lgb_models[col].predict(meta)
            else:
                out[col] = chronos_preds.get(col, 0.0)
        return out

    def get_feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Model not fitted.")
        rows = {"feature": self._meta_feature_names}
        for q in self.quantiles:
            col = f"q{int(q * 100)}"
            if col in self._lgb_models:
                rows[col] = self._lgb_models[col].feature_importance(importance_type)
        df = pd.DataFrame(rows)
        qcols = [f"q{int(q * 100)}" for q in self.quantiles if f"q{int(q * 100)}" in df.columns]
        df["mean_importance"] = df[qcols].mean(axis=1) if qcols else 0.0
        return df.sort_values("mean_importance", ascending=False)
