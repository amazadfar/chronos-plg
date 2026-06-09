"""
Chronos-2 model runner with strict rolling out-of-sample inference.

This module keeps the model API aligned with baseline models:
- ``fit(X, y)``
- ``predict(X) -> DataFrame[q10, q50, q90]``

Key safety behavior:
- ``predict`` never reads realized test-period targets.
- Forecasts are generated one timestamp at a time using only training history
  plus prior model predictions, preventing test-period contamination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.baselines.random_walk import BaselineModel

logger = logging.getLogger(__name__)


def _is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series)


@dataclass(frozen=True)
class _ChronosConfig:
    model_name: str
    context_length: int
    prediction_length: int
    num_samples: int
    quantiles: tuple[float, ...]
    device: str
    use_covariates: bool
    covariate_columns: tuple[str, ...]
    min_context: int


class Chronos2Runner(BaselineModel):
    """
    Chronos-2 forecaster for return quantiles with strict OOS prediction semantics.

    If ``chronos-forecasting`` is unavailable, a deterministic empirical-quantile
    fallback is used so the pipeline remains testable.
    """
    _PROVENANCE_LOG: list[dict[str, Any]] = []

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        context_length: int = 256,
        prediction_length: int = 1,
        num_samples: int = 64,
        quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
        device: str = "auto",
        use_covariates: bool = False,
        covariate_columns: list[str] | None = None,
        min_context: int = 32,
    ):
        if context_length < 8:
            raise ValueError("context_length must be >= 8")
        if prediction_length != 1:
            raise ValueError("Only prediction_length=1 is supported in strict rolling mode")
        if min_context < 8:
            raise ValueError("min_context must be >= 8")

        covs = tuple(covariate_columns or ())
        self._cfg = _ChronosConfig(
            model_name=model_name,
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=num_samples,
            quantiles=quantiles,
            device=device,
            use_covariates=use_covariates,
            covariate_columns=covs,
            min_context=min_context,
        )

        self._fitted = False
        self._resolved_device = "cpu"
        self._backend_name = "empirical_fallback"
        self._pipeline: Any = None
        self._torch_version: str | None = None
        self._chronos_version: str | None = None
        self._fallback_reason: str | None = "pipeline_not_initialized"

        self._y_history: pd.Series | None = None
        self._feature_columns: list[str] = []
        self._covariate_means: dict[str, float] = {}
        self._covariate_stds: dict[str, float] = {}
        self._covariate_beta: np.ndarray | None = None
        self._covariate_intercept: float = 0.0

    @property
    def name(self) -> str:
        return "Chronos2"

    @classmethod
    def reset_provenance_log(cls) -> None:
        """Clear class-level provenance log for the next run."""
        cls._PROVENANCE_LOG = []

    @classmethod
    def get_provenance_log(cls) -> list[dict[str, Any]]:
        """Return class-level provenance log copy."""
        return [dict(item) for item in cls._PROVENANCE_LOG]

    def provenance_payload(self) -> dict[str, Any]:
        """Current backend provenance metadata."""
        return {
            "model_id": self._cfg.model_name,
            "backend": self._backend_name,
            "fallback_active": bool(self._backend_name == "empirical_fallback"),
            "fallback_reason": self._fallback_reason,
            "requested_device": self._cfg.device,
            "resolved_device": self._resolved_device,
            "chronos_version": self._chronos_version,
            "torch_version": self._torch_version,
            "context_length": int(self._cfg.context_length),
            "prediction_length": int(self._cfg.prediction_length),
            "use_covariates": bool(self._cfg.use_covariates),
        }

    def _record_provenance(self, *, event: str, extra: dict[str, Any] | None = None) -> None:
        payload = self.provenance_payload()
        payload["event"] = event
        if extra:
            payload.update(extra)
        type(self)._PROVENANCE_LOG.append(payload)

    def clone_unfitted(self) -> Chronos2Runner:
        """Create an unfitted copy with identical configuration."""
        return self.__class__(
            model_name=self._cfg.model_name,
            context_length=self._cfg.context_length,
            prediction_length=self._cfg.prediction_length,
            num_samples=self._cfg.num_samples,
            quantiles=self._cfg.quantiles,
            device=self._cfg.device,
            use_covariates=self._cfg.use_covariates,
            covariate_columns=list(self._cfg.covariate_columns),
            min_context=self._cfg.min_context,
        )

    def _resolve_device(self) -> str:
        requested = self._cfg.device.lower()
        try:
            import torch

            self._torch_version = str(getattr(torch, "__version__", None))
        except Exception:
            self._torch_version = None
            if requested == "cuda":
                logger.warning("CUDA requested but torch unavailable; falling back to CPU")
            return "cpu"

        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            return "cpu"
        return "cpu"

    def _maybe_init_pipeline(self) -> None:
        """Try initializing real Chronos pipeline; otherwise use fallback backend."""
        if self._pipeline is not None:
            return

        try:
            import torch
            import chronos as chronos_mod  # type: ignore
            from chronos import ChronosPipeline  # type: ignore

            dtype = torch.bfloat16 if self._resolved_device == "cuda" else torch.float32
            self._pipeline = ChronosPipeline.from_pretrained(
                self._cfg.model_name,
                device_map=self._resolved_device,
                torch_dtype=dtype,
            )
            self._backend_name = "chronos_pipeline"
            self._chronos_version = str(getattr(chronos_mod, "__version__", None))
            self._fallback_reason = None
            logger.info(
                "Initialized Chronos backend model=%s device=%s",
                self._cfg.model_name,
                self._resolved_device,
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._pipeline = None
            self._backend_name = "empirical_fallback"
            self._chronos_version = None
            self._fallback_reason = str(exc)
            logger.warning(
                "Chronos backend unavailable (%s). Using deterministic empirical fallback.",
                exc,
            )

    def _fit_covariate_adjuster(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Fit a lightweight linear shift model for optional covariate adjustment."""
        if not self._cfg.use_covariates:
            self._covariate_beta = None
            self._covariate_intercept = 0.0
            return

        if self._cfg.covariate_columns:
            candidates = [col for col in self._cfg.covariate_columns if col in x.columns]
        else:
            excluded = {"open", "high", "low", "close", "volume"}
            candidates = [
                col
                for col in x.columns
                if col not in excluded and _is_numeric_dtype(x[col]) and not col.startswith("forward_")
            ][:16]

        if not candidates:
            self._covariate_beta = None
            self._covariate_intercept = 0.0
            return

        frame = x[candidates].copy()
        valid = ~(frame.isna().any(axis=1) | y.isna())
        frame = frame.loc[valid]
        y_fit = y.loc[valid]
        if len(frame) < 64:
            self._covariate_beta = None
            self._covariate_intercept = 0.0
            return

        means = frame.mean()
        stds = frame.std().replace(0.0, 1.0)
        norm = (frame - means) / stds

        design = np.column_stack([np.ones(len(norm)), norm.to_numpy(dtype=float)])
        target = y_fit.to_numpy(dtype=float)
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)

        self._feature_columns = list(candidates)
        self._covariate_means = means.to_dict()
        self._covariate_stds = stds.to_dict()
        self._covariate_intercept = float(coeffs[0])
        self._covariate_beta = coeffs[1:].astype(float, copy=True)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> Chronos2Runner:
        if y.empty:
            raise ValueError("Cannot fit Chronos2Runner on empty target series")

        y_clean = y.dropna().astype(float)
        if len(y_clean) < self._cfg.min_context:
            raise ValueError(
                f"Need at least {self._cfg.min_context} non-null target samples; got {len(y_clean)}"
            )

        self._resolved_device = self._resolve_device()
        self._maybe_init_pipeline()

        self._y_history = y_clean.copy()
        self._fit_covariate_adjuster(x=x.reindex(y_clean.index), y=y_clean)
        self._fitted = True
        self._record_provenance(event="fit", extra={"n_samples": int(len(y_clean))})

        logger.info(
            "Fitted %s backend=%s samples=%s context=%s",
            self.name,
            self._backend_name,
            len(y_clean),
            self._cfg.context_length,
        )
        return self

    def _predict_quantiles_from_context(self, context: np.ndarray) -> dict[str, float]:
        """Predict one-step quantiles from context using available backend."""
        if self._pipeline is None:
            return {
                f"q{int(q * 100)}": float(np.quantile(context, q))
                for q in self._cfg.quantiles
            }

        try:  # pragma: no cover - optional dependency path
            import torch

            context_tensor = torch.tensor(context, dtype=torch.float32).reshape(1, -1)
            quantiles_out, _ = self._pipeline.predict_quantiles(
                context=context_tensor,
                prediction_length=1,
                quantile_levels=list(self._cfg.quantiles),
                num_samples=self._cfg.num_samples,
            )
            arr = np.asarray(quantiles_out)
            # Expected shape: [batch=1, horizon=1, n_quantiles]
            if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] != 1:
                raise ValueError(f"Unexpected Chronos quantile output shape: {arr.shape}")
            values = arr[0, 0]
            return {
                f"q{int(q * 100)}": float(values[i])
                for i, q in enumerate(self._cfg.quantiles)
            }
        except Exception as exc:
            logger.warning(
                "Chronos backend inference failed (%s). Falling back to empirical quantiles.",
                exc,
            )
            self._pipeline = None
            self._backend_name = "empirical_fallback"
            self._fallback_reason = str(exc)
            self._record_provenance(event="predict_fallback")
            return {
                f"q{int(q * 100)}": float(np.quantile(context, q))
                for q in self._cfg.quantiles
            }

    def _covariate_shift(self, row: pd.Series) -> float:
        if self._covariate_beta is None or not self._feature_columns:
            return 0.0
        vals: list[float] = []
        for col in self._feature_columns:
            mean = self._covariate_means.get(col, 0.0)
            std = self._covariate_stds.get(col, 1.0)
            raw = float(row.get(col, mean))
            if np.isnan(raw):
                raw = mean
            vals.append((raw - mean) / (std if std != 0 else 1.0))
        x = np.asarray(vals, dtype=float)
        return float(self._covariate_intercept + np.dot(x, self._covariate_beta))

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._y_history is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if x.empty:
            return pd.DataFrame(index=x.index, columns=[f"q{int(q * 100)}" for q in self._cfg.quantiles])

        if not isinstance(x.index, pd.DatetimeIndex):
            raise ValueError("Chronos2Runner expects DatetimeIndex for strict rolling inference")

        idx = x.index.sort_values()
        if not idx.equals(x.index):
            x = x.loc[idx]

        history = self._y_history.to_numpy(dtype=float)
        if len(history) < self._cfg.min_context:
            raise ValueError("Insufficient fitted history for rolling prediction")

        rows: list[dict[str, float]] = []
        rolling_context = history.copy()

        for ts in x.index:
            context = rolling_context[-self._cfg.context_length :]
            q_row = self._predict_quantiles_from_context(context=context)
            shift = self._covariate_shift(x.loc[ts]) if self._cfg.use_covariates else 0.0
            if shift != 0.0:
                for col in list(q_row):
                    q_row[col] = float(q_row[col] + shift)
            rows.append(q_row)

            # Strict OOS update: append model-implied median, never realized future target.
            q50_col = "q50"
            next_point = float(q_row[q50_col]) if q50_col in q_row else float(np.median(context))
            rolling_context = np.append(rolling_context, next_point)

        predictions = pd.DataFrame(rows, index=x.index)
        for q in self._cfg.quantiles:
            col = f"q{int(q * 100)}"
            if col not in predictions.columns:
                predictions[col] = 0.0
        return predictions[[f"q{int(q * 100)}" for q in self._cfg.quantiles]]


class Chronos2ForReturns(Chronos2Runner):
    """Alias for semantic clarity in return forecasting pipelines."""
