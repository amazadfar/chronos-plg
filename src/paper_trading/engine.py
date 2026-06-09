"""Paper-trading replay engine using the same execution-cost assumptions as backtests."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import get_settings
from src.backtest.costs import CostModel
from src.backtest.engine import BacktestResult
from src.common.metrics import profit_factor_from_returns
from src.models.baselines.random_walk import BaselineModel
from src.strategy.position_sizing import PositionSizer
from src.strategy.regime_detector import RegimeDetector
from src.strategy.signals import QuantileSignalGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperTradingConfig:
    """Configuration for sequential paper-trading replay."""

    retrain_interval_bars: int
    training_window_bars: int
    min_train_samples: int


@dataclass
class PaperTradingReplay:
    """Output of paper-trading replay."""

    model_name: str
    scenario_name: str
    feature_columns: list[str]
    paper_log: pd.DataFrame
    backtest_result: BacktestResult
    retrain_timestamps: list[str]
    skipped_bars: int

    def to_dict(self) -> dict[str, object]:
        """Serializable replay summary."""
        backtest = self.backtest_result
        pf_net = backtest.profit_factor_net if backtest.profit_factor_net > 0 else backtest.profit_factor
        return {
            "model_name": self.model_name,
            "scenario_name": self.scenario_name,
            "bars_replayed": int(len(self.paper_log)),
            "retrain_count": int(len(self.retrain_timestamps)),
            "skipped_bars": int(self.skipped_bars),
            "feature_columns": self.feature_columns,
            "metrics": {
                "total_return": float(backtest.total_return),
                "annualized_return": float(backtest.annualized_return),
                "sharpe_ratio": float(backtest.sharpe_ratio),
                "max_drawdown": float(backtest.max_drawdown),
                "profit_factor_net": float(pf_net),
                "win_rate": float(backtest.win_rate),
                "num_trades": int(backtest.num_trades),
                "total_costs": float(backtest.total_costs),
                "total_fees": float(backtest.total_fees),
                "total_slippage": float(backtest.total_slippage),
                "total_funding": float(backtest.total_funding),
                "total_interest": float(backtest.total_interest),
                "total_other_costs": float(backtest.total_other_costs),
            },
        }


class PaperTradingEngine:
    """Sequential paper-trading engine with strict past-only training."""

    def __init__(
        self,
        model_class: type[BaselineModel],
        model_kwargs: dict | None = None,
        config: PaperTradingConfig | None = None,
        cost_model: CostModel | None = None,
        signal_generator: QuantileSignalGenerator | None = None,
        position_sizer: PositionSizer | None = None,
        regime_detector: RegimeDetector | None = None,
        target_column: str = "forward_return",
    ):
        settings = get_settings()
        default_cfg = PaperTradingConfig(
            retrain_interval_bars=max(1, settings.walk_forward.effective_step_days * 6),
            training_window_bars=max(120, settings.walk_forward.effective_train_days * 6),
            min_train_samples=settings.walk_forward.min_train_samples,
        )

        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.config = config or default_cfg
        self.cost_model = cost_model or CostModel()
        self.signal_generator = signal_generator or QuantileSignalGenerator()
        self.position_sizer = position_sizer or PositionSizer()
        self.regime_detector = regime_detector or RegimeDetector()
        self.target_column = target_column

    @staticmethod
    def _resolve_start_index(index: pd.DatetimeIndex, start_date: str | None) -> int:
        if start_date is None:
            return 1

        start_ts = pd.Timestamp(start_date)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(index.tz)
        else:
            start_ts = start_ts.tz_convert(index.tz)
        return int(max(1, index.searchsorted(start_ts, side="left")))

    @staticmethod
    def _select_feature_columns(
        data: pd.DataFrame,
        feature_columns: list[str] | None,
    ) -> list[str]:
        if feature_columns:
            columns = [
                col for col in feature_columns
                if col in data.columns and data[col].notna().sum() > 0
            ]
            if not columns:
                raise ValueError("No requested feature columns found in dataset")
            return columns

        exclude_patterns = (
            "forward_",
            "regime",
            "hist_q",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        )

        columns: list[str] = []
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
            if any(pattern in col for pattern in exclude_patterns):
                continue
            if data[col].notna().sum() == 0:
                continue
            columns.append(col)

        if not columns:
            raise ValueError("No numeric feature columns available for paper-trading replay")
        return columns

    @staticmethod
    def _build_regime_series(data: pd.DataFrame, detector: RegimeDetector) -> pd.Series:
        if "close" not in data.columns:
            return pd.Series("normal", index=data.index, dtype="object")
        return detector.detect_regimes(data)["regime"]

    def run(
        self,
        data: pd.DataFrame,
        feature_columns: list[str] | None = None,
        start_date: str | None = None,
        model_name: str = "model",
        scenario_name: str = "scenario",
    ) -> PaperTradingReplay:
        """Run paper-trading replay on a historical dataset."""
        if self.target_column not in data.columns:
            raise ValueError(f"Missing target column '{self.target_column}'")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Paper-trading data index must be a DatetimeIndex")
        if data.index.tz is None:
            raise ValueError("Paper-trading data index must be timezone-aware")
        if not data.index.is_monotonic_increasing:
            raise ValueError("Paper-trading data index must be monotonic increasing")

        selected_features = self._select_feature_columns(data, feature_columns)
        start_idx = self._resolve_start_index(data.index, start_date)
        regimes = self._build_regime_series(data, self.regime_detector)

        model: BaselineModel | None = None
        steps_since_retrain = self.config.retrain_interval_bars

        rows: list[dict[str, object]] = []
        retrain_timestamps: list[str] = []
        skipped_bars = 0
        bar_seconds = self.cost_model._infer_bar_seconds(data.index)

        for i in range(start_idx, len(data)):
            ts = data.index[i]
            realized = data.at[ts, self.target_column]

            if pd.isna(realized):
                skipped_bars += 1
                continue

            train_end_idx = i - 1
            if train_end_idx < 0:
                skipped_bars += 1
                continue

            train_start_idx = max(0, train_end_idx - self.config.training_window_bars + 1)
            train_slice = data.iloc[train_start_idx : train_end_idx + 1]

            y_train = train_slice[self.target_column].dropna()
            if len(y_train) < self.config.min_train_samples:
                skipped_bars += 1
                continue
            x_train = train_slice.loc[y_train.index, selected_features]

            needs_retrain = model is None or steps_since_retrain >= self.config.retrain_interval_bars
            if needs_retrain:
                model = self.model_class(**self.model_kwargs)
                model.fit(x_train, y_train)
                retrain_timestamps.append(ts.isoformat())
                steps_since_retrain = 0

            x_live = data.loc[[ts], selected_features]
            predictions = model.predict(x_live).copy()
            required_columns = {"q10", "q50", "q90"}
            missing = required_columns - set(predictions.columns)
            if missing:
                raise ValueError(f"Model predictions missing columns: {sorted(missing)}")
            predictions["regime"] = str(regimes.loc[ts])

            if "realized_vol_6" in data.columns and pd.notna(data.at[ts, "realized_vol_6"]):
                volatility = float(data.at[ts, "realized_vol_6"])
            else:
                volatility = 0.02
            if not np.isfinite(volatility) or volatility <= 0:
                volatility = 0.02

            expected_funding = (
                float(data.at[ts, "funding_rate"])
                if "funding_rate" in data.columns and pd.notna(data.at[ts, "funding_rate"])
                else 0.0
            )
            expected_borrow = (
                float(data.at[ts, "borrow_rate_per_day"])
                if "borrow_rate_per_day" in data.columns and pd.notna(data.at[ts, "borrow_rate_per_day"])
                else np.nan
            )
            expected_other = (
                float(data.at[ts, "other_costs"])
                if "other_costs" in data.columns and pd.notna(data.at[ts, "other_costs"])
                else 0.0
            )
            expected_cost = self.cost_model.estimate_entry_cost_rate(
                volatility=volatility,
                expected_funding_rate=expected_funding,
                expected_borrow_rate_per_day=expected_borrow,
                bar_seconds=bar_seconds,
                expected_holding_bars=max(
                    1,
                    int(getattr(self.signal_generator, "expected_cost_holding_bars", 1)),
                ),
                include_exit=bool(getattr(self.signal_generator, "expected_cost_round_trip", True)),
                target_notional=1.0,
                other_cost=expected_other,
            )
            predictions[getattr(self.signal_generator, "expected_cost_column", "expected_cost")] = expected_cost
            predictions[getattr(self.signal_generator, "predicted_risk_column", "predicted_risk")] = volatility

            signals = self.signal_generator.generate_signals(predictions)

            vol_series = pd.Series([volatility], index=[ts], dtype=float)
            regime_series = pd.Series([regimes.loc[ts]], index=[ts])
            regime_mult = self.regime_detector.get_regime_multipliers(regime_series)

            prices = None
            if "close" in data.columns:
                prices = pd.Series([float(data.at[ts, "close"])], index=[ts], dtype=float)

            short_allowed: pd.Series | bool | None = None
            if "short_allowed" in data.columns:
                short_allowed = pd.Series([bool(data.at[ts, "short_allowed"])], index=[ts])

            position = self.position_sizer.calculate_sizes(
                signals=signals,
                predicted_vol=vol_series,
                regime_multipliers=regime_mult,
                prices=prices,
                short_allowed=short_allowed,
            )

            rows.append(
                {
                    "timestamp": ts,
                    "q10": float(predictions.at[ts, "q10"]),
                    "q50": float(predictions.at[ts, "q50"]),
                    "q90": float(predictions.at[ts, "q90"]),
                    "uncertainty": float(signals.at[ts, "uncertainty"]),
                    "signal": int(signals.at[ts, "signal"]),
                    "signal_strength": float(signals.at[ts, "signal_strength"]),
                    "signal_confidence": float(signals.at[ts, "signal_confidence"]),
                    "decision_reason": str(signals.at[ts, "decision_reason"]),
                    "expected_cost": float(signals.at[ts, "expected_cost"]),
                    "predicted_risk": float(signals.at[ts, "predicted_risk"]),
                    "required_edge": float(signals.at[ts, "required_edge"]),
                    "edge_margin": float(signals.at[ts, "edge_margin"]),
                    "entry_policy": str(signals.at[ts, "entry_policy"]),
                    "position": float(position.at[ts]),
                    "regime": str(regimes.loc[ts]),
                    "retrained": int(needs_retrain),
                    "train_samples": int(len(y_train)),
                    "target_return": float(realized),
                }
            )
            steps_since_retrain += 1

        if not rows:
            raise ValueError("No paper-trading bars could be replayed under current config")

        paper_log = pd.DataFrame(rows).set_index("timestamp")
        paper_log.index = pd.to_datetime(paper_log.index, utc=True)

        index = paper_log.index
        positions = paper_log["position"].astype(float)
        actual_returns = paper_log["target_return"].astype(float)

        volatilities = (
            data.loc[index, "realized_vol_6"].astype(float)
            if "realized_vol_6" in data.columns
            else pd.Series(0.02, index=index, dtype=float)
        )
        funding = (
            data.loc[index, "funding_rate"].astype(float)
            if "funding_rate" in data.columns
            else pd.Series(0.0, index=index, dtype=float)
        )
        borrow = (
            data.loc[index, "borrow_rate_per_day"].astype(float)
            if "borrow_rate_per_day" in data.columns
            else pd.Series(np.nan, index=index, dtype=float)
        )
        other = (
            data.loc[index, "other_costs"].astype(float)
            if "other_costs" in data.columns
            else pd.Series(0.0, index=index, dtype=float)
        )

        costs = self.cost_model.calculate_execution_costs(
            positions=positions,
            volatilities=volatilities,
            funding_rates=funding,
            borrow_rates_per_day=borrow,
            other_costs=other,
        )

        gross_returns = positions.shift(1).fillna(0.0) * actual_returns
        net_returns = gross_returns - costs["total_costs"]

        returns = pd.DataFrame(
            {
                "gross_return": gross_returns,
                "event_type": costs["event_type"],
                "prev_position": costs["prev_position"],
                "position": costs["position"],
                "open_notional": costs["open_notional"],
                "close_notional": costs["close_notional"],
                "traded_notional": costs["traded_notional"],
                "fees": costs["fees"],
                "funding": costs["funding"],
                "interest": costs["interest"],
                "slippage": costs["slippage"],
                "other_costs": costs["other_costs"],
                "total_costs": costs["total_costs"],
                "net_return": net_returns,
                "regime": paper_log["regime"],
                "signal": paper_log["signal"],
                "turnover": (positions - positions.shift(1).fillna(0.0)).abs(),
            },
            index=index,
        )

        equity = (1 + returns["net_return"].fillna(0.0)).cumprod()
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak

        valid_returns = returns["net_return"].dropna()
        periods_per_year = 6 * 365
        years = len(valid_returns) / periods_per_year

        total_return = float(equity.iloc[-1] - 1)
        annualized_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

        mean_ret = valid_returns.mean()
        std_ret = valid_returns.std()
        sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

        downside = valid_returns[valid_returns < 0]
        downside_std = downside.std()
        sortino = (
            float(mean_ret / downside_std * np.sqrt(periods_per_year))
            if len(downside) > 0 and downside_std > 0
            else 0.0
        )

        pf_net = float(profit_factor_from_returns(valid_returns))
        wins = valid_returns[valid_returns > 0]

        backtest = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=float(drawdown.min()),
            total_fees=float(returns["fees"].sum()),
            total_slippage=float(returns["slippage"].sum()),
            total_funding=float(returns["funding"].sum()),
            total_interest=float(returns["interest"].sum()),
            total_other_costs=float(returns["other_costs"].sum()),
            total_costs=float(returns["total_costs"].sum()),
            num_trades=int((returns["traded_notional"] > 0.01).sum()),
            win_rate=float(len(wins) / len(valid_returns)) if len(valid_returns) > 0 else 0.0,
            profit_factor_net=pf_net,
            profit_factor=pf_net,
            positions=paper_log,
            returns=returns,
            equity_curve=equity,
            trades=returns.loc[returns["traded_notional"] > 0].copy(),
        )

        for regime in returns["regime"].dropna().unique():
            mask = returns["regime"] == regime
            regime_returns = returns.loc[mask, "net_return"].dropna()
            if len(regime_returns) <= 10:
                continue
            backtest.regime_returns[str(regime)] = float(regime_returns.sum())
            regime_std = regime_returns.std()
            if regime_std > 0:
                backtest.regime_sharpes[str(regime)] = float(
                    regime_returns.mean() / regime_std * np.sqrt(periods_per_year)
                )

        logger.info(
            "Paper replay complete: bars=%s trades=%s PF=%.3f Sharpe=%.3f",
            len(paper_log),
            backtest.num_trades,
            pf_net,
            sharpe,
        )

        return PaperTradingReplay(
            model_name=model_name,
            scenario_name=scenario_name,
            feature_columns=selected_features,
            paper_log=paper_log,
            backtest_result=backtest,
            retrain_timestamps=retrain_timestamps,
            skipped_bars=skipped_bars,
        )
