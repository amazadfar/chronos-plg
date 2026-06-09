"""Execution intent abstraction between position targets and cost/execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class ExecutionPolicy(str, Enum):
    """Execution style policy used to map transitions to order types."""

    TAKER_ONLY = "taker_only"
    MAKER_PREFERRED = "maker_preferred"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ExecutionIntent:
    """Execution intent for one timestamp."""

    timestamp: pd.Timestamp
    prev_position: float
    target_position: float
    action: str
    side: str
    order_type: str
    policy: str


def classify_transition(prev_position: float, new_position: float) -> str:
    """Classify one position transition."""
    eps = 1e-12
    prev_abs = abs(prev_position)
    new_abs = abs(new_position)
    delta = new_position - prev_position

    if abs(delta) <= eps:
        return "hold"
    if prev_abs <= eps and new_abs > eps:
        return "open"
    if new_abs <= eps and prev_abs > eps:
        return "close"

    same_direction = prev_position * new_position > 0
    if same_direction:
        if new_abs > prev_abs:
            return "increase"
        return "reduce"
    return "reverse"


class ExecutionIntentBuilder:
    """Build execution intent records from a position series."""

    def __init__(self, policy: ExecutionPolicy | str = ExecutionPolicy.TAKER_ONLY):
        if isinstance(policy, ExecutionPolicy):
            self.policy = policy
        else:
            self.policy = ExecutionPolicy(policy)

    def _order_type_for_action(self, action: str) -> str:
        if action == "hold":
            return "none"
        if self.policy == ExecutionPolicy.TAKER_ONLY:
            return "taker"
        if self.policy == ExecutionPolicy.MAKER_PREFERRED:
            return "taker" if action == "reverse" else "maker"

        # Hybrid policy: aggressive when adding risk, passive when reducing.
        if action in {"reduce", "close"}:
            return "maker"
        return "taker"

    @staticmethod
    def _side(prev_position: float, new_position: float) -> str:
        delta = new_position - prev_position
        if abs(delta) <= 1e-12:
            return "none"
        return "buy" if delta > 0 else "sell"

    def build_for_positions(self, positions: pd.Series) -> pd.DataFrame:
        """
        Build an execution-intent table from position targets.

        Output columns:
        - execution_action
        - execution_side
        - execution_order_type
        - execution_policy
        - requires_execution (0/1)
        """
        if positions.empty:
            return pd.DataFrame(
                columns=[
                    "execution_action",
                    "execution_side",
                    "execution_order_type",
                    "execution_policy",
                    "requires_execution",
                ]
            )

        prev_positions = positions.shift(1).fillna(0.0)
        rows: list[dict[str, str | int]] = []
        for ts in positions.index:
            prev = float(prev_positions.loc[ts])
            target = float(positions.loc[ts])
            action = classify_transition(prev, target)
            rows.append(
                {
                    "execution_action": action,
                    "execution_side": self._side(prev, target),
                    "execution_order_type": self._order_type_for_action(action),
                    "execution_policy": self.policy.value,
                    "requires_execution": int(action != "hold"),
                }
            )

        return pd.DataFrame(rows, index=positions.index)
