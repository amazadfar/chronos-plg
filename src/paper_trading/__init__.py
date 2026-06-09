"""Paper-trading replay, monitoring, and deployment policy helpers."""

from src.paper_trading.engine import PaperTradingConfig, PaperTradingEngine, PaperTradingReplay
from src.paper_trading.monitoring import build_daily_weekly_dashboards, build_monitoring_dashboard
from src.paper_trading.policy import (
    CapitalRampDecision,
    CapitalRampPolicy,
    DeploymentReadiness,
    DeploymentReadinessPolicy,
    KillSwitchEvent,
    KillSwitchThresholds,
    build_kill_event_taxonomy,
    build_low_activity_diagnostics,
    default_capital_ramp_policy,
    evaluate_deployment_readiness,
    evaluate_kill_switch,
    recommend_capital_action,
    render_capital_ramp_policy,
    serialize_kill_events,
)

__all__ = [
    "PaperTradingConfig",
    "PaperTradingEngine",
    "PaperTradingReplay",
    "build_monitoring_dashboard",
    "build_daily_weekly_dashboards",
    "KillSwitchThresholds",
    "KillSwitchEvent",
    "evaluate_kill_switch",
    "build_kill_event_taxonomy",
    "build_low_activity_diagnostics",
    "DeploymentReadinessPolicy",
    "DeploymentReadiness",
    "evaluate_deployment_readiness",
    "CapitalRampPolicy",
    "CapitalRampDecision",
    "default_capital_ramp_policy",
    "recommend_capital_action",
    "render_capital_ramp_policy",
    "serialize_kill_events",
]
