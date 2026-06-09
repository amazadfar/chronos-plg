"""Dataset quality gate for degraded-run prevention in benchmark pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DegradedRunGateResult:
    """Result payload for degraded-run gate checks."""

    passed: bool
    market_type: str
    availability: dict[str, float]
    thresholds: dict[str, float]
    reasons: list[str]
    warnings: list[str]
    quality_report_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "market_type": self.market_type,
            "availability": self.availability,
            "thresholds": self.thresholds,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "quality_report_path": self.quality_report_path,
        }


MIN_KEY_FAMILY_AVAILABILITY: dict[str, dict[str, float]] = {
    "futures": {
        "funding": 0.70,
        "open_interest": 0.70,
        "liquidations": 0.70,
        "macro": 0.70,
    },
    "margin": {
        "funding": 0.0,
        "open_interest": 0.0,
        "liquidations": 0.0,
        "macro": 0.70,
    },
    "spot": {
        "funding": 0.0,
        "open_interest": 0.0,
        "liquidations": 0.0,
        "macro": 0.70,
    },
}


def _quality_report_path_for_dataset(dataset_path: Path) -> Path:
    """
    Resolve quality-report path for a processed dataset file.

    Priority:
    1) `<stem>_metadata.json` -> `quality_report_file`
    2) `<stem>_quality.json`
    """
    dataset_path = Path(dataset_path)
    metadata_path = dataset_path.with_name(f"{dataset_path.stem}_metadata.json")
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            rel = payload.get("quality_report_file")
            if isinstance(rel, str) and rel.strip():
                candidate = metadata_path.parent / rel
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return dataset_path.with_name(f"{dataset_path.stem}_quality.json")


def _extract_family_ratio(report: dict[str, Any], family: str) -> float:
    payload = report.get("key_family_availability", {}).get(family, {})
    try:
        return float(payload.get("availability_ratio", 0.0))
    except Exception:
        return 0.0


def evaluate_degraded_run_gate(
    quality_report: dict[str, Any],
    *,
    market_type: str,
    quality_report_path: str = "",
) -> DegradedRunGateResult:
    """Evaluate degraded-run gate from a quality report and target market type."""
    market = str(market_type).lower()
    thresholds = MIN_KEY_FAMILY_AVAILABILITY.get(
        market,
        MIN_KEY_FAMILY_AVAILABILITY["futures"],
    )
    reasons: list[str] = []
    warnings: list[str] = []

    if "key_family_availability" not in quality_report:
        reasons.append("quality_report_missing_key_family_availability(rebuild_dataset_required)")
        return DegradedRunGateResult(
            passed=False,
            market_type=market,
            availability={family: 0.0 for family in thresholds},
            thresholds=thresholds,
            reasons=reasons,
            warnings=warnings,
            quality_report_path=quality_report_path,
        )

    availability = {
        "funding": _extract_family_ratio(quality_report, "funding"),
        "open_interest": _extract_family_ratio(quality_report, "open_interest"),
        "liquidations": _extract_family_ratio(quality_report, "liquidations"),
        "macro": _extract_family_ratio(quality_report, "macro"),
    }

    for family, threshold in thresholds.items():
        if threshold <= 0:
            continue
        ratio = availability.get(family, 0.0)
        if ratio < threshold:
            reasons.append(f"{family}_availability_below_threshold({ratio:.3f}<{threshold:.3f})")

    try:
        real_liq_ratio = float(
            quality_report.get("key_family_availability", {})
            .get("liquidations", {})
            .get("real_ratio", 0.0)
        )
    except Exception:
        real_liq_ratio = 0.0
    if availability["liquidations"] >= thresholds.get("liquidations", 0.0) and real_liq_ratio <= 0.01:
        warnings.append("liquidation_data_is_proxy_only(no_real_liquidation_coverage)")

    return DegradedRunGateResult(
        passed=len(reasons) == 0,
        market_type=market,
        availability=availability,
        thresholds=thresholds,
        reasons=reasons,
        warnings=warnings,
        quality_report_path=quality_report_path,
    )


def enforce_degraded_run_gate(
    dataset_path: Path,
    *,
    market_type: str,
    artifact_path: Path | None = None,
) -> DegradedRunGateResult:
    """
    Enforce degraded-run gate for a processed dataset.

    Raises:
        RuntimeError: when dataset quality fails gate checks.
    """
    quality_path = _quality_report_path_for_dataset(dataset_path)
    if not quality_path.exists():
        raise RuntimeError(
            "Missing dataset quality report required for degraded-run gate: "
            f"{quality_path}"
        )

    report = json.loads(quality_path.read_text(encoding="utf-8"))
    result = evaluate_degraded_run_gate(
        report,
        market_type=market_type,
        quality_report_path=str(quality_path),
    )

    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )

    if not result.passed:
        reasons = ";".join(result.reasons) if result.reasons else "unknown_quality_failure"
        raise RuntimeError(
            "Dataset degraded-run gate failed: "
            f"{reasons}. "
            f"Quality report={quality_path}"
        )
    return result

