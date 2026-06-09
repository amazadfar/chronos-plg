"""Experiment/run metadata helpers for CLI scripts."""

from __future__ import annotations

import json
import random
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_FILENAME = "run_manifest.json"


def now_utc_iso() -> str:
    """UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def set_global_seed(seed: int) -> None:
    """Set deterministic seed across common libraries."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional for many script paths.
        pass


def _safe_git_commit(cwd: Path) -> str | None:
    """Best-effort Git commit lookup."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": "1.0", "runs": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def start_experiment_run(
    *,
    script_name: str,
    args: dict[str, Any],
    seed: int,
    output_dir: Path,
    project_root: Path | None = None,
) -> tuple[str, Path]:
    """
    Start a tracked experiment run.

    Returns:
        tuple of (run_id, manifest_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    run_id = (
        f"{script_name.replace('.py', '')}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    root = project_root or Path.cwd()
    run_record = {
        "run_id": run_id,
        "script": script_name,
        "status": "running",
        "started_at": now_utc_iso(),
        "ended_at": None,
        "seed": seed,
        "args": args,
        "git_commit": _safe_git_commit(root),
        "artifacts": [],
        "notes": {},
    }

    manifest = _load_manifest(manifest_path)
    manifest.setdefault("runs", []).append(run_record)
    _save_manifest(manifest_path, manifest)
    return run_id, manifest_path


def finalize_experiment_run(
    *,
    manifest_path: Path,
    run_id: str,
    status: str,
    artifacts: list[str] | None = None,
    notes: dict[str, Any] | None = None,
) -> None:
    """Finalize run status in manifest."""
    manifest = _load_manifest(manifest_path)
    runs = manifest.get("runs", [])
    target = next((r for r in runs if r.get("run_id") == run_id), None)
    if target is None:
        return

    target["status"] = status
    target["ended_at"] = now_utc_iso()
    if artifacts:
        target["artifacts"] = artifacts
    if notes:
        merged_notes = dict(target.get("notes", {}))
        merged_notes.update(notes)
        target["notes"] = merged_notes

    _save_manifest(manifest_path, manifest)
