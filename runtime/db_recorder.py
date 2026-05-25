"""
RuntimeRecorder — SQLite-backed step trace recorder for pipeline observability.

Records pipeline runs, DAG snapshots, per-step outcomes, and step-level events
into SQLite. Schema matches config/mvp_runtime_events_ddl.sql (adapted for SQLite).

Usage:
    recorder = RuntimeRecorder()
    recorder.ensure_schema()

    recorder.create_run(run_id="...", template_id="...", ...)
    recorder.save_plan_snapshot(run_id="...", graph_json=..., ...)

    step_id = recorder.start_step(run_id="...", step_type="verify", ...)
    # ... execute step ...
    recorder.finish_step(step_id=step_id, status="success", ...)

    recorder.record_event(step_id=step_id, event_type="quality_scores", payload={...})

    recorder.finish_run(run_id="...", status="completed", total_latency_ms=1234)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_DEFAULT = Path(__file__).resolve().parent.parent / "storage" / "runtime_events.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runtime_run (
    run_id            TEXT PRIMARY KEY,
    template_id       TEXT,
    template_version  TEXT,
    graph_hash        TEXT,
    plan_hash         TEXT,
    input_asset_id    TEXT,
    status            TEXT,
    started_at        TEXT,
    finished_at       TEXT,
    total_latency_ms  INTEGER,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runtime_plan_snapshot (
    snapshot_id       TEXT PRIMARY KEY,
    run_id            TEXT REFERENCES runtime_run(run_id),
    graph_json        TEXT,
    graph_hash        TEXT,
    created_at        TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runtime_step (
    step_id               TEXT PRIMARY KEY,
    run_id                TEXT REFERENCES runtime_run(run_id),
    step_name             TEXT,
    step_type             TEXT,
    handler               TEXT,
    model_id              TEXT,
    status                TEXT,
    started_at            TEXT,
    finished_at           TEXT,
    latency_ms            INTEGER,
    input_candidate_count INTEGER,
    output_candidate_count INTEGER,
    error_message         TEXT,
    created_at            TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS runtime_step_event (
    event_id      TEXT PRIMARY KEY,
    step_id       TEXT REFERENCES runtime_step(step_id),
    event_type    TEXT,
    payload       TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_step_run_id ON runtime_step(run_id);
CREATE INDEX IF NOT EXISTS idx_event_step_id ON runtime_step_event(step_id);
"""


class RuntimeRecorder:
    """SQLite-backed recorder for pipeline run observability.

    Thread-safe for read operations. Writes are serialized through a
    single connection with ``check_same_thread=False``.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_DEFAULT
        self._conn: sqlite3.Connection | None = None

    # ── connection management ────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── run lifecycle ────────────────────────────────────────────────

    def create_run(
        self,
        *,
        run_id: str,
        template_id: str = "",
        template_version: str = "",
        graph_hash: str = "",
        plan_hash: str = "",
        input_asset_id: str = "",
    ) -> None:
        """Insert a new run record with status='running'."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO runtime_run
               (run_id, template_id, template_version, graph_hash, plan_hash,
                input_asset_id, status, started_at, total_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0)""",
            (run_id, template_id, template_version, graph_hash, plan_hash,
             input_asset_id, now),
        )
        conn.commit()

    def finish_run(
        self,
        *,
        run_id: str,
        status: str = "completed",
        total_latency_ms: float = 0.0,
    ) -> None:
        """Mark a run as completed / failed with total latency."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE runtime_run SET
               status = ?, finished_at = ?, total_latency_ms = ?
               WHERE run_id = ?""",
            (status, now, int(total_latency_ms), run_id),
        )
        conn.commit()

    # ── plan snapshot ────────────────────────────────────────────────

    def save_plan_snapshot(
        self,
        *,
        run_id: str,
        graph_json: dict[str, Any] | None = None,
        graph_hash: str = "",
    ) -> None:
        """Store the serialised DAG snapshot for a run."""
        conn = self._get_conn()
        snapshot_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO runtime_plan_snapshot
               (snapshot_id, run_id, graph_json, graph_hash)
               VALUES (?, ?, ?, ?)""",
            (snapshot_id, run_id,
             json.dumps(graph_json or {}, ensure_ascii=False, default=str),
             graph_hash),
        )
        conn.commit()

    # ── step lifecycle ───────────────────────────────────────────────

    def start_step(
        self,
        *,
        run_id: str,
        step_name: str = "",
        step_type: str = "",
        handler: str = "",
        model_id: str = "",
        input_count: int = 0,
    ) -> str:
        """Record the start of a step. Returns the new step_id."""
        conn = self._get_conn()
        step_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO runtime_step
               (step_id, run_id, step_name, step_type, handler, model_id,
                status, started_at, input_candidate_count)
               VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)""",
            (step_id, run_id, step_name, step_type, handler, model_id,
             now, input_count),
        )
        conn.commit()
        return step_id

    def finish_step(
        self,
        *,
        step_id: str,
        status: str = "success",
        latency_ms: float = 0.0,
        output_count: int = 0,
        error_message: str = "",
    ) -> None:
        """Record the completion / failure of a step."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """UPDATE runtime_step SET
               status = ?, finished_at = ?, latency_ms = ?,
               output_candidate_count = ?, error_message = ?
               WHERE step_id = ?""",
            (status, now, int(latency_ms), output_count, error_message, step_id),
        )
        conn.commit()

    # ── step events ──────────────────────────────────────────────────

    def record_event(
        self,
        *,
        step_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record a free-form event attached to a step."""
        conn = self._get_conn()
        event_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO runtime_step_event
               (event_id, step_id, event_type, payload)
               VALUES (?, ?, ?, ?)""",
            (event_id, step_id, event_type,
             json.dumps(payload or {}, ensure_ascii=False, default=str)),
        )
        conn.commit()

    # ── batch recording from StepResult list ──────────────────────────

    def record_step_results(
        self,
        *,
        run_id: str,
        step_results: list[Any],
    ) -> None:
        """Record step results post-hoc (when real-time recording was not used)."""
        for sr in step_results:
            step_id = self.start_step(
                run_id=run_id,
                step_name=f"{sr.step}:{sr.model_id}" if sr.model_id else sr.step,
                step_type=sr.step,
                model_id=sr.model_id,
                input_count=sr.input_count,
            )
            self.finish_step(
                step_id=step_id,
                status=sr.status,
                latency_ms=sr.latency_ms,
                output_count=sr.output_count,
                error_message=sr.error or "",
            )
            for ev in sr.events:
                if isinstance(ev, dict):
                    self.record_event(
                        step_id=step_id,
                        event_type=ev.get("type", "step_event"),
                        payload=ev.get("payload", ev),
                    )
