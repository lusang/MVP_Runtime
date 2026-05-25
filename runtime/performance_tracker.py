"""
PerformanceTracker — SQLite-backed model performance history for Planner decisions.

Records per-model, per-step metrics and provides aggregated stats.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

DB_DEFAULT = Path(__file__).resolve().parent.parent / "storage" / "performance.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS performance_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    step          TEXT    NOT NULL,
    model_id      TEXT    NOT NULL,
    template_name TEXT    NOT NULL DEFAULT '',
    success       INTEGER NOT NULL DEFAULT 1,
    latency_ms    REAL    NOT NULL DEFAULT 0,
    confidence    REAL    NOT NULL DEFAULT 0,
    object_count  INTEGER NOT NULL DEFAULT 0,
    details       TEXT    NOT NULL DEFAULT '{}',
    timestamp     INTEGER NOT NULL DEFAULT (CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER))
);

CREATE INDEX IF NOT EXISTS idx_perf_step_model ON performance_log(step, model_id);
CREATE INDEX IF NOT EXISTS idx_perf_template ON performance_log(template_name);
CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_log(timestamp);
"""


class PerformanceTracker:
    """Records per-model, per-step metrics and provides aggregated stats for the Planner."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else DB_DEFAULT
        self._conn: sqlite3.Connection | None = None

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
        if self._conn:
            self._conn.close()
            self._conn = None

    def record_step(
        self,
        *,
        run_id: str,
        step: str,
        model_id: str,
        template_name: str = "",
        success: bool = True,
        latency_ms: float = 0.0,
        confidence: float = 0.0,
        object_count: int = 0,
        details: dict[str, Any] | None = None,
    ) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO performance_log
               (run_id, step, model_id, template_name, success, latency_ms, confidence, object_count, details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, step, model_id, template_name, int(success),
                latency_ms, confidence, object_count,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        conn.commit()

    def get_step_stats(
        self,
        step: str,
        model_id: str,
        template_name: str | None = None,
        limit_days: int = 30,
    ) -> dict[str, Any]:
        """Aggregated stats: success_rate, avg_latency, avg_confidence, sample_count."""
        conn = self._get_conn()
        since_ts = (time.time() - limit_days * 86400) * 1000
        where = "WHERE step = ? AND model_id = ? AND timestamp > ?"
        params: list[Any] = [step, model_id, since_ts]
        if template_name:
            where += " AND template_name = ?"
            params.append(template_name)

        row = conn.execute(
            f"""SELECT
                    COUNT(*) AS cnt,
                    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) AS success_rate,
                    AVG(latency_ms) AS avg_latency,
                    AVG(confidence) AS avg_confidence
                FROM performance_log {where}""",
            params,
        ).fetchone()

        return {
            "sample_count": row["cnt"] or 0,
            "success_rate": round(row["success_rate"] or 0.0, 3),
            "avg_latency_ms": round(row["avg_latency"] or 0.0, 1),
            "avg_confidence": round(row["avg_confidence"] or 0.0, 3),
        }

    def get_all_stats_for_planner(self, template_name: str | None = None) -> dict[str, Any]:
        """Return compact stats dict suitable for injection into the Planner prompt."""
        steps = ["detect", "verify", "merge", "attribute", "quality", "negative"]
        result: dict[str, dict[str, Any]] = {}
        for step in steps:
            result[step] = {}
            from runtime.model_registry import get_models_for_step
            for model in get_models_for_step(step):
                stats = self.get_step_stats(step, model.model_id, template_name)
                if stats["sample_count"] > 0:
                    result[step][model.model_id] = stats
        return result

    def performance_summary_text(self, template_name: str | None = None) -> str:
        """Render aggregated stats as text for the Planner prompt."""
        stats = self.get_all_stats_for_planner(template_name)
        if not any(stats.values()):
            return "No historical performance data available yet."

        lines = ["Historical Performance (last 30 days):"]
        for step_name, models in stats.items():
            for model_id, s in models.items():
                lines.append(
                    f"  {step_name}/{model_id}: success={s['success_rate']:.2f}, "
                    f"latency={s['avg_latency_ms']:.0f}ms, "
                    f"confidence={s['avg_confidence']:.2f}, n={s['sample_count']}"
                )
        return "\n".join(lines)
