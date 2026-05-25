#!/usr/bin/env python3
"""
tools.last_run — 查看最新一次 pipeline 运行的每步详情。

Usage:
    python -m tools.last_run                # default: last 1 run
    python -m tools.last_run --last 3       # last 3 runs
    python -m tools.last_run --steps-only   # only step table, no run header
    python -m tools.last_run --events       # also show step events
    python -m tools.last_run --db PATH      # custom db path

Requires: storage/runtime_events.db (created by RuntimeRecorder).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB_DEFAULT = (
    Path(__file__).resolve().parent.parent / "storage" / "runtime_events.db"
)


def _fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms/1000:.1f}s"


def _fmt_time(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%H:%M:%S.%f")[:12]
    except Exception:
        return iso[:19]


def _load_runs(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM runtime_run
           ORDER BY started_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()


def _load_steps(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM runtime_step
           WHERE run_id = ?
           ORDER BY rowid""",
        (run_id,),
    ).fetchall()


def _load_events(conn: sqlite3.Connection, step_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT * FROM runtime_step_event
           WHERE step_id = ?
           ORDER BY rowid""",
        (step_id,),
    ).fetchall()


def _load_plan_snapshot(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """SELECT * FROM runtime_plan_snapshot
           WHERE run_id = ?
           ORDER BY created_at DESC
           LIMIT 1""",
        (run_id,),
    ).fetchone()


def _print_run(run: sqlite3.Row, conn: sqlite3.Connection, show_events: bool) -> None:
    run_id = run["run_id"]
    sep = "=" * 72

    # ── Run header ──────────────────────────────────────────────────
    print(sep)
    print(f"  Run        {run_id}")
    print(f"  Status     {run['status']}")
    print(f"  Template   {run['template_id']}  v{run['template_version'] or '-'}")
    print(f"  Image      {run['input_asset_id'] or '-'}")
    print(f"  Graph hash {run['graph_hash'] or '-'}")
    print(f"  Total      {_fmt_ms(run['total_latency_ms'] or 0)}")
    print(f"  Started    {_fmt_time(run['started_at'])}")
    print(f"  Finished   {_fmt_time(run['finished_at'])}")
    print(sep)

    # ── Plan snapshot ───────────────────────────────────────────────
    snapshot = _load_plan_snapshot(conn, run_id)
    if snapshot and snapshot["graph_json"]:
        try:
            dag = json.loads(snapshot["graph_json"])
            nodes = dag.get("nodes", [])
            lines = []
            for n in nodes:
                attrs = f" ({', '.join(n.get('attributes', []))})" if n.get("attributes") else ""
                lines.append(f"      [{n['order']}] {n['step']:12s} {n['model_id']:20s} {n['data_flow']}{attrs}")
            if lines:
                print("  DAG:")
                print("\n".join(lines))
                print(sep)
        except (json.JSONDecodeError, KeyError):
            pass

    # ── Steps ───────────────────────────────────────────────────────
    steps = _load_steps(conn, run_id)
    if not steps:
        print("  (no steps recorded)")
        print()
        return

    # Header
    print(f"  Steps ({len(steps)}):")
    hdr = f"  {'#':>3s}  {'STEP':<28s} {'HANDLER':<22s} {'STATUS':<10s} {'LATENCY':<8s} {'IN/OUT':<8s}"
    print(hdr)
    print(f"  {'-'*len(hdr)}")

    for i, s in enumerate(steps):
        name = f"{s['step_name'] or s['step_type']}"
        handler = s["handler"] or "-"
        status = s["status"]
        latency = _fmt_ms(s["latency_ms"] or 0)
        in_out = f"{s['input_candidate_count'] or 0}/{s['output_candidate_count'] or 0}"

        print(f"  {i:>3d}  {name:<28s} {handler:<22s} {status:<10s} {latency:<8s} {in_out:<8s}")

        # Error detail
        if s["error_message"]:
            print(f"       └─ error: {s['error_message'][:120]}")

        # Step events
        if show_events:
            events = _load_events(conn, s["step_id"])
            for ev in events:
                payload = ev["payload"]
                if isinstance(payload, str) and len(payload) > 100:
                    payload = payload[:100] + "..."
                print(f"       [{ev['event_type']}] {payload}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="View latest pipeline run details from runtime_events.db",
    )
    parser.add_argument("--last", type=int, default=1, help="Number of recent runs to show (default: 1)")
    parser.add_argument("--steps-only", action="store_true", help="Only show step table, no run header")
    parser.add_argument("--events", action="store_true", help="Also show step events")
    parser.add_argument("--db", type=str, default=None, help=f"Path to runtime_events.db (default: {DB_DEFAULT})")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else DB_DEFAULT
    if not db_path.exists():
        print(f"runtime_events.db not found at: {db_path}")
        print("No pipeline runs recorded yet. Run the pipeline first.")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    runs = _load_runs(conn, args.last)
    if not runs:
        print("No runs found in database.")
        return 0

    for run in runs:
        if args.steps_only:
            steps = _load_steps(conn, run["run_id"])
            for s in steps:
                print(f"{s['step_name']:30s} {s['handler']:22s} {s['status']:10s} {_fmt_ms(s['latency_ms'] or 0):>8s}")
        else:
            _print_run(run, conn, args.events)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
