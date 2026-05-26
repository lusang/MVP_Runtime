#!/usr/bin/env python3
"""
tools.tail_log — 查看实时运行日志。

Usage:
    python -m tools.tail_log                    # tail -20 (默认)
    python -m tools.tail_log --lines 50         # 查看最近 50 行
    python -m tools.tail_log --level WARNING    # 只看 WARNING 及以上
    python -m tools.tail_log --follow           # 持续跟踪（类似 tail -f）
    python -m tools.tail_log --batch RUN_ID     # 只看某个 batch 的日志
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "storage" / "runtime.log"


def _level_filter(level: str) -> re.Pattern:
    # Match lines with level >= given level
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if level.upper() not in levels:
        level = "INFO"
    idx = levels.index(level.upper())
    allowed = "|".join(levels[idx:])
    return re.compile(f"\\|\\s*({allowed})\\s*\\|")


def _read_lines(path: Path, n: int) -> list[str]:
    """Read last n lines from a file efficiently."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines[-n:]


def _follow(path: Path, level_filter: re.Pattern, batch_filter: str | None):
    """tail -f equivalent."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # end of file
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                line = line.rstrip("\n")
                if not level_filter.search(line):
                    continue
                if batch_filter and batch_filter not in line:
                    continue
                print(line)
                sys.stdout.flush()
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="View MVP runtime logs.")
    parser.add_argument("--lines", "-n", type=int, default=20, help="Number of recent lines (default: 20)")
    parser.add_argument("--level", "-l", type=str, default="INFO", help="Minimum log level (DEBUG|INFO|WARNING|ERROR)")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow log output (tail -f)")
    parser.add_argument("--batch", "-b", type=str, default=None, help="Filter by batch run_id")
    args = parser.parse_args()

    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        print("Start the server first (python main.py)")
        return 1

    level_re = _level_filter(args.level)

    if args.follow:
        _follow(LOG_FILE, level_re, args.batch)
        return 0

    lines = _read_lines(LOG_FILE, args.lines)
    for line in lines:
        line = line.rstrip("\n")
        if not level_re.search(line):
            continue
        if args.batch and args.batch not in line:
            continue
        print(line)

    log_size = LOG_FILE.stat().st_size
    print(f"\n--- {LOG_FILE} ({log_size/1024:.0f} KB) ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
