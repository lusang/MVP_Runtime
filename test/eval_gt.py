"""
Test script: run full-image pipeline on GT dataset, compare results.

Reads 20 image+json pairs from test/resource_of_GT, runs the pipeline on each,
compares the output against ground_truth, and prints a summary report.
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# ── Bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MVP_USE_DETECTOR"] = "0"
os.environ["MVP_DISABLE_PLANNER"] = "1"
# os.environ["GEMINI_MODEL"] = "gemini-2.5-pro"  # kept 2.0-flash baseline for now

from config.load_env import load_project_env
load_project_env()

GT_DIR = Path(__file__).resolve().parent / "resource_of_GT"
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "temp" / "show_temple.json"


# ── Attribute taxonomy (from show_temple.json) ──────────────────────────

QUALITY_KEYS = {"occlusion", "blur", "lighting", "background_clutter"}
SEMANTIC_KEYS = {"scenario", "person_action", "object_type", "is_package", "size_category"}
NEGATIVE_KEYS = {"pure_negative", "ambiguous", "open_set_negative"}

# For exact-match scoring: which values can we compare directly?
EXACT_MATCH_KEYS = QUALITY_KEYS | {"scenario", "person_action", "size_category", "is_package"}


def load_gt(path: Path) -> dict | None:
    """Load ground truth JSON, return the first object or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        objs = data.get("ground_truth", {}).get("objects", [])
        return objs[0] if objs else None
    except Exception:
        return None


def extract_gt_values(gt: dict) -> dict[str, str | list | bool]:
    """Flatten GT into a single {key: value} dict for easy comparison."""
    result = {}
    # Quality
    for k, v in gt.get("quality", {}).items():
        result[k] = v
    # Negative flags
    for k, v in gt.get("negative", {}).items():
        result[f"neg_{k}"] = v
    # Semantic attributes
    for k, v in gt.get("attributes", {}).items():
        result[k] = v
    return result


def _pluck_value(v):
    """Recursively extract the 'value' key from nested dicts.

    Also handles the known issue where Gemini attribute plugins return
    stringified dicts like "{'value': 'clear', 'label': 'clear', ...}"
    instead of actual values.
    """
    # String that looks like a Python dict → try to parse it
    if isinstance(v, str) and v.startswith("{'") and "'value'" in v:
        try:
            import ast
            parsed = ast.literal_eval(v)
            if isinstance(parsed, dict):
                return _pluck_value(parsed.get("value", parsed))
        except Exception:
            pass

    if isinstance(v, dict):
        inner = v.get("value", v)
        if isinstance(inner, dict):
            return _pluck_value(inner)
        return inner
    if isinstance(v, list):
        return [_pluck_value(i) for i in v]
    return v


def extract_pred_values(ann_obj) -> dict[str, str | list | bool]:
    """Flatten AnnotationResult object attributes into {key: value} dict.

    The pipeline merges quality + semantic + negative into one attributes dict.
    Recursively unwraps any residual {value: ..., confidence: ...} wrappers.
    """
    result = {}
    for k, v in ann_obj.attributes.items():
        if k.startswith("neg_"):
            result[k] = _pluck_value(v)
        elif k in QUALITY_KEYS | SEMANTIC_KEYS:
            result[k] = _pluck_value(v)
    return result


def value_match(gt_val, pred_val, key: str) -> bool:
    """Compare a single attribute value, handling types."""
    # Normalise booleans
    if isinstance(gt_val, bool) or key == "is_package":
        return bool(gt_val) == bool(pred_val)

    # Multi-select → compare as sets
    if isinstance(gt_val, list):
        gt_set = set(gt_val)
        pred_set = set(pred_val) if isinstance(pred_val, list) else {pred_val}
        return gt_set == pred_set

    # String comparison (case-insensitive for enum values)
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()


def compare_one(gt_values: dict, pred_values: dict) -> dict:
    """Compare GT vs prediction for one image. Return detailed results."""
    all_keys = set(gt_values.keys()) | set(pred_values.keys())
    results = {}
    for k in sorted(all_keys):
        gt_v = gt_values.get(k)
        pred_v = pred_values.get(k)

        if k not in gt_values:
            results[k] = {"status": "extra", "gt": None, "pred": pred_v}
        elif k not in pred_values:
            results[k] = {"status": "missing", "gt": gt_v, "pred": None}
        else:
            match = value_match(gt_v, pred_v, k)
            results[k] = {"status": "ok" if match else "mismatch",
                          "gt": gt_v, "pred": pred_v, "match": match}
    return results


# ── Main ─────────────────────────────────────────────────────────────────


async def main():
    from di.container import build_container

    # Gather test files
    json_files = sorted(GT_DIR.glob("*.json"))
    pairs = []
    for jf in json_files:
        stem = jf.stem
        # Find matching image (try .jpeg then .jpg)
        img = GT_DIR / f"{stem}.jpeg"
        if not img.exists():
            img = GT_DIR / f"{stem}.jpg"
        if img.exists():
            pairs.append((img, jf, stem))
        else:
            print(f"  SKIP {stem}: no matching image file")

    print(f"Found {len(pairs)} test pairs\n")

    # Init engine
    print("Initializing engine ...")
    container = build_container()
    engine = container.runtime_engine
    total_start = time.perf_counter()

    # Results accumulator
    per_image: list[dict] = []
    per_key_counts: dict[str, Counter] = {}

    for idx, (img_path, json_path, stem) in enumerate(pairs):
        gt = load_gt(json_path)
        if gt is None:
            print(f"  [{idx+1}/{len(pairs)}] {stem}  SKIP (bad GT)")
            continue

        gt_values = extract_gt_values(gt)

        # Run pipeline
        t0 = time.perf_counter()
        try:
            response = await engine.run(
                image_path=str(img_path),
                template_path=str(TEMPLATE_PATH),
            )
        except Exception as e:
            print(f"  [{idx+1}/{len(pairs)}] {stem}  FAIL ({e})")
            per_image.append({"stem": stem, "status": "error", "error": str(e)})
            continue

        elapsed = time.perf_counter() - t0

        # Get first annotated object
        ann = response.annotation_result
        if not ann.objects:
            print(f"  [{idx+1}/{len(pairs)}] {stem}  EMPTY (no objects)")
            per_image.append({"stem": stem, "status": "empty", "elapsed": elapsed})
            continue

        pred_values = extract_pred_values(ann.objects[0])
        comparisons = compare_one(gt_values, pred_values)

        # Update per-key counters
        for k, cr in comparisons.items():
            per_key_counts.setdefault(k, Counter())
            per_key_counts[k][cr["status"]] += 1

        # Summary for this image
        total = len(comparisons)
        ok_count = sum(1 for cr in comparisons.values() if cr.get("match"))
        mismatches = {k: cr for k, cr in comparisons.items()
                      if cr["status"] == "mismatch"}
        missing = {k: cr for k, cr in comparisons.items()
                   if cr["status"] == "missing"}
        extra = {k: cr for k, cr in comparisons.items()
                 if cr["status"] == "extra"}
        score = ok_count / total if total > 0 else 0.0

        line = (f"  [{idx+1}/{len(pairs)}] {stem}: "
                f"{ok_count}/{total} match ({score:.0%})  [{elapsed:.1f}s]")
        if mismatches:
            line += f"  MISMATCH: {', '.join(mismatches.keys())}"
        if missing:
            line += f"  MISSING: {', '.join(missing.keys())}"
        if extra:
            line += f"  EXTRA: {', '.join(extra.keys())}"
        print(line)

        per_image.append({
            "stem": stem,
            "status": "ok",
            "score": round(score, 4),
            "ok": ok_count,
            "total": total,
            "elapsed": round(elapsed, 2),
            "comparisons": comparisons,
        })

    total_elapsed = time.perf_counter() - total_start

    # ── Summary report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    ok_images = [p for p in per_image if p.get("status") == "ok"]
    error_images = [p for p in per_image if p.get("status") == "error"]
    empty_images = [p for p in per_image if p.get("status") == "empty"]

    print(f"\nTotal images: {len(pairs)}")
    print(f"  OK:     {len(ok_images)}")
    print(f"  Error:  {len(error_images)}")
    print(f"  Empty:  {len(empty_images)}")

    if ok_images:
        avg_score = sum(p["score"] for p in ok_images) / len(ok_images)
        avg_time = sum(p["elapsed"] for p in ok_images) / len(ok_images)
        print(f"\nAverage match rate: {avg_score:.1%}")
        print(f"Average latency:    {avg_time:.1f}s")
        print(f"Total time:         {total_elapsed:.0f}s")

        print(f"\n── Per-attribute match rates ──")
        for k in sorted(per_key_counts.keys()):
            c = per_key_counts[k]
            ok_ = c.get("ok", 0)
            mismatch_ = c.get("mismatch", 0)
            missing_ = c.get("missing", 0)
            extra_ = c.get("extra", 0)
            total_ = ok_ + mismatch_ + missing_
            rate = ok_ / total_ if total_ > 0 else 0.0
            extras = f" +{extra_} extra" if extra_ else ""
            print(f"  {k:25s}  {ok_:>2d}/{total_:>2d} ({rate:>5.1%})  "
                  f"mismatch={mismatch_}  missing={missing_}{extras}")

    # Top mismatches detail
    print(f"\n── Mismatch detail (first 5 per attribute) ──")
    shown = Counter()
    for p in ok_images:
        for k, cr in p["comparisons"].items():
            if cr.get("status") == "mismatch" and shown[k] < 5:
                if shown[k] == 0:
                    print(f"\n  [{k}]")
                print(f"    {p['stem']}: GT={cr['gt']!r}  PRED={cr['pred']!r}")
                shown[k] += 1

    print(f"\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
