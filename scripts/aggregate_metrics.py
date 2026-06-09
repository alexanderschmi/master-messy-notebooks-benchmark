import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

from src.core.output_layout import parse_run_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate mean time_taken and token usage per model-runner-order combination "
            "from output/**/**/**/**/**/metrics.json files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Path to benchmark output directory (default: output)",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Where to save the aggregated CSV (default: <output-dir>/metrics_report_aggregated.csv)",
    )
    return parser.parse_args()
def file_token_totals(usage: Dict) -> Dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    if not isinstance(usage, dict):
        return totals

    # Flat usage payload (token fields at the top level, e.g. simple/cot LLM runners).
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        totals["prompt_tokens"] = prompt_tokens
        totals["completion_tokens"] = completion_tokens
        totals["total_tokens"] = total_tokens
        return totals

    # OpenHands-style usage payload.
    accumulated = usage.get("accumulated_token_usage") if isinstance(usage, dict) else None
    if isinstance(accumulated, dict):
        prompt_tokens = int(accumulated.get("prompt_tokens") or 0)
        completion_tokens = int(accumulated.get("completion_tokens") or 0)
        total_tokens = int(accumulated.get("total_tokens") or 0)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        totals["prompt_tokens"] = prompt_tokens
        totals["completion_tokens"] = completion_tokens
        totals["total_tokens"] = total_tokens
        return totals

    # DSPy/OpenAI-compatible usage payload keyed by model/provider.
    for details in usage.values():
        if not isinstance(details, dict):
            continue

        prompt_tokens = int(details.get("prompt_tokens") or 0)
        completion_tokens = int(details.get("completion_tokens") or 0)
        total_tokens = int(details.get("total_tokens") or 0)
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += completion_tokens
        totals["total_tokens"] += total_tokens

    return totals


def aggregate(output_dir: Path):
    metrics_files = sorted(
        {
            *output_dir.glob("*/*/*/*/*/*/metrics.json"),
            *output_dir.glob("*/*/*/*/*/metrics.json"),
        }
    )
    grouped = defaultdict(
        lambda: {
            "count": 0,
            "time_taken_sum": 0.0,
            "prompt_tokens_sum": 0,
            "completion_tokens_sum": 0,
            "total_tokens_sum": 0,
        }
    )

    skipped = 0

    for metrics_path in metrics_files:
        try:
            model_dir = metrics_path.parent
            meta = parse_run_dir(model_dir, output_dir)
            if meta is None:
                skipped += 1
                continue

            model = meta["model"]
            runner = meta["runner"]
            notebook_order = meta.get("notebook_order", "original")

            with metrics_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            time_taken = float(payload.get("time_taken") or 0.0)
            usage = payload.get("usage") or {}
            token_totals = file_token_totals(usage)

            key = (model, runner, notebook_order)
            grouped[key]["count"] += 1
            grouped[key]["time_taken_sum"] += time_taken
            grouped[key]["prompt_tokens_sum"] += token_totals["prompt_tokens"]
            grouped[key]["completion_tokens_sum"] += token_totals["completion_tokens"]
            grouped[key]["total_tokens_sum"] += token_totals["total_tokens"]
        except Exception:
            skipped += 1

    rows = []
    for (model, runner, notebook_order), stats in sorted(grouped.items()):
        count = stats["count"]
        rows.append(
            {
                "model": model,
                "runner": runner,
                "notebook_order": notebook_order,
                "samples": count,
                "mean_time_taken": stats["time_taken_sum"] / count if count else 0.0,
                "mean_prompt_tokens": stats["prompt_tokens_sum"] / count if count else 0.0,
                "mean_completion_tokens": stats["completion_tokens_sum"] / count if count else 0.0,
                "mean_total_tokens": stats["total_tokens_sum"] / count if count else 0.0,
            }
        )

    return rows, len(metrics_files), skipped


def save_csv(rows, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "runner",
        "notebook_order",
        "samples",
        "mean_time_taken",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_total_tokens",
    ]

    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if not output_dir.exists() or not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    save_path = args.save_path.resolve() if args.save_path else (output_dir / "metrics_report_aggregated.csv")

    rows, scanned, skipped = aggregate(output_dir)
    save_csv(rows, save_path)

    print(f"Scanned metrics files: {scanned}")
    print(f"Aggregated combinations: {len(rows)}")
    print(f"Skipped files: {skipped}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()