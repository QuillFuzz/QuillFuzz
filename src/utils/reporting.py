import csv
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple


KS_TEST_PATTERN = re.compile(r"Optimisation level\s+(\d+)\s+ks-test p-value:\s*([0-9eE+\-.]+)")


def _has_compilation_error(compilation_metrics: Dict[str, Any]) -> bool:
    return bool((compilation_metrics or {}).get("error_summary") or (compilation_metrics or {}).get("error_full"))

def build_metrics_row(model: str, file_name: str, success: bool, execution_metrics: Dict[str, Any], compilation_metrics: Dict[str, Any], cost: float = 0.0) -> Dict[str, Any]:
    """Build a single metrics row for CSV export. Includes optional `cost` field."""
    row = {
        "model": model,
        "file": file_name,
        "success": success,
        "coverage_percent": execution_metrics.get("coverage_percent", 0.0) if execution_metrics else 0.0,
        "cost": cost,
    }
    # Merge flattened metrics (compilation then execution) while keeping cost at top-level
    row.update(flatten_metrics_for_csv("compilation", compilation_metrics or {}))
    row.update(flatten_metrics_for_csv("execution", execution_metrics or {}))
    return row


def build_phase_summary(
        phase_name: str,
        model: str,
        total_programs: int,
        total_time: float,
        stats_list: List[Any],
        report_entries: List[Dict[str, Any]],
        ks_low_threshold: float,
) -> Tuple[str, Dict[str, Any]]:
    """Build the formatted summary log and summary dict for a run phase."""
    total_cost = sum(getattr(stats, "cost", 0.0) for stats in stats_list)
    quality_scores = [
        stats.quality_score
        for stats in stats_list
        if getattr(stats, "quality_score", None) is not None
        and not _has_compilation_error(getattr(stats, "metrics", {}).get("compilation", {}))
    ]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    total_prompt_tokens = sum(getattr(stats, "prompt_tokens", 0) for stats in stats_list)
    total_completion_tokens = sum(getattr(stats, "completion_tokens", 0) for stats in stats_list)
    total_tokens = sum(getattr(stats, "total_tokens", 0) for stats in stats_list)

    successful_entries = [entry for entry in report_entries if entry.get("success")]
    valid_count = len(successful_entries)
    failed_programs = max(0, total_programs - valid_count)
    avg_time_per_valid = total_time / valid_count if valid_count else 0.0

    successful_coverages = [float(entry.get("coverage_percent", 0.0) or 0.0) for entry in successful_entries]
    avg_coverage_percent = sum(successful_coverages) / len(successful_coverages) if successful_coverages else 0.0
    low_ks_file_count = sum(1 for entry in report_entries if entry.get("low_ks_test_levels"))

    summary = {
            "model": model,
            "total_cost": total_cost,
            "total_time": total_time,
            "total_programs": total_programs,
            "valid_programs": valid_count,
            "failed_programs": failed_programs,
            "avg_quality_score": avg_quality,
            "avg_coverage_percent": avg_coverage_percent,
            "low_ks_file_count": low_ks_file_count,
            "ks_low_threshold": ks_low_threshold,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
    }

    summary_log = f"""
============================================================
    {phase_name} SUMMARY for {model}
------------------------------------------------------------
    Target Number of Programs : {total_programs}
    Total Valid Programs     : {valid_count}
    Total Time Taken         : {total_time:.2f} seconds
    Avg Time per Valid Prog  : {avg_time_per_valid:.2f} seconds
    Avg Quality Score        : {avg_quality:.4f}
------------------------------------------------------------
    Total Cost (Estimated)   : ${total_cost:.6f}
    Total Prompt Tokens      : {total_prompt_tokens}
    Total Completion Tokens  : {total_completion_tokens}
    Total Tokens             : {total_tokens}
============================================================
"""

    return summary_log, summary


class Logger:
    def __init__(self, logfile_path: str):
        self.logfile_path = logfile_path
        self.lock = threading.Lock()

    def log(self, message: str):
        if not self.logfile_path:
            return

        with self.lock:
            if not message.endswith("\n"):
                message += "\n"
            with open(self.logfile_path, "a", encoding="utf-8") as logfile:
                logfile.write(message)


class StreamingMetricsCsvWriter:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.fieldnames: List[str] = []
        self.rows: List[Dict[str, Any]] = []

        with open(self.csv_path, "w", encoding="utf-8", newline=""):
            pass

    def append_row(self, row: Dict[str, Any]):
        self.rows.append(row)

        for key in row.keys():
            if key not in self.fieldnames:
                self.fieldnames.append(key)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for existing_row in self.rows:
                writer.writerow({name: existing_row.get(name, "") for name in self.fieldnames})

    def row_count(self) -> int:
        return len(self.rows)


def flatten_metrics_for_csv(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        if key in {"error", "error_full", "error_summary", "coverage_error"}:
            continue
        col = f"{prefix}_{key}"
        if isinstance(value, (dict, list)):
            flattened[col] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[col] = value
    return flattened


def extract_ks_test_results(output: str) -> Dict[str, float]:
    if not output:
        return {}

    results: Dict[str, float] = {}
    for match in KS_TEST_PATTERN.finditer(output):
        level = match.group(1)
        raw_value = match.group(2)
        try:
            results[level] = float(raw_value)
        except ValueError:
            continue

    return results


def find_low_ks_values(ks_results: Dict[str, float], threshold: float) -> List[Tuple[str, float]]:
    lows = [(level, value) for level, value in (ks_results or {}).items() if value < threshold]
    return sorted(lows, key=lambda item: int(item[0]))


def populate_ks_test_metrics(metrics: Dict[str, Any], output: str, threshold: float) -> List[Tuple[str, float]]:
    ks_results = extract_ks_test_results(output)
    if not ks_results:
        return []

    metrics["ks_test_p_values"] = ks_results
    low_ks_values = find_low_ks_values(ks_results, threshold)
    if low_ks_values:
        metrics["low_ks_test_levels"] = low_ks_values

    return low_ks_values


def format_low_ks_values(low_ks_values: List[Tuple[str, float]]) -> str:
    return ", ".join(f"L{level}={value:.6g}" for level, value in low_ks_values)


def ensure_clean_file(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def _full_error_text(error: Any) -> str:
    if error is None:
        return ""
    return str(error).strip()


def _summary_from_error(error: Any, max_len: Optional[int] = None) -> str:
    raw = _full_error_text(error)
    if not raw:
        return ""

    lines = [line.strip() for line in raw.splitlines() if line and line.strip()]
    summary = lines[-1] if lines else raw.replace("\n", " ").strip()
    if max_len is not None and max_len > 0 and len(summary) > max_len:
        return f"{summary[:max_len]}..."
    return summary


def build_error_details(errors: List[Any], max_summary_len: Optional[int] = None) -> Dict[str, str]:
    if not errors:
        return {"error": "", "error_full": ""}

    summary_parts: List[str] = []
    full_parts: List[str] = []

    for err in errors:
        if isinstance(err, dict):
            summary_text = _full_error_text(err.get("error"))
            full_text = _full_error_text(err.get("error_full")) or summary_text
        else:
            summary_text = _summary_from_error(err, max_summary_len)
            full_text = _full_error_text(err)

        if summary_text:
            if max_summary_len is not None and max_summary_len > 0 and len(summary_text) > max_summary_len:
                summary_text = f"{summary_text[:max_summary_len]}..."
            summary_parts.append(summary_text)
        if full_text:
            full_parts.append(full_text)

    return {
        "error": " | ".join(part for part in summary_parts if part),
        "error_full": "\n\n---\n\n".join(part for part in full_parts if part),
    }


def summarize_errors(errors: List[str], max_len: Optional[int] = None) -> str:
    if not errors:
        return ""

    return build_error_details(errors, max_summary_len=max_len).get("error", "")


def append_rows_to_csv(csv_path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return

    existing_rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    all_rows = existing_rows + rows

    with open(csv_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
