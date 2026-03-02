import csv
import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple


KS_TEST_PATTERN = re.compile(r"Optimisation level\s+(\d+)\s+ks-test p-value:\s*([0-9eE+\-.]+)")


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
