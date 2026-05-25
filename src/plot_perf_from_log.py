#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path
import re, sys, argparse

# ensure repo root is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.utils import generate_summary_plot


def _to_float(value):
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None

def parse_perf_block(text: str) -> dict:
    d = {}
    m = re.search(r"PERFORMANCE SUMMARY for\s*(.+)", text)
    if m:
        d['model'] = m.group(1).strip()

    def g(pat, cast=int):
        m = re.search(pat, text, re.M)
        if not m:
            return None
        s = m.group(1).replace(',', '')
        try:
            return cast(s)
        except:
            return None

    d['total_programs'] = g(r"Target Number of Programs\s*:\s*(\d+)") or 0
    d['valid_programs'] = g(r"Total Valid Programs\s*:\s*(\d+)") or 0
    d['total_time'] = g(r"Total Time Taken\s*:\s*([0-9.]+) seconds", float) or 0.0
    d['avg_time_per_valid_prog'] = g(r"Avg Time per Valid Prog\s*:\s*([0-9.]+) seconds", float) or 0.0
    d['avg_quality_score'] = g(r"Avg Quality Score\s*:\s*([0-9.]+)", float) or 0.0
    d['total_cost'] = g(r"Total Cost.*:\s*\$?([0-9.]+)", float) or 0.0
    d['total_prompt_tokens'] = g(r"Total Prompt Tokens\s*:\s*([0-9,]+)") or 0
    d['total_completion_tokens'] = g(r"Total Completion Tokens\s*:\s*([0-9,]+)") or 0
    d['total_tokens'] = g(r"Total Tokens\s*:\s*([0-9,]+)") or 0
    return d


def parse_all_perf(text: str):
    # split on delimiter lines of equals; keep blocks that contain PERFORMANCE SUMMARY
    blocks = re.split(r"=+\n", text)
    stats = []
    for b in blocks:
        if 'PERFORMANCE SUMMARY for' in b:
            s = parse_perf_block(b)
            if s.get('model'):
                stats.append(s)
    return stats


def parse_csv_metrics(csv_path: Path):
    per_model = defaultdict(lambda: {
        "model": "",
        "total_programs": 0,
        "valid_programs": 0,
        "total_time": 0.0,
        "total_cost": 0.0,
        "quality_scores": [],
    })

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = (row.get("model") or "unknown").strip() or "unknown"
            record = per_model[model]
            record["model"] = model
            record["total_programs"] += 1

            compilation_quality = _to_float(row.get("compilation_quality_score"))
            has_compile_error = bool((row.get("compilation_error_summary") or "").strip() or (row.get("compilation_error_full") or "").strip())
            compile_valid = compilation_quality is not None and compilation_quality > 0 and not has_compile_error

            if not compile_valid:
                continue

            record["valid_programs"] += 1
            record["quality_scores"].append(compilation_quality)

            comp_wall = _to_float(row.get("compilation_wall_time")) or 0.0
            exec_wall = _to_float(row.get("execution_wall_time")) or 0.0
            record["total_time"] += comp_wall + exec_wall

    stats = []
    for record in per_model.values():
        quality_scores = record.pop("quality_scores")
        record["avg_quality_score"] = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        stats.append(record)

    return sorted(stats, key=lambda item: item["model"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_path', help='path to a .log or .csv file')
    p.add_argument('--csv', action='store_true', help='Parse execution_metrics.csv instead of performance log blocks')
    args = p.parse_args()

    input_path = Path(args.input_path)
    use_csv = args.csv or input_path.suffix.lower() == '.csv'
    out = input_path.resolve().parent / 'plots' / 'performance'

    if use_csv:
        stats = parse_csv_metrics(input_path)
        if stats:
            generate_summary_plot(stats, str(out))
        else:
            print('No valid CSV rows found in', args.input_path)
        return

    txt = input_path.read_text()
    stats = parse_all_perf(txt)
    if stats:
        generate_summary_plot(stats, str(out))
    else:
        print('No PERFORMANCE SUMMARY blocks found in', args.input_path)


if __name__ == '__main__':
    main()
