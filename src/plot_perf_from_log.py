#!/usr/bin/env python3
from pathlib import Path
import re, sys, argparse

# ensure repo root is importable when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.utils import generate_summary_plot

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log', help='path to .log file')
    args = p.parse_args()
    txt = Path(args.log).read_text()
    stats = parse_all_perf(txt)
    out = Path(args.log).resolve().parent / 'plots' / 'performance'
    if stats:
        generate_summary_plot(stats, str(out))
    else:
        print('No PERFORMANCE SUMMARY blocks found in', args.log)


if __name__ == '__main__':
    main()
