#!/usr/bin/env python3
"""
Backfill cost column in execution metrics CSV using execution.log entries.
Usage:
    python scripts/backfill_csv_costs.py --csv path/to/execution_metrics.csv --log path/to/execution.log [--inplace]

Searches the log for lines containing a filename and a cost token like:
"<file.py> ... Cost: $0.004056 | Tokens (In/Out/Total): 2146/13413/15559"
and writes a new CSV with a `cost` column populated.
"""
from pathlib import Path
import re
import csv
import argparse

COST_LINE_RE = re.compile(r"(?P<file>\S+\.py).*?Cost:\s*\$(?P<cost>[0-9.]+)\s*\|\s*Tokens\s*\(In/Out/Total\):\s*(?P<in>\d+)/(?:\d+)/(?:\d+)")


def extract_costs_from_log(log_path: Path):
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    costs = {}
    for m in COST_LINE_RE.finditer(text):
        fname = m.group('file')
        cost = float(m.group('cost'))
        # store last-seen cost for file (assume final is desired)
        costs[fname] = cost
    return costs


def backfill(csv_path: Path, log_path: Path, inplace: bool = False):
    costs = extract_costs_from_log(log_path)
    if not costs:
        print(f'No cost entries found in {log_path}')
    out_path = csv_path if inplace else csv_path.with_name(csv_path.stem + '_with_cost' + csv_path.suffix)

    with csv_path.open(newline='', encoding='utf-8') as inf:
        reader = csv.DictReader(inf)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if 'cost' not in fieldnames:
        fieldnames.append('cost')

    for row in rows:
        fname = (row.get('file') or '').strip()
        if not fname:
            continue
        if fname in costs:
            row['cost'] = costs[fname]
        else:
            # try matching by suffix if log uses relative paths
            if fname in costs:
                row['cost'] = costs[fname]
            else:
                row['cost'] = row.get('cost', '')

    with out_path.open('w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'Wrote backfilled CSV to {out_path} (inplace={inplace})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='Path to execution_metrics.csv')
    p.add_argument('--log', required=True, help='Path to execution.log')
    p.add_argument('--inplace', action='store_true', help='Overwrite the original CSV')
    args = p.parse_args()
    backfill(Path(args.csv), Path(args.log), inplace=args.inplace)
