import os
import sys
import json
import time
import argparse
import concurrent.futures
from typing import List

from tqdm import tqdm

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

# Import from local library
from utils.execution_pipeline import (
    FileResult,
    build_metrics_csv_row,
    build_summary,
    list_python_files,
    process_single_file,
)
from utils.utils import generate_complexity_scatter_plots
from utils.reporting import Logger, StreamingMetricsCsvWriter, ensure_clean_file

def main():
    parser = argparse.ArgumentParser(description="Run tests on existing generated circuits without generating new ones.")
    parser.add_argument("input_dir", help="Directory containing .py files to test")
    parser.add_argument("--language", choices=["guppy", "qiskit", "pytket", "pennylane"], help="Language of the files to test")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--verbose", action="store_true", help="Include wrapped code in the log output")
    parser.add_argument("--debug", action="store_true", help="Enable diff-testing debug mode and stage timing output")
    parser.add_argument("--output-log", help="Optional path for the execution log file")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for report/json/csv/plots")
    parser.add_argument("--compile-only", action="store_true", help="Only compile the programs, do not run them.")
    parser.add_argument(
        "--ks-low-threshold",
        type=float,
        default=0.01,
        help="Threshold below which KS-test p-values are flagged as low in report/log outputs.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory.")
        sys.exit(1)

    workers = max(1, args.workers)
    files = list_python_files(input_dir)

    if not files:
        print(f"No .py files found in {input_dir}")
        sys.exit(1)

    if os.path.basename(input_dir) == "assembled":
        os.environ["QUILLFUZZ_RUN_DIR"] = os.path.dirname(input_dir)
    else:
        os.environ["QUILLFUZZ_RUN_DIR"] = input_dir

    os.environ["QUILLFUZZ_DEBUG"] = "1" if args.debug else "0"

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.output_log:
        log_path = os.path.abspath(args.output_log)
    else:
        log_path = os.path.join(output_dir, "retest_execution.log")

    summary_json_path = os.path.join(output_dir, "retest_summary.json")
    metrics_csv_path = os.path.join(output_dir, "retest_execution_metrics.csv")
    plots_dir = os.path.join(output_dir, "_retest_plots")

    model_name = os.path.basename(input_dir) or "retest_run"

    ensure_clean_file(log_path)

    logger = Logger(log_path)
    metrics_csv_writer = StreamingMetricsCsvWriter(metrics_csv_path)

    logger.log(f"Retest run started at {time.ctime()}")
    logger.log(f"Input directory: {input_dir}")
    logger.log(f"Language: {args.language}")
    logger.log(f"Compile only: {args.compile_only}")
    logger.log(f"Debug: {args.debug}")
    logger.log(f"Workers: {workers}")
    logger.log(f"Files discovered: {len(files)}")

    print(f"Found {len(files)} files in {input_dir}")
    print(f"Execution log: {log_path}")
    print(f"Using {workers} workers")
    if args.debug:
        print("Debug mode enabled: diff-testing stages will print timing output")

    start_time = time.time()
    results: List[FileResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file_path,
                logger,
                args.verbose,
                args.language,
                args.compile_only,
                args.ks_low_threshold,
                index,
            ): file_path
            for index, file_path in enumerate(files)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), unit="files"):
            try:
                result = future.result()
                results.append(result)
                metrics_csv_writer.append_row(build_metrics_csv_row(model_name, result, args.compile_only))
            except Exception as exc:
                failed_path = futures[future]
                logger.log(f"Unexpected error for {failed_path}: {exc}")
                failed_result = FileResult(
                    file_path=failed_path,
                    success=False,
                    error=str(exc),
                    metrics={},
                    low_ks_test_levels=[],
                )
                results.append(failed_result)
                metrics_csv_writer.append_row(build_metrics_csv_row(model_name, failed_result, args.compile_only))

    duration = time.time() - start_time

    summary = build_summary(model_name, results, args.compile_only, duration, args.ks_low_threshold)
    with open(summary_json_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    all_metrics = [
        {"model": model_name, "metrics": result.metrics}
        for result in results
        if result.metrics
    ]

    if all_metrics:
        os.makedirs(plots_dir, exist_ok=True)
        generate_complexity_scatter_plots(all_metrics, plots_dir)

    logger.log(f"Metrics CSV rows written: {metrics_csv_writer.row_count()}")
    logger.log(f"Metrics CSV path: {metrics_csv_path}")

    print(f"Finished. {summary['successful_files']}/{summary['total_files']} passed.")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Execution metrics CSV: {metrics_csv_path}")
    if all_metrics:
        print(f"Complexity plots: {plots_dir}")
    else:
        print("Complexity plots: skipped (no metrics available)")


if __name__ == "__main__":
    main()
