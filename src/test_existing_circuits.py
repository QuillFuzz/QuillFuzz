import os
import sys
import json
import time
import argparse
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local library
from utils.execution import run_generated_program, compile_generated_program
from utils.utils import generate_complexity_scatter_plots
from utils.reporting import (
    Logger,
    StreamingMetricsCsvWriter,
    extract_ks_test_results,
    find_low_ks_values,
    flatten_metrics_for_csv,
    ensure_clean_file,
    build_error_details,
)


@dataclass
class FileResult:
    file_path: str
    success: bool
    error: str
    metrics: Dict[str, Any]
    low_ks_test_levels: List[Tuple[str, float]]


def _validate_workers(workers: int) -> int:
    return max(1, workers)


def _list_python_files(input_dir: str) -> List[str]:
    return sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".py") and os.path.isfile(os.path.join(input_dir, f))
    ])


def _coverage_from_metrics(metrics: Dict[str, Any], compile_only: bool) -> float:
    if compile_only:
        compilation_metrics = metrics.get("compilation", {})
        if compilation_metrics:
            return float(compilation_metrics.get("coverage_percent", 0.0) or 0.0)
        return float(metrics.get("coverage_percent", 0.0) or 0.0)

    execution_metrics = metrics.get("execution", {})
    if execution_metrics:
        return float(execution_metrics.get("coverage_percent", 0.0) or 0.0)

    return float(metrics.get("coverage_percent", 0.0) or 0.0)


def _build_metrics_csv_row(model_name: str, result: FileResult, compile_only: bool) -> Dict[str, Any]:
    metrics_root = result.metrics or {}
    execution_metrics = metrics_root.get("execution", {})
    compilation_metrics = metrics_root.get("compilation", {})

    if compile_only:
        execution_metrics = {}

    return {
        "model": model_name,
        "file": os.path.basename(result.file_path),
        "success": result.success,
        "coverage_percent": _coverage_from_metrics(metrics_root, compile_only),
        **flatten_metrics_for_csv("execution", execution_metrics),
        **flatten_metrics_for_csv("compilation", compilation_metrics),
    }


def _build_summary(model_name: str, results: List[FileResult], compile_only: bool, duration: float, ks_low_threshold: float) -> Dict[str, Any]:
    total = len(results)
    successful = sum(1 for result in results if result.success)
    failed = total - successful

    successful_coverages = [_coverage_from_metrics(result.metrics, compile_only) for result in results if result.success]
    avg_coverage = sum(successful_coverages) / len(successful_coverages) if successful_coverages else 0.0

    def _error_payload(result: FileResult) -> Dict[str, str]:
        metrics_root = result.metrics or {}
        execution_metrics = metrics_root.get("execution", {})
        compilation_metrics = metrics_root.get("compilation", {})
        return build_error_details([
            {
                "error": execution_metrics.get("error_summary")
                or compilation_metrics.get("error_summary")
                or result.error,
                "error_full": execution_metrics.get("error_full")
                or compilation_metrics.get("error_full")
                or result.error,
            }
        ])

    return {
        "model": model_name,
        "total_files": total,
        "successful_files": successful,
        "failed_files": failed,
        "pass_rate": (successful / total * 100.0) if total else 0.0,
        "average_coverage_percent": avg_coverage,
        "files_with_low_ks": sum(1 for result in results if result.low_ks_test_levels),
        "ks_low_threshold": ks_low_threshold,
        "compile_only": compile_only,
        "duration_seconds": duration,
        "avg_time_per_file_seconds": (duration / total) if total else 0.0,
        "per_file_reports": [
            {
                "file": os.path.basename(result.file_path),
                "success": result.success,
                "coverage_percent": _coverage_from_metrics(result.metrics, compile_only),
                "error": _error_payload(result)["error"],
                "error_full": _error_payload(result)["error_full"],
                "low_ks_test_levels": result.low_ks_test_levels,
            }
            for result in results
        ],
    }


def process_single_file(
    file_path: str,
    logger: Logger,
    verbose: bool,
    language: str,
    compile_only: bool,
    ks_low_threshold: float,
    file_index: int,
) -> FileResult:
    filename = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as exc:
        err = f"Error reading {filename}: {exc}"
        logger.log(err)
        return FileResult(file_path=file_path, success=False, error=err, metrics={}, low_ks_test_levels=[])

    compile_error, compile_stdout, compilation_metrics, compilation_wrapped_code = compile_generated_program(
        code,
        language=language,
        source_file_path=file_path,
    )

    metrics: Dict[str, Any] = {
        "compilation": compilation_metrics or {},
        "execution": {},
    }

    logger.log(f"--- Processing {filename} ---")
    logger.log(f"Metrics: {metrics}")

    has_compile_error = bool(compile_error and compile_error.strip())

    if compile_stdout and not has_compile_error:
        logger.log(f"Compile Output:\n{compile_stdout}")

    if verbose:
        logger.log(f"Compilation Wrapped Code:\n{compilation_wrapped_code}")

    if has_compile_error:
        full_compile_error = (compilation_metrics or {}).get("error_full") or compile_error
        logger.log(f"Compilation Error:\n{full_compile_error}")
        return FileResult(file_path=file_path, success=False, error=compile_error, metrics=metrics, low_ks_test_levels=[])

    if compile_only:
        logger.log("Status: success")
        return FileResult(file_path=file_path, success=True, error="", metrics=metrics, low_ks_test_levels=[])

    run_error, run_stdout, execution_metrics, runtime_wrapped_code = run_generated_program(
        code,
        language=language,
        source_file_path=file_path,
        circuit_id=file_index,
    )
    execution_metrics = execution_metrics or {}

    ks_results = extract_ks_test_results(run_stdout)
    if ks_results:
        execution_metrics["ks_test_p_values"] = ks_results

    low_ks_test_levels = find_low_ks_values(ks_results, ks_low_threshold)
    if low_ks_test_levels:
        execution_metrics["low_ks_test_levels"] = low_ks_test_levels
        low_text = ", ".join([f"L{level}={value:.6g}" for level, value in low_ks_test_levels])
        logger.log(f"LOW KS detected for {filename} (threshold={ks_low_threshold}): {low_text}")

    metrics["execution"] = execution_metrics

    logger.log(f"Metrics: {metrics}")

    has_run_error = bool(run_error and run_error.strip())

    if run_stdout and not has_run_error:
        logger.log(f"Run Output:\n{run_stdout}")

    if verbose:
        logger.log(f"Runtime Wrapped Code:\n{runtime_wrapped_code}")

    if has_run_error:
        full_run_error = execution_metrics.get("error_full") or run_error
        logger.log(f"Runtime Error:\n{full_run_error}")
        return FileResult(
            file_path=file_path,
            success=False,
            error=run_error,
            metrics=metrics,
            low_ks_test_levels=low_ks_test_levels,
        )

    logger.log("Status: success")
    return FileResult(
        file_path=file_path,
        success=True,
        error="",
        metrics=metrics,
        low_ks_test_levels=low_ks_test_levels,
    )


def main():
    parser = argparse.ArgumentParser(description="Run tests on existing generated circuits without generating new ones.")
    parser.add_argument("input_dir", help="Directory containing .py files to test")
    parser.add_argument("--language", choices=["guppy", "qiskit"], default="guppy", help="Language of the files to test")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent workers")
    parser.add_argument("--verbose", action="store_true", help="Include wrapped code in the log output")
    parser.add_argument("--output-log", help="Optional path for the execution log file")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for report/json/csv/plots")
    parser.add_argument("--compile-only", action="store_true", help="Only compile the programs, do not run them.")
    parser.add_argument(
        "--ks-low-threshold",
        type=float,
        default=0.05,
        help="Threshold below which KS-test p-values are flagged as low in report/log outputs.",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory.")
        sys.exit(1)

    workers = _validate_workers(args.workers)
    files = _list_python_files(input_dir)

    if not files:
        print(f"No .py files found in {input_dir}")
        sys.exit(1)

    if os.path.basename(input_dir) == "assembled":
        os.environ["QUILLFUZZ_RUN_DIR"] = os.path.dirname(input_dir)
    else:
        os.environ["QUILLFUZZ_RUN_DIR"] = input_dir

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.output_log:
        log_path = os.path.abspath(args.output_log)
    else:
        log_path = os.path.join(output_dir, "test_execution.log")

    summary_json_path = os.path.join(output_dir, "test_summary.json")
    metrics_csv_path = os.path.join(output_dir, "execution_metrics.csv")
    plots_dir = os.path.join(output_dir, "_plots")

    model_name = os.path.basename(input_dir) or "retest_run"

    ensure_clean_file(log_path)

    logger = Logger(log_path)
    metrics_csv_writer = StreamingMetricsCsvWriter(metrics_csv_path)

    logger.log(f"Retest run started at {time.ctime()}")
    logger.log(f"Input directory: {input_dir}")
    logger.log(f"Language: {args.language}")
    logger.log(f"Compile only: {args.compile_only}")
    logger.log(f"Workers: {workers}")
    logger.log(f"Files discovered: {len(files)}")

    print(f"Found {len(files)} files in {input_dir}")
    print(f"Execution log: {log_path}")
    print(f"Using {workers} workers")

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
                metrics_csv_writer.append_row(_build_metrics_csv_row(model_name, result, args.compile_only))
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
                metrics_csv_writer.append_row(_build_metrics_csv_row(model_name, failed_result, args.compile_only))

    duration = time.time() - start_time

    summary = _build_summary(model_name, results, args.compile_only, duration, args.ks_low_threshold)
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
