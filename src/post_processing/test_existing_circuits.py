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
from utils.execution import (
    build_hourly_coverage_timeline,
    format_hourly_coverage_points,
    combine_coverage_artifacts,
    generate_coverage_report_from_data_file,
    load_compact_coverage_summary,
)
from utils.gen_workflow import assemble_circuits
from utils.interactive_stop import GracefulStopController
from utils.utils import generate_complexity_scatter_plots
from utils.reporting import Logger, StreamingMetricsCsvWriter, ensure_clean_file

def main():
    parser = argparse.ArgumentParser(description="Run tests on existing generated circuits without generating new ones.")
    parser.add_argument("input_dir", help="Directory containing .py files to test")
    parser.add_argument("--language", choices=["guppy", "qiskit", "cirq", "pytket", "pennylane"], help="Language of the files to test")
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
    parser.add_argument(
        "--coverage-artifacts-dir",
        type=str,
        default=None,
        help="Directory to save per-file .coverage artifacts for timeline generation. "
             "If omitted, no coverage timeline is produced.",
    )
    parser.add_argument(
        "--assemble",
        action="store_true",
        help="Enable the assembly stage after running existing tests.",
    )
    parser.add_argument(
        "--n-assemble",
        "--n_assemble",
        type=int,
        default=10,
        dest="n_assemble",
        help="Target number of assemblies to create (default: 10)",
    )
    parser.add_argument(
        "--n-circuits-per-assembly",
        "--n_circuits_per_assembly",
        type=int,
        default=3,
        dest="n_circuits_per_assembly",
        help="Max circuits per assembly (default: 3)",
    )
    parser.add_argument(
        "--max-workers",
        "--max_workers",
        type=int,
        default=None,
        dest="max_workers",
        help="Max parallel execution workers for assembly (defaults to value of --workers)",
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

    if args.max_workers is None:
        args.max_workers = args.workers

    if args.assemble:
        if not args.language:
            print("Error: --language must be specified when using --assemble.")
            sys.exit(1)

    coverage_artifacts_dir = None
    if args.coverage_artifacts_dir:
        coverage_artifacts_dir = os.path.abspath(args.coverage_artifacts_dir)
    elif args.assemble:
        coverage_artifacts_dir = os.path.join(output_dir, "coverage_artifacts")

    if coverage_artifacts_dir:
        os.makedirs(coverage_artifacts_dir, exist_ok=True)
        os.environ["QUILLFUZZ_COVERAGE_ARTIFACT_DIR"] = coverage_artifacts_dir
        args.coverage_artifacts_dir = coverage_artifacts_dir

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
                coverage_artifacts_dir,
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

    end_time = time.time()
    duration = end_time - start_time

    coverage_timeline_points = []
    if coverage_artifacts_dir:
        timeline_points, timeline_error = build_hourly_coverage_timeline(
            coverage_artifacts_dir,
            run_start_epoch=start_time,
            run_end_epoch=end_time,
        )
        if timeline_points:
            coverage_timeline_points = timeline_points
            logger.log("Coverage Timeline:")
            for line in format_hourly_coverage_points(timeline_points):
                logger.log(f"  {line}")
        elif timeline_error:
            logger.log(f"Coverage timeline error: {timeline_error}")

    summary = build_summary(model_name, results, args.compile_only, duration, args.ks_low_threshold)
    summary["coverage_timeline_points"] = coverage_timeline_points
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
    if coverage_artifacts_dir:
        if coverage_timeline_points:
            print(f"Coverage timeline: {len(coverage_timeline_points)} points (see summary JSON)")
        else:
            print("Coverage timeline: no data collected")

    if args.assemble:
        print("\nStarting assembly phase...")
        assembly_log_path = os.path.join(output_dir, "assembly_execution.log")
        ensure_clean_file(assembly_log_path)
        assembly_logger = Logger(assembly_log_path)

        stop_controller = GracefulStopController(enabled=True, logger=assembly_logger)
        args.stop_controller = stop_controller
        stop_controller.start()

        assembly_logger.log("Assembly Executor started")
        assembly_logger.log(f"Input dir: {input_dir}, Language: {args.language}")
        assembly_logger.log(f"Output dir: {output_dir}")
        assembly_logger.log(f"Loaded {len(files)} circuit files from {input_dir}")

        run_start_epoch = time.time()

        try:
            # Execute assembly
            assembled_files, assembled_metrics, assembled_reports = assemble_circuits(
                "", files, args, output_dir, assembly_logger
            )

            run_end_epoch = time.time()

            # Log summary
            kept_count = len([
                m for m in assembled_metrics
                if m.get("success") or any(m.get("metrics", {}).get("execution", {}).values())
            ])
            discarded_count = len(assembled_metrics) - kept_count
            assembly_logger.log("\n=== ASSEMBLY SUMMARY ===")
            assembly_logger.log(f"Submitted: {len(assembled_metrics)}")
            assembly_logger.log(f"Completed: {len(assembled_metrics)}")
            assembly_logger.log(f"Kept (interesting): {kept_count}")
            assembly_logger.log(f"Discarded (uninteresting): {discarded_count}")

            # Generate scatter plots
            if assembled_metrics:
                assembly_plots_dir = os.path.join(output_dir, "assembly_complexity_plots")
                generate_complexity_scatter_plots(assembled_metrics, assembly_plots_dir)
                assembly_logger.log(f"Generated complexity scatter plots at {assembly_plots_dir}")

            combined_coverage_file = None
            coverage_summary_path = None
            coverage_summary_compact = {}
            coverage_hourly_points = []
            coverage_hourly_point_text = []
            coverage_combine_error = ""

            if coverage_artifacts_dir and os.path.isdir(coverage_artifacts_dir):
                combined_file, combine_err = combine_coverage_artifacts(coverage_artifacts_dir)
                if combined_file:
                    combined_coverage_file = combined_file
                    coverage_summary_path = os.path.join(coverage_artifacts_dir, "coverage_summary.json")
                    _, report_error = generate_coverage_report_from_data_file(
                        combined_coverage_file,
                        report_format="json",
                        output_path=coverage_summary_path,
                    )
                    if report_error:
                        coverage_combine_error = report_error
                    elif os.path.exists(coverage_summary_path):
                        coverage_summary_compact = load_compact_coverage_summary(coverage_summary_path)
                elif combine_err:
                    coverage_combine_error = combine_err

                hourly_points, hourly_error = build_hourly_coverage_timeline(
                    coverage_artifacts_dir,
                    run_start_epoch=run_start_epoch,
                    run_end_epoch=run_end_epoch,
                )
                if hourly_points:
                    coverage_hourly_points = hourly_points
                    coverage_hourly_point_text = format_hourly_coverage_points(hourly_points)
                elif hourly_error and not coverage_combine_error:
                    coverage_combine_error = hourly_error

            if coverage_hourly_point_text:
                assembly_logger.log("Coverage Hourly Timeline:")
                for line in coverage_hourly_point_text:
                    assembly_logger.log(f"  {line}")

            # Generate summary JSON
            assembly_summary_path = os.path.join(output_dir, "assembly_summary.json")
            assembly_summary = {
                "language": args.language,
                "input_dir": input_dir,
                "output_dir": output_dir,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "circuits_loaded": len(files),
                "assembled_submitted": len(assembled_metrics),
                "assembled_completed": len(assembled_metrics),
                "assembled_kept": kept_count,
                "assembled_discarded": discarded_count,
                "coverage_artifacts_dir": coverage_artifacts_dir,
                "coverage_combined_data_file": combined_coverage_file,
                "coverage_summary_path": coverage_summary_path,
                "coverage_summary_compact": coverage_summary_compact,
                "coverage_hourly_points": coverage_hourly_points,
                "coverage_hourly_points_text": coverage_hourly_point_text,
                "coverage_combine_error": coverage_combine_error,
                "per_file_reports": assembled_reports,
            }
            with open(assembly_summary_path, "w", encoding="utf-8") as f:
                json.dump(assembly_summary, f, indent=2)
            assembly_logger.log(f"Generated assembly summary JSON: {assembly_summary_path}")

            print(f"\nAssembly complete.")
            print(f"Log: {assembly_log_path}")
            print(f"Summary: {assembly_summary_path}")
            if assembled_metrics:
                print(f"Complexity plots: {os.path.join(output_dir, 'assembly_complexity_plots')}")
            else:
                print("Complexity plots: skipped (no metrics available)")

        finally:
            stop_controller.close()


if __name__ == "__main__":
    main()
