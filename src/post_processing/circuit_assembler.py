import argparse
import glob
import json
import os
import sys
import time
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gen_workflow import assemble_circuits
from utils.interactive_stop import GracefulStopController
from utils.reporting import Logger
from utils.execution import (
    combine_coverage_artifacts,
    generate_coverage_report_from_data_file,
    load_compact_coverage_summary,
    build_hourly_coverage_timeline,
    format_hourly_coverage_points,
)
from utils.utils import generate_complexity_scatter_plots


def load_circuit_corpus(input_dir: str) -> List[str]:
    """Load all .py circuit files from input directory"""
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")

    circuits = sorted(glob.glob(os.path.join(input_dir, "*.py")))
    if not circuits:
        raise ValueError(f"No .py circuit files found in {input_dir}")

    return circuits

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Standalone assembly executor: combine and test quantum circuits."
    )
    parser.add_argument("input_dir", help="Directory containing valid circuit files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for assembled circuits, metrics, and logs",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=["guppy", "qiskit", "cirq", "pytket", "pennylane"],
        help="Circuit language",
    )
    parser.add_argument(
        "--n_assemble",
        type=int,
        default=10,
        help="Target number of assemblies to create (default: 10)",
    )
    parser.add_argument(
        "--n_circuits_per_assembly",
        type=int,
        default=3,
        help="Max circuits per assembly (default: 3)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Max parallel execution workers (default: 10)",
    )
    parser.add_argument(
        "--ks_low_threshold",
        type=float,
        default=0.0001,
        help="KS p-value threshold for flagging low results (default: 0.0001)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging including full code/error dumps",
    )
    try:

        args = parser.parse_args()

        # Setup logfile path
        base_dir = os.path.abspath(args.output_dir)
        os.environ["QUILLFUZZ_RUN_DIR"] = base_dir # Environment variable for test harnesses to find the run directory and logs

        coverage_artifacts_dir = os.path.join(base_dir, "coverage_artifacts")
        os.makedirs(coverage_artifacts_dir, exist_ok=True)
        os.environ["QUILLFUZZ_COVERAGE_ARTIFACT_DIR"] = coverage_artifacts_dir
        args.coverage_artifacts_dir = coverage_artifacts_dir
        logfile_path = os.path.join(base_dir, "assembly_execution.log")

        # Setup logger and stop controller
        logger = Logger(logfile_path)
        stop_controller = GracefulStopController(enabled=True, logger=logger)
        args.stop_controller = stop_controller
        stop_controller.start()

        logger.log(f"Assembly Executor started")
        logger.log(f"Input dir: {args.input_dir}, Language: {args.language}")
        logger.log(f"Output dir: {base_dir}")

        # Load circuit corpus
        circuits = load_circuit_corpus(args.input_dir)
        logger.log(f"Loaded {len(circuits)} circuit files from {args.input_dir}")

        run_start_epoch = time.time()

        # Execute assembly
        assembled_files, assembled_metrics, assembled_reports = assemble_circuits(
            "", circuits, args, base_dir, logger
        )

        run_end_epoch = time.time()

        # Log summary
        kept_count = len([m for m in assembled_metrics if m.get("success") or any(m.get("metrics", {}).get("execution", {}).values())])
        discarded_count = len(assembled_metrics) - kept_count
        logger.log(f"\n=== ASSEMBLY SUMMARY ===")
        logger.log(f"Submitted: {len(assembled_metrics)}")
        logger.log(f"Completed: {len(assembled_metrics)}")
        logger.log(f"Kept (interesting): {kept_count}")
        logger.log(f"Discarded (uninteresting): {discarded_count}")

        # Generate scatter plots
        if assembled_metrics:
            plots_dir = os.path.join(base_dir, "assembly_complexity_plots")
            generate_complexity_scatter_plots(
                assembled_metrics,
                plots_dir,
            )
            logger.log(f"Generated complexity scatter plots at {plots_dir}")

        combined_coverage_file = None
        coverage_summary_path = None
        coverage_summary_compact = {}
        coverage_hourly_points = []
        coverage_hourly_point_text = []
        coverage_combine_error = ""

        if os.path.isdir(coverage_artifacts_dir):
            combined_coverage_file, coverage_combine_error = combine_coverage_artifacts(coverage_artifacts_dir)
            if combined_coverage_file:
                coverage_summary_path = os.path.join(coverage_artifacts_dir, "coverage_summary.json")
                _, coverage_report_error = generate_coverage_report_from_data_file(
                    combined_coverage_file,
                    report_format="json",
                    output_path=coverage_summary_path,
                )
                if coverage_report_error:
                    coverage_combine_error = coverage_report_error
                elif os.path.exists(coverage_summary_path):
                    coverage_summary_compact = load_compact_coverage_summary(coverage_summary_path)

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
            logger.log("Coverage Hourly Timeline:")
            for line in coverage_hourly_point_text:
                logger.log(f"  {line}")

        # Generate summary JSON
        summary_path = os.path.join(base_dir, "assembly_summary.json")
        summary = {
            "language": args.language,
            "input_dir": args.input_dir,
            "output_dir": base_dir,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "circuits_loaded": len(circuits),
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
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.log(f"Generated assembly summary JSON: {summary_path}")

        print(f"\nAssembly complete.")
        print(f"Log: {logfile_path}")
        print(f"Summary: {summary_path}")
        return 0 if len(assembled_metrics) > 0 else 1

    finally:
        stop_controller.close()


if __name__ == "__main__":
    sys.exit(main())
