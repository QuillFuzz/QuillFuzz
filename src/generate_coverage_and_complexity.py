import os
import sys
import time
import json
import csv
import tempfile
import argparse
import concurrent.futures
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from tqdm import tqdm

# Add both src and project root to path so local imports work in direct and wrapped runs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from utils.execution import run_generated_program, compile_generated_program
from utils.utils import generate_complexity_scatter_plots
from utils.reporting import Logger, StreamingMetricsCsvWriter, ensure_clean_file, build_error_details


@dataclass
class FileResult:
    file_path: str
    success: bool
    error: str
    metrics: Dict[str, Any]


def _list_python_files(input_dir: str, recursive: bool) -> List[str]:
    if recursive:
        files = []
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(root, filename))
        return sorted(files)

    return sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith(".py") and os.path.isfile(os.path.join(input_dir, f))
        ]
    )


def process_single_file(
    file_path: str,
    language: str,
    compile_only: bool,
    logger: Logger,
    verbose: bool,
    file_index: int,
) -> FileResult:
    filename = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as exc:
        err = f"Error reading {filename}: {exc}"
        logger.log(err)
        return FileResult(file_path=file_path, success=False, error=err, metrics={})

    compile_error, compile_stdout, compilation_metrics, compilation_wrapped_code = compile_generated_program(
        code,
        language=language,
        source_file_path=file_path,
    )

    metrics: Dict[str, Any] = {
        "compilation": compilation_metrics or {},
        "execution": {},
    }

    has_compile_error = bool(compile_error and compile_error.strip())

    if compile_only:
        logger.log(f"--- Processing {filename} ---")
        logger.log(f"Metrics: {metrics}")

        if compile_stdout and not has_compile_error:
            logger.log(f"Output:\n{compile_stdout}")

        if verbose:
            logger.log(f"Wrapped Code:\n{compilation_wrapped_code}")

        if has_compile_error:
            full_compile_error = (compilation_metrics or {}).get("error_full") or compile_error
            logger.log(f"Error:\n{full_compile_error}")
            return FileResult(file_path=file_path, success=False, error=compile_error, metrics=metrics)

        logger.log("Status: success")
        return FileResult(file_path=file_path, success=True, error="", metrics=metrics)

    if has_compile_error:
        logger.log(f"--- Processing {filename} ---")
        logger.log(f"Metrics: {metrics}")
        full_compile_error = (compilation_metrics or {}).get("error_full") or compile_error
        logger.log(f"Error:\n{full_compile_error}")
        if verbose:
            logger.log(f"Wrapped Code:\n{compilation_wrapped_code}")
        return FileResult(file_path=file_path, success=False, error=compile_error, metrics=metrics)

    run_error, run_stdout, execution_metrics, wrapped_code = run_generated_program(
        code,
        language=language,
        source_file_path=file_path,
        circuit_id=file_index,
    )
    metrics["execution"] = execution_metrics or {}

    logger.log(f"--- Processing {filename} ---")
    logger.log(f"Metrics: {metrics}")

    has_run_error = bool(run_error and run_error.strip())

    if run_stdout and not has_run_error:
        logger.log(f"Output:\n{run_stdout}")

    if verbose:
        logger.log(f"Wrapped Code:\n{wrapped_code}")

    if has_run_error:
        full_run_error = (execution_metrics or {}).get("error_full") or run_error
        logger.log(f"Error:\n{full_run_error}")
        return FileResult(file_path=file_path, success=False, error=run_error, metrics=metrics or {})

    logger.log("Status: success")
    return FileResult(file_path=file_path, success=True, error="", metrics=metrics or {})


def _coverage_from_metrics(metrics: Dict[str, Any], compile_only: bool) -> float:
    if compile_only:
        compilation_metrics = metrics.get("compilation", {})
        if compilation_metrics:
            return float(compilation_metrics.get("coverage_percent", 0.0))
        return float(metrics.get("coverage_percent", 0.0))

    execution_metrics = metrics.get("execution", {})
    if execution_metrics:
        return float(execution_metrics.get("coverage_percent", 0.0))

    return float(metrics.get("coverage_percent", 0.0))


def _build_summary(model_name: str, results: List[FileResult], compile_only: bool) -> Dict[str, Any]:
    total = len(results)
    successes = sum(1 for r in results if r.success)

    coverages = []
    for result in results:
        if result.success:
            coverages.append(_coverage_from_metrics(result.metrics, compile_only))

    avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    def _error_payload(result: FileResult) -> Dict[str, str]:
        metrics_root = result.metrics or {}
        execution_metrics = metrics_root.get("execution", {})
        compilation_metrics = metrics_root.get("compilation", {})
        error_details = build_error_details([
            {
                "error": execution_metrics.get("error_summary")
                or compilation_metrics.get("error_summary")
                or result.error,
                "error_full": execution_metrics.get("error_full")
                or compilation_metrics.get("error_full")
                or result.error,
            }
        ])
        return error_details

    return {
        "model": model_name,
        "total_files": total,
        "successful_files": successes,
        "failed_files": total - successes,
        "pass_rate": (successes / total * 100.0) if total else 0.0,
        "average_coverage_percent": avg_coverage,
        "per_file_reports": [
            {
                "file": os.path.basename(result.file_path),
                "success": result.success,
                "coverage_percent": _coverage_from_metrics(result.metrics, compile_only),
                "error": _error_payload(result)["error"],
                "error_full": _error_payload(result)["error_full"],
            }
            for result in results
        ],
    }


def _find_default_group_name(input_dir: str) -> str:
    base = os.path.basename(os.path.abspath(input_dir))
    return base or "coverage_run"


def _validate_workers(workers: int) -> int:
    if workers < 1:
        return 1
    return workers


def _parse_csv_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        try:
            if "." in stripped or "e" in stripped.lower():
                return float(stripped)
            return int(stripped)
        except ValueError:
            return stripped
    return value


def _build_metrics_from_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    execution_metrics: Dict[str, Any] = {}
    compilation_metrics: Dict[str, Any] = {}

    for key, raw_value in row.items():
        parsed_value = _parse_csv_value(raw_value)

        if key.startswith("execution_"):
            execution_metrics[key[len("execution_"):]] = parsed_value
            continue

        if key.startswith("compilation_"):
            compilation_metrics[key[len("compilation_"):]] = parsed_value
            continue

        if key in {"line_count", "function_count", "coverage_percent", "wall_time", "quality_score", "nesting_depth"}:
            execution_metrics[key] = parsed_value

    metrics: Dict[str, Any] = {}
    if execution_metrics:
        metrics["execution"] = execution_metrics
    if compilation_metrics:
        metrics["compilation"] = compilation_metrics

    if not metrics:
        metrics = {
            "execution": {
                "line_count": _parse_csv_value(row.get("line_count")),
                "function_count": _parse_csv_value(row.get("function_count")),
                "coverage_percent": _parse_csv_value(row.get("coverage_percent")),
                "wall_time": _parse_csv_value(row.get("wall_time")),
            }
        }

    return metrics


def _load_complexity_metrics_from_csv(csv_path: str, default_model: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue

            model_name = row.get("model") or default_model
            metrics = _build_metrics_from_csv_row(row)
            entries.append({
                "model": model_name,
                "metrics": metrics,
            })

    return entries


def _normalize_csv_row_for_current_fields(row: Dict[str, str]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}

    for key, value in row.items():
        if key in {"file", "success", "coverage_percent"}:
            normalized[key] = value
            continue

        if key.startswith("execution_") or key.startswith("compilation_"):
            normalized[key] = value

    return normalized


def _row_has_compilation_metrics(row: Dict[str, Any]) -> bool:
    for key, value in row.items():
        if key.startswith("compilation_") and str(value).strip() != "":
            return True
    return False


def _is_success_row(row: Dict[str, Any]) -> bool:
    value = row.get("success")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _enrich_row_with_compilation_metrics(row: Dict[str, Any], source_path: str, language: str) -> Dict[str, Any]:
    if _row_has_compilation_metrics(row):
        return row

    if not os.path.isfile(source_path):
        return row

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception:
        return row

    compile_error, _, compilation_metrics, _ = compile_generated_program(
        code,
        language=language,
        source_file_path=source_path,
    )

    if compilation_metrics:
        for key, value in compilation_metrics.items():
            row[f"compilation_{key}"] = value

    if compile_error and compile_error.strip():
        row["success"] = "False"

    return row


def _build_temp_pruned_csv_for_plotting(
    csv_path: str,
    output_dir: str,
    language: str,
    backfill_compilation: bool,
    csv_source_dir: Optional[str],
) -> tuple[str, int]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        input_rows = list(reader)

    pruned_rows = [_normalize_csv_row_for_current_fields(r) for r in input_rows if r]
    enriched_count = 0

    if backfill_compilation and csv_source_dir:
        csv_source_dir = os.path.abspath(csv_source_dir)
        compile_cache: Dict[str, Dict[str, Any]] = {}

        missing_rows = [
            row for row in pruned_rows
            if row.get("file") and not _row_has_compilation_metrics(row) and _is_success_row(row)
        ]

        for row in tqdm(missing_rows, desc="Backfilling compilation metrics", unit="file"):
            file_name = row.get("file", "")
            if not file_name:
                continue

            source_path = os.path.join(csv_source_dir, file_name)
            if source_path in compile_cache:
                cached = compile_cache[source_path]
                for key, value in cached.items():
                    row[key] = value
                if cached:
                    enriched_count += 1
                continue

            before_has_comp = _row_has_compilation_metrics(row)
            _enrich_row_with_compilation_metrics(row, source_path, language)

            cached = {k: v for k, v in row.items() if k.startswith("compilation_") and str(v).strip() != ""}
            compile_cache[source_path] = cached

            if not before_has_comp and _row_has_compilation_metrics(row):
                enriched_count += 1

    preferred = ["file", "success", "coverage_percent"]
    execution_fields = sorted({k for r in pruned_rows for k in r.keys() if k.startswith("execution_")})
    compilation_fields = sorted({k for r in pruned_rows for k in r.keys() if k.startswith("compilation_")})
    fieldnames = [k for k in preferred if any(k in r for r in pruned_rows)] + execution_fields + compilation_fields

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix="_pruned_metrics.csv",
        prefix="quillfuzz_",
        dir=output_dir,
        delete=False,
    ) as temp_csv:
        writer = csv.DictWriter(temp_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in pruned_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        temp_path = temp_csv.name

    return temp_path, enriched_count


def _flatten_metrics_for_csv(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
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


def _build_metrics_csv_row(
    model_name: str,
    result: FileResult,
    compile_only: bool,
) -> Dict[str, Any]:
    metrics_root = result.metrics or {}
    if compile_only:
        compilation_metrics = metrics_root.get("compilation", metrics_root)
        execution_metrics = metrics_root.get("execution", {})
    else:
        execution_metrics = metrics_root.get("execution", {})
        compilation_metrics = metrics_root.get("compilation", {})

        if not execution_metrics and "compilation" not in metrics_root:
            execution_metrics = metrics_root

    return {
        "model": model_name,
        "file": os.path.basename(result.file_path),
        "success": result.success,
        "coverage_percent": _coverage_from_metrics(metrics_root, compile_only),
        **_flatten_metrics_for_csv("execution", execution_metrics),
        **_flatten_metrics_for_csv("compilation", compilation_metrics),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate coverage reports and complexity scatter plots for existing circuit files "
            "(no generation/fixing phase), or generate plots directly from a metrics CSV file."
        )
    )
    parser.add_argument("input_path", help="Directory containing circuit .py files, or CSV file path when --input-type csv")
    parser.add_argument(
        "--input-type",
        choices=["py", "csv"],
        default="py",
        help="Input mode: 'py' runs compile/execute checks on Python files; 'csv' reads metrics CSV and generates plots only.",
    )
    parser.add_argument("--language", choices=["guppy", "qiskit"], default="guppy")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--recursive", action="store_true", help="Recursively find .py files")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only run compile checks (skips execution checks)",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store reports and plots",
    )
    parser.add_argument(
        "--csv-source-dir",
        type=str,
        default=None,
        help="When --input-type csv, optional directory containing source .py files to backfill missing compilation metrics for 3D plots.",
    )
    parser.add_argument(
        "--csv-backfill-compilation",
        action="store_true",
        help="When --input-type csv, enable recompiling source files to backfill missing compilation metrics (can be slow).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.input_type == "csv":
        if not os.path.isfile(input_path):
            print(f"Error: {input_path} is not a CSV file path")
            sys.exit(1)

        csv_model_name = os.path.splitext(os.path.basename(input_path))[0] or "csv_metrics"
        output_dir = args.output_dir or os.path.join(os.path.dirname(input_path), f"coverage_reports_{timestamp}")
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        complexity_plots_dir = os.path.join(output_dir, "complexity_plots")
        temp_csv_path, enriched_count = _build_temp_pruned_csv_for_plotting(
            input_path,
            output_dir,
            args.language,
            args.csv_backfill_compilation,
            args.csv_source_dir,
        )
        complexity_metrics = _load_complexity_metrics_from_csv(temp_csv_path, csv_model_name)

        if not complexity_metrics:
            print(f"No metrics rows found in {input_path}")
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
            sys.exit(1)

        generate_complexity_scatter_plots(complexity_metrics, complexity_plots_dir)
        print(f"CSV rows processed: {len(complexity_metrics)}")
        if args.csv_backfill_compilation:
            print(f"Compilation metrics backfilled: {enriched_count}")
        else:
            print("Compilation metrics backfilled: 0 (backfill disabled)")
        print(f"Complexity plots: {complexity_plots_dir}")
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        return

    input_dir = input_path
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    workers = _validate_workers(args.workers)
    files = _list_python_files(input_dir, args.recursive)

    if not files:
        print(f"No .py files found in {input_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(input_dir, f"coverage_reports_{timestamp}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "coverage_execution.log")
    summary_json_path = os.path.join(output_dir, "coverage_summary.json")
    metrics_csv_path = os.path.join(output_dir, "execution_metrics.csv")
    complexity_plots_dir = os.path.join(output_dir, "complexity_plots")

    model_name = _find_default_group_name(input_dir)

    ensure_clean_file(log_path)

    logger = Logger(log_path)
    metrics_csv_writer = StreamingMetricsCsvWriter(metrics_csv_path)

    logger.log(f"Coverage+Complexity run started at {time.ctime()}")
    logger.log(f"Input directory: {input_dir}")
    logger.log(f"Language: {args.language}")
    logger.log(f"Compile only: {args.compile_only}")
    logger.log(f"Workers: {workers}")
    logger.log(f"Files discovered: {len(files)}")

    results: List[FileResult] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file_path,
                args.language,
                args.compile_only,
                logger,
                args.verbose,
                index,
            ): file_path
            for index, file_path in enumerate(files)
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                result = future.result()
                results.append(result)
                metrics_csv_writer.append_row(_build_metrics_csv_row(model_name, result, args.compile_only))
            except Exception as exc:
                file_path = futures[future]
                logger.log(f"Unexpected error for {file_path}: {exc}")
                failed_result = FileResult(file_path=file_path, success=False, error=str(exc), metrics={})
                results.append(failed_result)
                metrics_csv_writer.append_row(_build_metrics_csv_row(model_name, failed_result, args.compile_only))

    complexity_metrics = [
        {"model": model_name, "metrics": result.metrics}
        for result in results
        if result.metrics
    ]
    if complexity_metrics:
        generate_complexity_scatter_plots(complexity_metrics, complexity_plots_dir)

    summary = _build_summary(model_name, results, args.compile_only)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_rows_written = metrics_csv_writer.row_count()
    logger.log(f"Metrics CSV rows written: {csv_rows_written}")
    logger.log(f"Metrics CSV path: {metrics_csv_path}")

    print(f"Processed files: {summary['total_files']}")
    print(f"Successful: {summary['successful_files']}")
    print(f"Average coverage (successful files): {summary['average_coverage_percent']:.2f}%")
    print(f"Complexity plots: {complexity_plots_dir}")
    print(f"Coverage summary JSON: {summary_json_path}")
    print(f"Execution log: {log_path}")
    print(f"Execution metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
