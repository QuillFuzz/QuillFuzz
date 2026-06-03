import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .execution import compile_generated_program, run_generated_program
from .reporting import build_file_result_metrics_row, build_file_result_summary, format_low_ks_values, populate_ks_test_metrics


@dataclass
class FileResult:
    file_path: str
    success: bool
    error: str
    metrics: Dict[str, Any]
    low_ks_test_levels: List[Tuple[str, float]]


def list_python_files(input_dir: str, recursive: bool = False) -> List[str]:
    if recursive:
        files: List[str] = []
        for root, _, filenames in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(os.path.join(root, filename))
        return sorted(files)

    return sorted(
        [
            os.path.join(input_dir, entry)
            for entry in os.listdir(input_dir)
            if entry.endswith(".py") and os.path.isfile(os.path.join(input_dir, entry))
        ]
    )


def build_metrics_csv_row(model_name: str, result: FileResult, compile_only: bool) -> Dict[str, Any]:
    return build_file_result_metrics_row(model_name, result, compile_only)


def build_summary(
    model_name: str,
    results: List[FileResult],
    compile_only: bool,
    duration: float,
    ks_low_threshold: float,
) -> Dict[str, Any]:
    return build_file_result_summary(
        model_name,
        results,
        compile_only,
        duration_seconds=duration,
        ks_low_threshold=ks_low_threshold,
        include_runtime_fields=True,
    )


def process_single_file(
    file_path: str,
    logger,
    verbose: bool,
    language: str,
    compile_only: bool,
    ks_low_threshold: float,
    file_index: int,
    coverage_artifact_dir: str = None,
) -> FileResult:
    filename = os.path.basename(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as file_handle:
            code = file_handle.read()
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
    has_compile_error = bool(compile_error and compile_error.strip())

    logger.log(f"--- Processing {filename} ---")
    logger.log(f"Metrics: {metrics}")

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
        ks_low_threshold=ks_low_threshold,
        coverage_artifact_dir=coverage_artifact_dir,
    )
    execution_metrics = execution_metrics or {}
    low_ks_test_levels = populate_ks_test_metrics(execution_metrics, run_stdout, ks_low_threshold)

    if low_ks_test_levels:
        logger.log(
            f"LOW KS detected for {filename} (threshold={ks_low_threshold}): {format_low_ks_values(low_ks_test_levels)}"
        )

    metrics["execution"] = execution_metrics
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