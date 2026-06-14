import os
import sys
import subprocess
import tempfile
import time
import json
import shutil
from typing import Any, Dict, List, Optional, Tuple
from .utils import strip_markdown_syntax, parse_time_metrics
import re
from .ast_ops import (
    wrap_for_compilation_guppy,
    wrap_for_testing_guppy,
    wrap_for_compilation_qiskit,
    wrap_for_testing_qiskit,
    wrap_for_compilation_cirq,
    wrap_for_testing_cirq,
    wrap_for_compilation_pytket,
    wrap_for_testing_pytket,
    wrap_for_compilation_pennylane,
    wrap_for_testing_pennylane,
    get_code_complexity_metrics,
)
from .reporting import summarize_error_text

# Default timeouts in seconds
DEFAULT_EXECUTION_TIMEOUT = 300
DEFAULT_COMPILE_TIMEOUT = 60
DEFAULT_REPORT_TIMEOUT = 60

LANGUAGE_DEFAULT_SOURCES = {
    "guppy": "guppylang_internals",
    "qiskit": "qiskit",
    "cirq": "cirq",
    "pytket": "pytket",
    "pennylane": "pennylane",
}


def _default_coverage_source(language: str) -> str:
    return LANGUAGE_DEFAULT_SOURCES.get(language, "qiskit")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_execution_env(
    coverage_file: str,
    source_file_path: Optional[str] = None,
    ks_low_threshold: Optional[float] = None,
) -> Dict[str, str]:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = coverage_file

    if source_file_path:
        env["QUILLFUZZ_SOURCE_FILE"] = os.path.abspath(source_file_path)

    if ks_low_threshold is not None:
        env["QUILLFUZZ_KS_LOW_THRESHOLD"] = str(ks_low_threshold)

    project_root = _project_root()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root
    )
    return env


def _coverage_artifact_dir(explicit_dir: Optional[str] = None) -> Optional[str]:
    artifact_dir = explicit_dir or os.environ.get("QUILLFUZZ_COVERAGE_ARTIFACT_DIR")
    if not artifact_dir:
        return None
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def _coverage_artifact_path(artifact_dir: str, coverage_file: str) -> str:
    return os.path.join(artifact_dir, os.path.basename(coverage_file))


def _persist_coverage_artifact(coverage_file: str, artifact_dir: Optional[str]) -> Optional[str]:
    if not artifact_dir:
        return None
    try:
        artifact_path = _coverage_artifact_path(artifact_dir, coverage_file)
        shutil.copy2(coverage_file, artifact_path)
        return artifact_path
    except Exception:
        return None


def list_coverage_artifacts(artifact_dir: str) -> list[str]:
    if not artifact_dir or not os.path.isdir(artifact_dir):
        return []
    return sorted(
        os.path.join(artifact_dir, entry)
        for entry in os.listdir(artifact_dir)
        if entry.endswith(".coverage") and os.path.isfile(os.path.join(artifact_dir, entry))
    )


def combine_coverage_artifacts(
    artifact_dir: str,
    python_executable: str = sys.executable,
) -> Tuple[Optional[str], str]:
    if not artifact_dir:
        return None, "coverage artifact directory was not provided"

    coverage_files = [
        path
        for path in list_coverage_artifacts(artifact_dir)
        if os.path.basename(path) != ".coverage"
    ]
    if not coverage_files:
        return None, "no retained coverage artifacts were found"

    env = os.environ.copy()
    env["COVERAGE_FILE"] = os.path.join(artifact_dir, ".coverage")

    try:
        combine_result = subprocess.run(
            [python_executable, "-m", "coverage", "combine", "--keep", *coverage_files],
            capture_output=True,
            text=True,
            timeout=DEFAULT_REPORT_TIMEOUT,
            env=env,
            cwd=artifact_dir,
        )
    except subprocess.TimeoutExpired:
        return None, f"coverage combine timed out after {DEFAULT_REPORT_TIMEOUT} seconds"
    except Exception as error:
        return None, str(error)

    if combine_result.returncode != 0:
        message = combine_result.stderr.strip() if combine_result.stderr else "coverage combine failed"
        return None, message

    combined_data_file = os.path.join(artifact_dir, ".coverage")
    if not os.path.exists(combined_data_file):
        return None, "combined coverage data file was not created"

    return combined_data_file, ""


def generate_coverage_report_from_data_file(
    data_file: str,
    report_format: str = "json",
    output_path: Optional[str] = None,
    python_executable: str = sys.executable,
) -> Tuple[str, str]:
    if not data_file:
        return "", "coverage data file was not provided"

    env = os.environ.copy()
    env["COVERAGE_FILE"] = data_file

    command = [python_executable, "-m", "coverage", report_format]
    if report_format == "json":
        command.append("-i")
    if report_format in {"json", "xml", "lcov"} and output_path:
        command.extend(["-o", output_path])
    elif report_format == "html" and output_path:
        command.extend(["-d", output_path])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DEFAULT_REPORT_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "", f"coverage {report_format} timed out after {DEFAULT_REPORT_TIMEOUT} seconds"
    except Exception as error:
        return "", str(error)

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        return stdout, stderr or f"coverage {report_format} failed"

    return stdout, ""


def compact_coverage_summary(report_data: Dict[str, Any]) -> Dict[str, Any]:
    totals = report_data.get("totals", {}) if isinstance(report_data, dict) else {}
    percent_covered = float(totals.get("percent_covered", 0.0) or 0.0)
    covered_lines = int(totals.get("covered_lines", 0) or 0)
    num_statements = int(totals.get("num_statements", 0) or 0)
    missing_lines = int(totals.get("missing_lines", 0) or 0)
    return {
        "percent_covered": percent_covered,
        "percent_covered_display": f"{percent_covered:.2f}%",
        "covered_lines": covered_lines,
        "num_statements": num_statements,
        "missing_lines": missing_lines,
    }


def load_compact_coverage_summary(json_report_file: str) -> Dict[str, Any]:
    if not json_report_file or not os.path.exists(json_report_file):
        return {}
    try:
        with open(json_report_file, "r", encoding="utf-8") as file_handle:
            report_data = json.load(file_handle)
    except Exception:
        return {}
    return compact_coverage_summary(report_data)


def format_hourly_coverage_points(hourly_points: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for point in hourly_points:
        label = point.get("label") or f"Hour {point.get('hour', '?')}"
        percent = float(point.get("percent_covered", 0.0) or 0.0)
        covered = int(point.get("covered_lines", 0) or 0)
        total = int(point.get("num_statements", 0) or 0)
        lines.append(f"{label}: {percent:.2f}% coverage ({covered} covered lines / {total} executable lines)")
    return lines


_NO_COVERAGE_DATA_MARKERS = (
    "No data to report",
    "No data was collected",
    "no-data-collected",
)


def _is_no_coverage_data_error(message: str) -> bool:
    """True for coverage's benign "nothing measured" condition.

    A generated program that never imports the monitored library produces a
    coverage data file with no data for the configured source. Reporting on it
    exits non-zero with one of these messages. That is not a real failure — the
    bucket simply has no coverage yet — so callers should skip it, not abort.
    """
    if not message:
        return False
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in _NO_COVERAGE_DATA_MARKERS)


def _build_coverage_checkpoints(run_duration_seconds: float) -> List[Tuple[float, str]]:
    """Return (cutoff_seconds_from_start, label) for each checkpoint.

    Schedule:
      - Every 60 s for the first 10 minutes
      - Every 600 s from 10–60 minutes
      - Every 3600 s (hourly) thereafter
    """
    checkpoints: List[Tuple[float, str]] = []
    # 1-minute marks: 60, 120, …, 600
    for m in range(1, 11):
        t = m * 60.0
        if t > run_duration_seconds:
            break
        checkpoints.append((t, f"{m} min"))
    # 10-minute marks: 20, 30, 40, 50, 60 min (skip 10 min, already covered)
    for step in range(2, 7):
        t = step * 600.0
        if t > run_duration_seconds:
            break
        checkpoints.append((t, f"{step * 10} min"))
    # Hourly marks from hour 2 onwards (hour 1 = 60 min already covered)
    hour = 2
    while True:
        t = hour * 3600.0
        if t > run_duration_seconds:
            break
        checkpoints.append((t, f"Hour {hour}"))
        hour += 1
    return checkpoints


def build_hourly_coverage_timeline(
    artifact_dir: str,
    run_start_epoch: float,
    run_end_epoch: Optional[float] = None,
    python_executable: str = sys.executable,
) -> Tuple[List[Dict[str, Any]], str]:
    if not artifact_dir:
        return [], "coverage artifact directory was not provided"

    raw_files: List[Tuple[str, float]] = []
    for path in list_coverage_artifacts(artifact_dir):
        name = os.path.basename(path)
        if name in {".coverage", ".coverage.timeline"}:
            continue
        try:
            raw_files.append((path, os.path.getmtime(path)))
        except OSError:
            continue

    if not raw_files:
        return [], "no retained coverage artifacts were found"

    raw_files.sort(key=lambda item: item[1])
    if run_end_epoch is None:
        run_end_epoch = raw_files[-1][1]

    run_start = float(run_start_epoch)
    timeline_end = max(float(run_end_epoch), run_start)
    run_duration = timeline_end - run_start

    checkpoints = _build_coverage_checkpoints(run_duration)

    combined_data_file = os.path.join(artifact_dir, ".coverage.timeline")
    _safe_remove(combined_data_file)
    combine_env = os.environ.copy()
    combine_env["COVERAGE_FILE"] = combined_data_file

    points: List[Dict[str, Any]] = []
    last_compact: Dict[str, Any] = {}
    consumed = 0
    total_files = len(raw_files)

    for bucket_idx, (offset_seconds, label) in enumerate(checkpoints):
        cutoff = run_start + offset_seconds
        new_files: List[str] = []
        while consumed < total_files and raw_files[consumed][1] <= cutoff:
            new_files.append(raw_files[consumed][0])
            consumed += 1

        if new_files:
            cmd = [
                python_executable,
                "-m",
                "coverage",
                "combine",
                "--keep",
                "--append",
                "--data-file",
                combined_data_file,
                *new_files,
            ]
            try:
                combine_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_REPORT_TIMEOUT,
                    env=combine_env,
                    cwd=artifact_dir,
                )
            except subprocess.TimeoutExpired:
                return points, f"coverage timeline combine timed out after {DEFAULT_REPORT_TIMEOUT} seconds"
            except Exception as error:
                return points, str(error)

            if combine_result.returncode != 0:
                message = combine_result.stderr.strip() if combine_result.stderr else "coverage timeline combine failed"
                return points, message

            timeline_json = os.path.join(artifact_dir, f"coverage_timeline_{bucket_idx}.json")
            _, report_error = generate_coverage_report_from_data_file(
                combined_data_file,
                report_format="json",
                output_path=timeline_json,
                python_executable=python_executable,
            )
            if report_error and not _is_no_coverage_data_error(report_error):
                return points, report_error

            # When the bucket has measured data, refresh the running total. A
            # benign "no data yet" error leaves last_compact untouched so this
            # bucket is simply skipped instead of killing the whole timeline.
            if not report_error:
                last_compact = load_compact_coverage_summary(timeline_json)
            _safe_remove(timeline_json)

        if not last_compact:
            continue

        point = {
            "label": label,
            "elapsed_seconds": offset_seconds,
            **last_compact,
        }
        points.append(point)

    remaining_files = [path for path, _ in raw_files[consumed:]]
    if remaining_files:
        cmd = [
            python_executable,
            "-m",
            "coverage",
            "combine",
            "--keep",
            "--append",
            "--data-file",
            combined_data_file,
            *remaining_files,
        ]
        try:
            combine_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DEFAULT_REPORT_TIMEOUT,
                env=combine_env,
                cwd=artifact_dir,
            )
        except subprocess.TimeoutExpired:
            return points, f"coverage timeline combine timed out after {DEFAULT_REPORT_TIMEOUT} seconds"
        except Exception as error:
            return points, str(error)

        if combine_result.returncode != 0:
            message = combine_result.stderr.strip() if combine_result.stderr else "coverage timeline combine failed"
            return points, message

    if not os.path.exists(combined_data_file):
        return [], "failed to build hourly coverage timeline"

    final_json = os.path.join(artifact_dir, "coverage_timeline_final.json")
    _, final_report_error = generate_coverage_report_from_data_file(
        combined_data_file,
        report_format="json",
        output_path=final_json,
        python_executable=python_executable,
    )
    if final_report_error and not _is_no_coverage_data_error(final_report_error):
        return points, final_report_error

    final_compact = {} if final_report_error else load_compact_coverage_summary(final_json)
    _safe_remove(final_json)

    # Cap the timeline with the cumulative total. Skip it only when nothing was
    # ever measured (no hourly points and no final data) so we don't emit a
    # misleading zero-coverage end point.
    if final_compact or points:
        points.append(
            {
                "label": "Run End",
                "elapsed_seconds": max(0.0, float(run_end_epoch) - float(run_start_epoch)),
                **final_compact,
            }
        )

    if not points:
        return [], "no coverage data was collected from any retained artifact"

    return points, ""


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _extract_run_error(result: subprocess.CompletedProcess) -> Tuple[str, str]:
    stderr = result.stderr or ""
    
    # Filter out coverage warnings and their source code printouts
    clean_lines = []
    lines = stderr.splitlines()
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            if line.startswith("  ") and ("self.warn" in line or "self._warn" in line):
                continue
        if "CoverageWarning" in line:
            if ":" in line:
                skip_next = True
            continue
        clean_lines.append(line)
    stderr = "\n".join(clean_lines)
    full_error = stderr.strip()

    if result.returncode != 0:
        fallback = f"Process exited with code {result.returncode}"
        if not full_error:
            full_error = fallback
        return summarize_error_text(full_error, fallback), full_error

    if full_error:
        return summarize_error_text(full_error, ""), full_error

    return "", ""


def _load_coverage_percent_from_json(json_report_file: str) -> Tuple[float, Dict[str, Any]]:
    with open(json_report_file, "r", encoding="utf-8") as file_handle:
        report_data = json.load(file_handle)
    coverage_percent = report_data.get("totals", {}).get("percent_covered", 0.0)
    return coverage_percent, report_data


def _run_coverage_json_report(
    python_executable: str,
    json_report_file: str,
    env: Dict[str, str],
) -> Tuple[Optional[float], Optional[Dict[str, Any]], str]:
    try:
        report_result = subprocess.run(
            [python_executable, "-m", "coverage", "json", "-o", json_report_file],
            capture_output=True,
            text=True,
            timeout=DEFAULT_REPORT_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, None, f"coverage json timed out after {DEFAULT_REPORT_TIMEOUT} seconds"
    except Exception as error:
        return None, None, str(error)

    if report_result.returncode != 0:
        message = report_result.stderr.strip() if report_result.stderr else "coverage json failed"
        return None, None, message

    if not os.path.exists(json_report_file):
        return None, None, "coverage json report was not created"

    try:
        coverage_percent, report_data = _load_coverage_percent_from_json(json_report_file)
        return coverage_percent, report_data, ""
    except Exception as error:
        return None, None, f"failed to parse coverage json: {error}"


def _compute_quality_score(metrics: Dict[str, Any]) -> float:
    cov = metrics.get("coverage_percent", 0.0)
    w_time = metrics.get("wall_time", 0.0)
    func_count = metrics.get("function_count", 0)
    line_count = metrics.get("line_count", 0)
    nesting_depth = metrics.get("nesting_depth", 0)
    return (
        cov
        + (w_time * 1.0)
        + (func_count * 1.0)
        + (line_count * 0.2)
        + (nesting_depth * 1.0)
    )


def _wrap_code_for_compilation(clean_code: str, language: str) -> str:
    if language == "guppy":
        return wrap_for_compilation_guppy(clean_code)
    if language == "qiskit":
        return wrap_for_compilation_qiskit(clean_code)
    if language == "cirq":
        return wrap_for_compilation_cirq(clean_code)
    if language == "pytket":
        return wrap_for_compilation_pytket(clean_code)
    if language == "pennylane":
        return wrap_for_compilation_pennylane(clean_code)
    return clean_code


def _wrap_code_for_testing(clean_code: str, language: str, circuit_id: int) -> str:
    if language == "guppy":
        return wrap_for_testing_guppy(clean_code, circuit_id)
    if language == "qiskit":
        return wrap_for_testing_qiskit(clean_code, circuit_id)
    if language == "cirq":
        return wrap_for_testing_cirq(clean_code, circuit_id)
    if language == "pytket":
        return wrap_for_testing_pytket(clean_code, circuit_id)
    if language == "pennylane":
        return wrap_for_testing_pennylane(clean_code, circuit_id)
    return clean_code


def _temp_file_prefix(source_file_path: str = None) -> str:
    if not source_file_path:
        return "quillfuzz_"

    basename = os.path.basename(source_file_path)
    stem, _ = os.path.splitext(basename)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    if not cleaned:
        cleaned = "source"
    return f"quillfuzz_{cleaned}_"


def _execute_python_code(
    program_code: str,
    timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    language: str = "guppy",
    coverage_source: str = None,
    source_file_path: str = None,
    ks_low_threshold: Optional[float] = None,
    coverage_artifact_dir: Optional[str] = None,
):
    """
    Internal helper to execute prepared Python code with coverage and metrics tracking.
    """
    if coverage_source is None:
        coverage_source = _default_coverage_source(language)

    temp_file_path = None
    metrics_file = None
    coverage_file = None
    json_report_file = None
    retained_coverage_file = None
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix=_temp_file_prefix(source_file_path), delete=False, dir='/tmp') as temp_file:
            temp_file.write(program_code)
            temp_file_path = temp_file.name
        
        metrics_file = temp_file_path + ".time"
        coverage_file = temp_file_path + ".coverage"
        json_report_file = temp_file_path + ".json"
        artifact_dir = _coverage_artifact_dir(coverage_artifact_dir)
        
        try:
            start_time = time.time()
            
            env = _build_execution_env(coverage_file, source_file_path, ks_low_threshold)
            
            # Execute
            cmd = [
                "/usr/bin/time", "-v", "-o", metrics_file, 
                sys.executable, "-m", "coverage", "run",
                f"--source={coverage_source}",
                temp_file_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

            metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as file_handle:
                    metrics = parse_time_metrics(file_handle.read())
            
            # Calculate static analysis metrics
            complexity = get_code_complexity_metrics(program_code)
            metrics["nesting_depth"] = complexity["nesting_depth"]
            metrics["function_count"] = complexity["function_count"]
            metrics["line_count"] = len(strip_markdown_syntax(program_code).splitlines())
            
            wall_time = time.time() - start_time
            metrics["wall_time"] = wall_time
            
            # Process coverage
            if os.path.exists(coverage_file):
                cov_percent, _, cov_error = _run_coverage_json_report(
                    sys.executable,
                    json_report_file,
                    env,
                )
                if cov_percent is not None:
                    metrics["coverage_percent"] = cov_percent
                elif cov_error:
                    metrics["coverage_error"] = cov_error

                retained_coverage_file = _persist_coverage_artifact(coverage_file, artifact_dir)
                if retained_coverage_file:
                    metrics["coverage_artifact_path"] = retained_coverage_file
                    metrics["coverage_artifact_dir"] = artifact_dir
                elif artifact_dir:
                    metrics["coverage_artifact_dir"] = artifact_dir
                    metrics["coverage_artifact_error"] = "Failed to persist coverage artifact"
            
            # Calculate combined quality score
            # Heuristic: maximize coverage, wall time, and static complexity signals
            metrics["quality_score"] = _compute_quality_score(metrics)

            # Return error (if any)
            error, error_full = _extract_run_error(result)
            if error_full:
                metrics["error_full"] = error_full
            if error:
                metrics["error_summary"] = error
            
            if not error and result.stdout and ("Panic" in result.stdout or "Error" in result.stdout):
                error = "Error detected in stdout"
                metrics["error_full"] = result.stdout
                metrics["error_summary"] = error
            return error, result.stdout, metrics
            
        finally:
            _safe_remove(temp_file_path)
            _safe_remove(metrics_file)
            if retained_coverage_file is None:
                _safe_remove(coverage_file)
            _safe_remove(json_report_file)
                
    except subprocess.TimeoutExpired:
        timeout_error = f"ERROR: Program execution timed out after {timeout} seconds"
        return timeout_error, "", {"wall_time": float(timeout), "note": "timed_out", "error_full": timeout_error, "error_summary": timeout_error}
    except Exception as e:
        full_error = f"ERROR: Failed to execute program: {str(e)}"
        summary = summarize_error_text(full_error, "Execution failed")
        return summary, "", {"error_full": full_error, "error_summary": summary}

def compile_generated_program(program_code: str, timeout: int = DEFAULT_COMPILE_TIMEOUT, language: str = 'guppy', coverage_source: str = None, source_file_path: str = None):
    """
    Compiles (or checks syntax/imports) of generated Python program.
    Does NOT run full tests, just verifies valid compilation/construction.
    
    Returns:
        tuple: (Error message, stdout, Metrics, Wrapped code)
    """
    clean_code = strip_markdown_syntax(program_code)
    wrapped_code = _wrap_code_for_compilation(clean_code, language)
    
    # Here, the wall_time metric will reflect compilation time only.
    error, stdout, metrics = _execute_python_code(wrapped_code, timeout, language, coverage_source, source_file_path)
    return error, stdout, metrics, wrapped_code

def run_generated_program(
    program_code: str,
    timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    language: str = 'guppy',
    coverage_source: str = None,
    source_file_path: str = None,
    circuit_id: int = 0,
    ks_low_threshold: Optional[float] = None,
    coverage_artifact_dir: Optional[str] = None,
):
    """
    Execute generated Python program with full test harness (KS diff test).
    
    Returns:
        tuple: (Error message, stdout, Metrics, Wrapped code)
    """
    clean_code = strip_markdown_syntax(program_code)
    wrapped_code = _wrap_code_for_testing(clean_code, language, circuit_id)
    
    # Here, the wall_time metric will reflect full execution time, including execution and compilation.
    error, stdout, metrics = _execute_python_code(
        wrapped_code,
        timeout,
        language,
        coverage_source,
        source_file_path,
        ks_low_threshold,
        coverage_artifact_dir,
    )
    return error, stdout, metrics, wrapped_code


def run_coverage_on_file(file_path: str, source_package: str = None, verbose: bool = False, timeout: int = DEFAULT_EXECUTION_TIMEOUT, python_executable=sys.executable, language: str = 'guppy'):
    """
    Run a single python file with coverage tracking.
    Returns the coverage percentage, any error message, coverage data, and verbose report.
    Automatically adds the main wrapper for execution.
    """
    if source_package is None:
        source_package = _default_coverage_source(language)
        
    temp_src_path = None
    coverage_file_path = None
    json_report_file = None

    try:
        # Read the code
        try:
            with open(file_path, "r", encoding="utf-8") as file_handle:
                code = file_handle.read()

            wrapped_code = _wrap_code_for_compilation(code, language)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix=_temp_file_prefix(file_path), delete=False, dir="/tmp") as temp_file:
                temp_file.write(wrapped_code)
                temp_src_path = temp_file.name
        except Exception as e:
            return 0.0, f"Error preparing file: {str(e)}", {}, ""

        with tempfile.NamedTemporaryFile(suffix=".coverage", delete=False) as cov_file:
            coverage_file_path = cov_file.name
        
        json_report_file = coverage_file_path + ".json"
        
        env = _build_execution_env(coverage_file_path)

        # Execute with coverage using the temporary wrapped file
        cmd = [python_executable, "-m", "coverage", "run", "--branch", f"--source={source_package}", temp_src_path]
        
        # We use a timeout to prevent hanging scripts
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            run_error, run_error_full = _extract_run_error(result)
                
        except subprocess.TimeoutExpired:
            return 0.0, "Timeout", {}, ""

        coverage_percent = 0.0
        coverage_data = {}
        verbose_report = ""

        # Generate JSON report
        if os.path.exists(coverage_file_path):
            cov_percent, cov_data, cov_error = _run_coverage_json_report(
                python_executable,
                json_report_file,
                env,
            )

            if cov_percent is not None:
                coverage_percent = cov_percent
            if cov_data is not None:
                coverage_data = cov_data
            if cov_error:
                run_error = f"{run_error}\nCoverage report error: {cov_error}".strip()
                run_error_full = f"{run_error_full}\nCoverage report error: {cov_error}".strip()

            if verbose:
                try:
                    report_res = subprocess.run(
                        [python_executable, "-m", "coverage", "report"],
                        capture_output=True,
                        text=True,
                        timeout=DEFAULT_REPORT_TIMEOUT,
                        env=env,
                    )
                    verbose_report = report_res.stdout
                    if report_res.returncode != 0 and report_res.stderr:
                        run_error = f"{run_error}\nCoverage report output error: {report_res.stderr.strip()}".strip()
                        run_error_full = f"{run_error_full}\nCoverage report output error: {report_res.stderr.strip()}".strip()
                except Exception as e:
                    run_error = f"{run_error}\nCoverage report output error: {str(e)}".strip()
                    run_error_full = f"{run_error_full}\nCoverage report output error: {str(e)}".strip()

        if run_error and run_error_full:
            return coverage_percent, summarize_error_text(run_error, "Coverage execution failed"), coverage_data, verbose_report

        return coverage_percent, run_error, coverage_data, verbose_report

    except Exception as e:
        return 0.0, str(e), {}, ""
    finally:
        # Cleanup
        _safe_remove(temp_src_path)
        _safe_remove(coverage_file_path)
        _safe_remove(json_report_file)
