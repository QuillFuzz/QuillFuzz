import os
import sys
import subprocess
import tempfile
import time
import json
from typing import Any, Dict, Optional, Tuple
from .utils import strip_markdown_syntax, parse_time_metrics
from .ast_ops import (
    wrap_for_compilation_guppy,
    wrap_for_testing_guppy,
    wrap_for_compilation_qiskit,
    wrap_for_testing_qiskit,
    get_code_complexity_metrics,
)

# Default timeouts in seconds
DEFAULT_EXECUTION_TIMEOUT = 300
DEFAULT_COMPILE_TIMEOUT = 60
DEFAULT_REPORT_TIMEOUT = 60

LANGUAGE_DEFAULT_SOURCES = {
    "guppy": "guppylang_internals",
    "qiskit": "qiskit",
}


def _default_coverage_source(language: str) -> str:
    return LANGUAGE_DEFAULT_SOURCES.get(language, "qiskit")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_execution_env(coverage_file: str, source_file_path: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["COVERAGE_FILE"] = coverage_file

    if source_file_path:
        env["QUILLFUZZ_SOURCE_FILE"] = os.path.abspath(source_file_path)

    project_root = _project_root()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else project_root
    )
    return env


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _summarize_error_text(error_text: str, fallback: str) -> str:
    if not error_text:
        return fallback

    lines = [line.strip() for line in error_text.splitlines() if line and line.strip()]
    if not lines:
        return fallback

    return lines[-1]


def _extract_run_error(result: subprocess.CompletedProcess) -> Tuple[str, str]:
    stderr = result.stderr or ""
    full_error = stderr.strip()

    if result.returncode != 0:
        fallback = f"Process exited with code {result.returncode}"
        if not full_error:
            full_error = fallback
        return _summarize_error_text(full_error, fallback), full_error

    if full_error:
        return _summarize_error_text(full_error, ""), full_error

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
        + (w_time * 2.0)
        + (func_count * 1.0)
        + (line_count * 0.1)
        + (nesting_depth * 2.0)
    )


def _wrap_code_for_compilation(clean_code: str, language: str) -> str:
    if language == "guppy":
        return wrap_for_compilation_guppy(clean_code)
    if language == "qiskit":
        return wrap_for_compilation_qiskit(clean_code)
    return clean_code


def _wrap_code_for_testing(clean_code: str, language: str, circuit_id: int) -> str:
    if language == "guppy":
        return wrap_for_testing_guppy(clean_code, circuit_id)
    if language == "qiskit":
        return wrap_for_testing_qiskit(clean_code, circuit_id)
    return clean_code


def _execute_python_code(
    program_code: str,
    timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    language: str = "guppy",
    coverage_source: str = None,
    source_file_path: str = None,
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
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as temp_file:
            temp_file.write(program_code)
            temp_file_path = temp_file.name
        
        metrics_file = temp_file_path + ".time"
        coverage_file = temp_file_path + ".coverage"
        json_report_file = temp_file_path + ".json"
        
        try:
            start_time = time.time()
            
            env = _build_execution_env(coverage_file, source_file_path)
            
            # Execute
            cmd = [
                "/usr/bin/time", "-v", "-o", metrics_file, 
                sys.executable, "-m", "coverage", "run", 
                "--branch", f"--source={coverage_source}", 
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
            _safe_remove(coverage_file)
            _safe_remove(json_report_file)
                
    except subprocess.TimeoutExpired:
        timeout_error = f"ERROR: Program execution timed out after {timeout} seconds"
        return timeout_error, "", {"wall_time": float(timeout), "note": "timed_out", "error_full": timeout_error, "error_summary": timeout_error}
    except Exception as e:
        full_error = f"ERROR: Failed to execute program: {str(e)}"
        summary = _summarize_error_text(full_error, "Execution failed")
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

def run_generated_program(program_code: str, timeout: int = DEFAULT_EXECUTION_TIMEOUT, language: str = 'guppy', coverage_source: str = None, source_file_path: str = None, circuit_id: int = 0):
    """
    Execute generated Python program with full test harness (KS diff test).
    
    Returns:
        tuple: (Error message, stdout, Metrics, Wrapped code)
    """
    clean_code = strip_markdown_syntax(program_code)
    wrapped_code = _wrap_code_for_testing(clean_code, language, circuit_id)
    
    # Here, the wall_time metric will reflect full execution time, including execution and compilation.
    error, stdout, metrics = _execute_python_code(wrapped_code, timeout, language, coverage_source, source_file_path)
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

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as temp_file:
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
            return coverage_percent, _summarize_error_text(run_error, "Coverage execution failed"), coverage_data, verbose_report

        return coverage_percent, run_error, coverage_data, verbose_report

    except Exception as e:
        return 0.0, str(e), {}, ""
    finally:
        # Cleanup
        _safe_remove(temp_src_path)
        _safe_remove(coverage_file_path)
        _safe_remove(json_report_file)
