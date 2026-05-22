import os
import sys
import time
import tempfile
import concurrent.futures
import random
import argparse
import json
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

# Add project root to path so we can import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from local library
from utils.circuit_assembler import assemble
from utils.llm_client import ask_any_model, get_dynamic_prompt, improve_prompt_logic
from utils.utils import save_text_to_file, generate_summary_plot, generate_complexity_scatter_plots, sanitize_model_name
from utils.execution import run_generated_program, compile_generated_program
from utils.reporting import (
    Logger,
    build_phase_summary,
    format_low_ks_values,
    append_rows_to_csv,
    build_error_details,
    summarize_errors,
    build_metrics_row,
    populate_ks_test_metrics,
)

SUPPORTED_LANGUAGES = ("guppy", "qiskit", "pytket", "pennylane")
DEFAULT_LANGUAGE = "guppy"

@dataclass
class GenerationStats:
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    quality_score: Optional[float] = None
    execution_quality_score: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=lambda: {'compilation': {}, 'execution': {}})

    def update(self, stats: Dict[str, Any]):
        if not stats:
            return
        self.cost += stats.get('cost', 0.0)
        self.prompt_tokens += stats.get('prompt_tokens', 0)
        self.completion_tokens += stats.get('completion_tokens', 0)
        self.total_tokens += stats.get('total_tokens', 0)
        if 'quality_score' in stats:
            self.quality_score = stats['quality_score']


class ProgramProcessor:
    def __init__(self, index, model, config, logger, start_time, stage="generation"):
        self.index = index
        self.model = model
        self.config = config
        self.logger = logger
        self.start_time = start_time
        self.stage = stage
        self.filename = f"{sanitize_model_name(self.model)}_{self.stage}output{self.index+1}.py"
        self.stats = GenerationStats()
        self.encountered_errors = []

    def generate(self, prompt_filename, prompt_kwargs=None):
        
        # Prepare prompts for asking model
        prompt_kwargs = prompt_kwargs or {}
        prompt_path = prompt_filename if os.path.isabs(prompt_filename) else os.path.join(self.config.prompt_dir, prompt_filename)

        if not os.path.exists(prompt_path):
            self.logger.log(f"{self.stage.capitalize()} prompt not found at {prompt_path}")
            return None
        
        prompt = get_dynamic_prompt(prompt_path, **prompt_kwargs)

        if prompt is None:
            self.logger.log(f"Error: Prompt for {self.filename} is empty.\n")
            return None

        # Sends prompt to model and gets back generated code, stats on cost and tokens used as well as any errors
        code, stats, err = ask_any_model(self.model, prompt, reasoning_effort=self.config.reasoning_effort)
        
        if code is None:
            self.logger.log(f"Failed to generate {self.filename}. Error: {err}\n")
            self.encountered_errors.append(f"Generation API Error: {err}")
            return None

        self.stats.update(stats)
        self.logger.log(f"{self.filename} Generation Cost: ${stats.get('cost', 0.0):.6f} | "
                         f"Tokens (In/Out/Total): {stats.get('prompt_tokens', 0)}/{stats.get('completion_tokens', 0)}/{stats.get('total_tokens', 0)}")
        return code

    def compile_check(self, code):
        
        # Safely log metrics and errors from no-optimisation compile check for syntax errors etc.
        self.logger.log(f"--- {f'[Elapsed: {time.time() - self.start_time:.2f}s]'} Testing {self.stage} {self.filename} ---\n")
        error, _, metrics, wrapped_code = compile_generated_program(code, language=self.config.language)
        metrics = metrics or {}
        self.stats.metrics['compilation'] = metrics
        
        # Store compiliation quality score as the main quality_score
        if error:
            self.stats.quality_score = 0.0
            full_error = metrics.get("error_full") or error
            self.logger.log(f"{self.filename} Compilation Failed:\n{full_error}\n")
            self.encountered_errors.append(f"Compilation Error:\n{error}")
            if self.config.verbose:
                self.logger.log(f"--- {self.filename} Code ---\n{wrapped_code}\n-----------------------\n")
            return False, error
        
        self.stats.quality_score = metrics.get('quality_score', 0.0)
        self.logger.log(f"{self.filename} compiled successfully.\n")
        return True, ""

    def run_check(self, code):

        # Running requires wrapping code in test harness and executing it in a tempfile
        self.logger.log(f"--- {f'[Elapsed: {time.time() - self.start_time:.2f}s]'} Running {self.filename} ---\n")
        source_file_path = None
        temp_source_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix=f"{self.filename.replace('.py', '')}_",
                delete=False,
                dir="/tmp",
            ) as temp_source_file:
                temp_source_file.write(code)
                temp_source_path = temp_source_file.name
            source_file_path = temp_source_path
        except Exception as e:
            self.logger.log(f"Warning: failed to prepare temp source for runtime check: {e}")

        error, output, metrics, runtime_code = run_generated_program(
            code,
            language=self.config.language,
            source_file_path=source_file_path,
            circuit_id=self.index + 1,
        )
        metrics = metrics or {}
        self.stats.metrics['execution'] = metrics

        low_ks_values = populate_ks_test_metrics(metrics, output, self.config.ks_low_threshold)
        if low_ks_values:
            self.logger.log(
                f"{self.filename} LOW KS detected (threshold={self.config.ks_low_threshold}): {format_low_ks_values(low_ks_values)}"
            )
        
        # Store execution quality score separately
        self.stats.execution_quality_score = metrics.get('quality_score', 0.0)

        if error.strip():
            # Log test harness stdout if verbose
            if self.config.verbose:
                self.logger.log(f"Test Harness Output:\n{output}\n")
                
            full_error = metrics.get("error_full") or error
            self.logger.log(f"{self.filename} Runtime Error:\n{full_error}\n")
            self.encountered_errors.append(f"Runtime Error:\n{error}")
            if self.config.verbose:
                self.logger.log(f"--- {self.filename} Code ---\n{runtime_code}\n-----------------------\n")
            if temp_source_path and os.path.exists(temp_source_path):
                try:
                    os.remove(temp_source_path)
                except Exception:
                    pass
            return False, error
        
        self.logger.log(f"{self.filename} ran successfully.\n")
        if output:
            self.logger.log(f"--- {self.filename} Output ---\n{output}\n--------------------------\n")
        if temp_source_path and self.config.current_generated_dir:
            generated_source_path = os.path.join(self.config.current_generated_dir, self.filename)
            try:
                save_text_to_file(code, generated_source_path, verbose=False)
            except Exception as e:
                self.logger.log(f"Warning: failed to persist successful runtime source ({generated_source_path}): {e}")
        if temp_source_path and os.path.exists(temp_source_path):
            try:
                os.remove(temp_source_path)
            except Exception:
                pass
        return True, ""

    def fix_loop(self, current_code, current_error, fix_cycles=None):
        
        last_failure_type = "compile_fail"
        total_cycles = self.config.n_fixing_cycles if fix_cycles is None else fix_cycles
        prompt_file = "mutate_prompt_template.txt" if self.stage == "mutation" else "fixing_prompt_template.txt"
        
        for cycle in range(total_cycles):
            prompt_path = os.path.join(self.config.prompt_dir, prompt_file)
            if not os.path.exists(prompt_path):
                self.logger.log(f"{self.stage.capitalize()} fix prompt missing at {prompt_path}\n")
                return current_code, last_failure_type

            prompt = get_dynamic_prompt(
                prompt_path,
                faulty_code=current_code,
                error_message=current_error,
                input_code=current_code,
            )
            
            if not prompt:
                self.logger.log(f"Error: Generated prompt for {self.filename} (cycle {cycle+1}) is empty.\n")
                break

            fixed_code, stats, err = ask_any_model(self.model, prompt, reasoning_effort=self.config.reasoning_effort)

            if err:
                 self.logger.log(f"Error from LLM during fixing cycle {cycle+1} for {self.filename}: {err}")

            if not fixed_code:
                self.logger.log(f"Fixing cycle {cycle+1} failed for {self.filename}: {err}\n")
                break

            current_code = fixed_code
            
            self.stats.update(stats)
            self.logger.log(f"{self.filename} Fixing (Cycle {cycle+1}) Cost: ${stats.get('cost', 0.0):.6f} | "
                            f"Tokens (In/Out/Total): {stats.get('prompt_tokens', 0)}/{stats.get('completion_tokens', 0)}/{stats.get('total_tokens', 0)}")

            # Verify fix (Compile only first)
            compile_ok, compile_err = self.compile_check(fixed_code)
            
            if not compile_ok:
                self.logger.log(f"Fixed {self.filename} (Cycle {cycle+1}) Compilation Failed.\n")
                current_error = compile_err
                last_failure_type = "compile_fail"
                continue

            self.logger.log(f"Fixed {self.filename} (Cycle {cycle+1}) compiled successfully.\n")
            
            # If we need to run it
            if not self.config.compile_only:
                run_ok, run_err = self.run_check(fixed_code)
                if run_ok:
                    return fixed_code, "success"
                else:
                    current_error = run_err
                    last_failure_type = "runtime_fail"
                    continue
            else:
                return fixed_code, "success"

        return current_code, last_failure_type

    def _setup_failure_dirs(self, failed_dir):
        """Create and return compile and runtime failure directories."""
        compile_fail_dir = os.path.join(failed_dir, "compile_fail")
        runtime_fail_dir = os.path.join(failed_dir, "runtime_fail")
        os.makedirs(compile_fail_dir, exist_ok=True)
        os.makedirs(runtime_fail_dir, exist_ok=True)
        return compile_fail_dir, runtime_fail_dir

    def process(self, generated_dir, failed_dir, prompt_filename="generation_prompt.txt", compile_only=False, prompt_kwargs=None, fix_cycles=None):
        
        self.config.compile_only = compile_only # augment config temporarily
        self.config.current_generated_dir = generated_dir

        compile_fail_dir, runtime_fail_dir = self._setup_failure_dirs(failed_dir)
        
        code = self.generate(prompt_filename, prompt_kwargs=prompt_kwargs)
        if not code:
            return None, self.stats, self.encountered_errors, False

        compile_ok, compile_err = self.compile_check(code)
        
        if compile_ok:
            if not compile_only:
                run_ok, _ = self.run_check(code)
                if not run_ok:
                    save_path = os.path.join(runtime_fail_dir, self.filename)
                    generated_path = os.path.join(generated_dir, self.filename)
                    if os.path.exists(generated_path):
                        os.remove(generated_path)
                    save_text_to_file(code, save_path)
                    return None, self.stats, list(set(self.encountered_errors)), False

            save_path = os.path.join(generated_dir, self.filename)
            save_text_to_file(code, save_path)
            return save_path, self.stats, list(set(self.encountered_errors)), False

        # If compilation failed, try fixing
        fixed_code, final_status = self.fix_loop(code, compile_err, fix_cycles=fix_cycles)
        
        if fixed_code and final_status == "success":
            save_path = os.path.join(generated_dir, self.filename)
            save_text_to_file(fixed_code, save_path)
            return save_path, self.stats, list(set(self.encountered_errors)), True
        else:
            failure_dir = runtime_fail_dir if final_status == "runtime_fail" else compile_fail_dir
            save_path = os.path.join(failure_dir, self.filename)
            generated_path = os.path.join(generated_dir, self.filename)
            if final_status == "runtime_fail" and os.path.exists(generated_path):
                os.remove(generated_path)
            save_text_to_file(fixed_code if fixed_code else code, save_path)
            return None, self.stats, list(set(self.encountered_errors)), True


def run_training_phase(model, args, common_run_dir, main_logfile_path):
    """Run the training phase to improve generation prompts."""
    prompt_filename = "generation_prompt.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    training_logger = Logger(main_logfile_path)
    
    if args.training_n <= 0:
        return prompt_filename

    max_rounds = args.max_rounds

    best_prompt = prompt_filename
    best_fix_ratio = 1.0
    total_start_time = time.time()
    training_stats = []
    training_reports = []
    rounds_completed = 0

    for round_idx in range(max_rounds + 1):
        # Setup directories
        model_name = sanitize_model_name(model)
        round_dir = os.path.join(common_run_dir, "training_phase", model_name, f"round_{round_idx}")
        t_gen_dir = os.path.join(round_dir, "generated")
        t_fail_dir = os.path.join(round_dir, "failed")
        os.makedirs(t_gen_dir, exist_ok=True)
        os.makedirs(t_fail_dir, exist_ok=True)

        training_logger.log(f"\n[Training Round {round_idx}] Model: {model} | Prompt: {prompt_filename}")
        training_logger.log(f"Reasoning Effort: {args.reasoning_effort}")

        training_errors = []
        count_needed_fix = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for i in range(args.training_n):
                processor = ProgramProcessor(i, model, args, training_logger, time.time())
                futures.append(executor.submit(
                    processor.process, t_gen_dir, t_fail_dir, prompt_filename, True
                ))

            for future in tqdm(concurrent.futures.as_completed(futures), total=args.training_n, desc=f"Training Round {round_idx}"):
                try:
                    save_path, stats, errors, was_fixed = future.result()
                    training_errors.extend(errors)
                    if stats:
                        training_stats.append(stats)
                    training_reports.append({
                        "file": os.path.basename(save_path) if save_path else f"round_{round_idx}_program",
                        "success": not was_fixed,
                        "coverage_percent": 0.0,
                        "low_ks_test_levels": [],
                        "error": summarize_errors(errors),
                        "error_full": "\n\n---\n\n".join(errors),
                    })
                    if was_fixed:
                        count_needed_fix += 1
                except Exception as e:
                    training_logger.log(f"Training error: {e}")

        fix_ratio = count_needed_fix / args.training_n
        rounds_completed += 1
        training_logger.log(f"Round {round_idx} Result: {count_needed_fix}/{args.training_n} fixes (Ratio: {fix_ratio:.2f})")

        # Update best prompt if this round is strictly better
        if fix_ratio < best_fix_ratio:
            best_fix_ratio = fix_ratio
            best_prompt = prompt_filename

        if fix_ratio < args.training_threshold:
            training_logger.log(f"Training success! Proceeding with {prompt_filename}")
            summary_log, _ = build_phase_summary(
                "TRAINING",
                model,
                rounds_completed * args.training_n,
                time.time() - total_start_time,
                training_stats,
                training_reports,
                args.ks_low_threshold,
            )
            training_logger.log(summary_log)
            return prompt_filename

        if args.improve_prompt and round_idx < max_rounds:
            training_logger.log("Threshold not met. Improving prompt...")
            
            # Resolve current prompt path for reading
            current_prompt_path = prompt_filename
            if not os.path.isabs(current_prompt_path):
                current_prompt_path = os.path.join(args.prompt_dir, current_prompt_path)
            
            # Define new prompt path
            new_prompt_path = os.path.join(round_dir, "improved_prompt.txt")

            try:
                # Returns absolute path to the new prompt
                prompt_filename = improve_prompt_logic(
                    args.improver_model, 
                    current_prompt_path,
                    os.path.join(project_root, "prompts", "common"),
                    new_prompt_path, 
                    training_errors, 
                    args.language, 
                    training_logger,
                    reasoning_effort=args.reasoning_effort
                )
            except Exception as e:
                training_logger.log(f"Error improving prompt: {e}")
                import traceback
                training_logger.log(traceback.format_exc())
        else:
            training_logger.log("Max rounds reached or improvement disabled.")

    summary_log, _ = build_phase_summary(
        "TRAINING",
        model,
        rounds_completed * args.training_n,
        time.time() - total_start_time,
        training_stats,
        training_reports,
        args.ks_low_threshold,
    )
    training_logger.log(summary_log)
    return best_prompt

def run_production_phase(model, prompt_filename, args, common_run_dir, logfile_path, logger=None):
    if logger is None:
        logger = Logger(logfile_path)
    start_time = time.time()
    gen_dir = os.path.join(common_run_dir, "generated")
    fail_dir = os.path.join(common_run_dir, "failed_programs")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    logger.log(f"\n{'='*60}\n PRODUCTION PHASE: {args.n_programs} programs\n{'='*60}")
    logger.log(f"Prompt: {prompt_filename}")
    logger.log(f"Reasoning Effort: {args.reasoning_effort}")

    if os.path.isabs(prompt_filename):
        prompt_path = prompt_filename
    else:
        prompt_path = os.path.join(args.prompt_dir, prompt_filename)
    
    try:
        with open(prompt_path, 'r') as f:
            prompt_content = f.read()
        logger.log(f"\n--- Prompt Content ---\n{prompt_content}\n----------------------\n")
    except Exception as e:
        logger.log(f"Could not read prompt file for verbose logging: {e}")

    stats_list = []
    metrics_rows = []
    report_entries = []
    successful_files = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i in range(args.n_programs):
            processor = ProgramProcessor(i, model, args, logger, time.time())
            future = executor.submit(
                processor.process, gen_dir, fail_dir, prompt_filename, False
            )
            futures[future] = processor.filename

        for future in tqdm(concurrent.futures.as_completed(futures), total=args.n_programs, desc=f"Production {model}"):
            try:
                save_path, stats, errors, _ = future.result()
                source_filename = futures[future]
                if stats:
                    stats_list.append(stats)

                execution_metrics = stats.metrics.get("execution", {}) if stats else {}
                compilation_metrics = stats.metrics.get("compilation", {}) if stats else {}
                low_ks_values = execution_metrics.get("low_ks_test_levels", []) if execution_metrics else []
                file_name = os.path.basename(save_path) if save_path else source_filename
                success = bool(save_path)
                
                error_details = build_error_details(
                    [{
                        "error": execution_metrics.get("error_summary")
                            or compilation_metrics.get("error_summary")
                            or summarize_errors(errors),
                        "error_full": execution_metrics.get("error_full")
                            or compilation_metrics.get("error_full")
                            or "\n\n---\n\n".join(errors),
                    }]
                )
                
                report_entries.append({
                    "file": file_name,
                    "success": success,
                    "coverage_percent": execution_metrics.get("coverage_percent", 0.0) if execution_metrics else 0.0,
                    "low_ks_test_levels": low_ks_values,
                    "error": error_details["error"],
                    "error_full": error_details["error_full"],
                })

                if success and save_path:
                    successful_files.append(os.path.abspath(save_path))

                metrics_rows.append(build_metrics_row(
                    model, file_name, success, execution_metrics, compilation_metrics
                ))
            except Exception as e:
                logger.log(f"Error: {e}")
                source_filename = futures[future]
                error_details = build_error_details([{"error": str(e), "error_full": str(e)}])
                
                report_entries.append({
                    "file": source_filename,
                    "success": False,
                    "coverage_percent": 0.0,
                    "low_ks_test_levels": [],
                    "error": error_details["error"],
                    "error_full": error_details["error_full"],
                })
                metrics_rows.append(build_metrics_row(model, source_filename, False, {}, {}))

    metrics_csv_path = os.path.join(os.path.dirname(logfile_path), "execution_metrics.csv")
    append_rows_to_csv(metrics_csv_path, metrics_rows)
    if metrics_rows:
        logger.log(f"Saved {len(metrics_rows)} run metrics rows to {metrics_csv_path}")

    total_time = time.time() - start_time
    summary_log, summary = build_phase_summary(
        "PERFORMANCE",
        model,
        args.n_programs,
        total_time,
        stats_list,
        report_entries,
        args.ks_low_threshold,
    )
    logger.log(summary_log)

    metrics = [{'model': model, 'metrics': s.metrics} for s in stats_list if s.metrics]
    
    # Return list of successful files for assembly phase
    return successful_files, summary, metrics, report_entries

def run_mutation_phase(model, files, args, common_run_dir, logfile_path, logger=None):
    """Runs mutation phase for given model and valid pool of source files to mutate"""
    mutation_generated_dir = os.path.join(common_run_dir, "generated")
    mutation_failed_dir = os.path.join(common_run_dir, "failed_programs")
    os.makedirs(mutation_generated_dir, exist_ok=True)
    os.makedirs(mutation_failed_dir, exist_ok=True)

    if logger is None:
        logger = Logger(logfile_path)
    mutation_logger = logger

    if not files:
        mutation_logger.log("Mutation stage enabled but no successful generated programs were available.")
        return

    mutation_prompt_file = "mutate_prompt_template.txt"
    mutation_prompt_path = os.path.join(args.prompt_dir, mutation_prompt_file)
    if not os.path.exists(mutation_prompt_path):
        mutation_logger.log(f"Mutation prompt missing at {mutation_prompt_path}")
        print(f"Error: Mutation prompt file '{mutation_prompt_path}' not found. Skipping mutation phase.")
        return

    try:
        with open(mutation_prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        mutation_logger.log(f"\n--- Mutation Prompt Content ---\n{prompt_content}\n-------------------------------\n")
    except Exception as e:
        mutation_logger.log(f"Could not read mutation prompt file: {e}")

    mutation_count = args.n_mutations if args.n_mutations > 0 else len(files)
    start_time = time.time()
    stats_list = []
    metrics_rows = []
    report_entries = []
    successful_pool = set(files)

    # Main mutation and fixing loops
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for i in range(mutation_count):
            seed_file = random.choice(files)
            try:
                with open(seed_file, 'r', encoding='utf-8') as seed_handle:
                    seed_code = seed_handle.read()
            except Exception as error:
                mutation_logger.log(f"Failed to read mutation seed {seed_file}: {error}")
                continue

            processor = ProgramProcessor(i, model, args, mutation_logger, time.time(), stage="mutation")
            future = executor.submit(
                processor.process,
                mutation_generated_dir,
                mutation_failed_dir,
                mutation_prompt_file,
                False,
                {"input_code": seed_code},
                args.mutation_fix_cycles,
            )
            futures[future] = (processor.filename, seed_file)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Mutation {model}"):
            try:
                save_path, stats, errors, _ = future.result()
                source_filename, seed_file = futures[future]
                if stats:
                    stats_list.append(stats)

                execution_metrics = stats.metrics.get("execution", {}) if stats else {}
                compilation_metrics = stats.metrics.get("compilation", {}) if stats else {}
                low_ks_values = execution_metrics.get("low_ks_test_levels", []) if execution_metrics else []
                file_name = os.path.basename(save_path) if save_path else source_filename
                success = bool(save_path)

                error_details = build_error_details(
                    [{
                        "error": execution_metrics.get("error_summary")
                            or compilation_metrics.get("error_summary")
                            or summarize_errors(errors),
                        "error_full": execution_metrics.get("error_full")
                            or compilation_metrics.get("error_full")
                            or "\n\n---\n\n".join(errors),
                    }]
                )

                report_entries.append({
                    "file": file_name,
                    "source_file": os.path.basename(seed_file),
                    "success": success,
                    "coverage_percent": execution_metrics.get("coverage_percent", 0.0) if execution_metrics else 0.0,
                    "low_ks_test_levels": low_ks_values,
                    "error": error_details["error"],
                    "error_full": error_details["error_full"],
                })

                if success and save_path:
                    abs_path = os.path.abspath(save_path)
                    successful_pool.add(abs_path)

                metrics_rows.append(build_metrics_row(
                    model, file_name, success, execution_metrics, compilation_metrics
                ))
            except Exception as e:
                mutation_logger.log(f"Error: {e}")
                source_filename, _ = futures[future]
                error_details = build_error_details([{"error": str(e), "error_full": str(e)}])

                report_entries.append({
                    "file": source_filename,
                    "success": False,
                    "coverage_percent": 0.0,
                    "low_ks_test_levels": [],
                    "error": error_details["error"],
                    "error_full": error_details["error_full"],
                })
                metrics_rows.append(build_metrics_row(model, source_filename, False, {}, {}))

    metrics_csv_path = os.path.join(os.path.dirname(logfile_path), "mutation_execution_metrics.csv")
    append_rows_to_csv(metrics_csv_path, metrics_rows)
    if metrics_rows:
        mutation_logger.log(f"Saved {len(metrics_rows)} mutation run metrics rows to {metrics_csv_path}")

    total_time = time.time() - start_time
    summary_log, summary = build_phase_summary(
        "MUTATION",
        model,
        mutation_count,
        total_time,
        stats_list,
        report_entries,
        args.ks_low_threshold,
    )
    mutation_logger.log(summary_log)
    metrics = [{'model': model, 'metrics': s.metrics} for s in stats_list if s.metrics]

    return sorted(successful_pool), summary, metrics, report_entries

def assemble_circuits(model, files, args, base_dir, logger=None):
    """
    Assemble programs sampled from `files` (consider the full pool).
    After creating each assembled candidate, execute it and keep the resulting
    assembled file only when it is `interesting` (low KS or runtime error)
    """
    out_dir = os.path.join(base_dir, "assembled")
    metrics_csv_path = os.path.join(base_dir, "assembled_execution_metrics.csv")
    os.makedirs(out_dir, exist_ok=True)
    model_name = sanitize_model_name(model)
    seen = set()
    count = 0
    pbar = tqdm(total=args.n_assemble, desc=f"Assembling {model}")
    attempts = 0
    metrics_rows = []
    assembled_metrics = []
    while count < args.n_assemble and attempts < 1000:
        if not files:
            break

        # Ensure we don't try to pick more files than exist, or if n_circuits_per_assembly is somehow < 1
        max_k = max(min(args.n_circuits_per_assembly, len(files)), 1)
        k = random.randint(1, max_k)
        selection = tuple(random.sample(files, k))
        if selection in seen:
            attempts += 1
            continue
        seen.add(selection)
        attempts = 0

        out_path = os.path.join(out_dir, f"{model_name}_assembled_{count}.py")

        try:
            if logger:
                logger.log(
                    f"[Assembly {count + 1}/{args.n_assemble}] Building candidate from {len(selection)} source file(s): "
                    f"{', '.join(os.path.basename(path) for path in selection)}"
                )

            assemble(list(selection), out_path, count, args.language)

            with open(out_path, 'r', encoding='utf-8') as f:
                assembled_code = f.read()

            # Run the assembled program to collect metrics and KS output.
            error, output, metrics, runtime_code = run_generated_program(
                assembled_code, language=args.language, source_file_path=out_path, circuit_id=count
            )
            low_ks_values = populate_ks_test_metrics(execution_metrics, output, args.ks_low_threshold)

            # Build and persist a metrics row for this assembled candidate so
            # downstream tooling (reports/CSV) see the execution regardless of
            # whether the assembled source is kept or deleted.
            file_name = os.path.basename(out_path)
            execution_metrics = metrics or {}
            compilation_metrics = execution_metrics.get("compilation", {}) if metrics else {}
            runtime_error_full = str(execution_metrics.get("error_full") or error or "").strip()
            runtime_error_summary = str(execution_metrics.get("error_summary") or error or "").strip()
            if not runtime_error_summary and runtime_error_full:
                summary_lines = [line.strip() for line in runtime_error_full.splitlines() if line.strip()]
                runtime_error_summary = summary_lines[-1] if summary_lines else runtime_error_full

            assembly_interesting = bool(low_ks_values) or bool(runtime_error_full)
            success = not bool(runtime_error_full)

            metrics_rows.append(build_metrics_row(
                model, file_name, success, execution_metrics, compilation_metrics
            ))

            # Also collect structured metrics for plotting
            metrics_for_plot = {"execution": metrics or {}, "compilation": {}}
            assembled_metrics.append({
                "model": model,
                "metrics": metrics_for_plot,
                "success": success,
                "file": file_name,
            })

            if logger:
                if runtime_error_full:
                    logger.log(f"{file_name} Assembly Runtime Error: {runtime_error_summary}")
                    if args.verbose:
                        logger.log(f"{file_name} Assembly Runtime Error Details:\n{runtime_error_full}\n")
                elif low_ks_values:
                    low_text = format_low_ks_values(low_ks_values)
                    logger.log(
                        f"{file_name} LOW KS detected (threshold={args.ks_low_threshold}): {low_text}"
                    )
                else:
                    logger.log(f"{file_name} assembled successfully.")

                if output:
                    logger.log(f"--- {file_name} Output ---\n{output}\n--------------------------\n")

            count += 1
            pbar.update(1)

            if not assembly_interesting:
                try:
                    os.remove(out_path)
                except Exception:
                    pass

        except Exception:
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            continue

    pbar.close()
    # Persist metrics rows collected for assembled runs so they appear in
    # the common execution metrics CSV alongside generation runs.
    try:
        if metrics_rows:
            append_rows_to_csv(metrics_csv_path, metrics_rows)
            if logger:
                logger.log(f"Saved {len(metrics_rows)} assembled run metrics to {metrics_csv_path}")
    except Exception as e:
        if logger:
            logger.log(f"Warning: failed to append assembled metrics CSV: {e}")
    # Return the list of assembled files and collected structured metrics
    assembled_files = []
    try:
        prefix = model_name + "_assembled_"
        for fname in sorted(os.listdir(out_dir)):
            if fname.startswith(prefix) and fname.endswith('.py'):
                assembled_files.append(os.path.join(out_dir, fname))
    except Exception:
        pass

    return assembled_files, assembled_metrics

def main():
    parser = argparse.ArgumentParser(description="LLM Circuit Generator")
    parser.add_argument("--config_file", type=str, help="Relative path to the configuration file.")
    parser.add_argument("--run_name", type=str, help="Name for the current run. Defaults to a timestamp.")
    parser.add_argument("--language", type=str, choices=SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE, help="Language for the generated code.")
    parser.add_argument("--output_dir", type=str, help="Base directory for all outputs. Defaults to 'local_saved_circuits' within the project.")
    parser.add_argument("--prompt_dir", type=str, help="Directory containing prompt templates. Defaults to 'prompts/<language>' within the project.")
    parser.add_argument("--models", nargs='+', help="List of models to evaluate (e.g. --models openai/gpt-5.5 anthropic/claude-sonnet-4-5)")
    parser.add_argument("--n_programs", type=int, default=20, help="Number of programs to generate for each model during the production phase")
    parser.add_argument("--n_fixing_cycles", type=int, default=2, help="Maximum number of fixing cycles to perform for each generated program during training and production")
    parser.add_argument("--enable_mutation", action="store_true", default=False, help="Enable the mutation stage after generation")
    parser.add_argument("--n_mutations", type=int, default=0, help="Number of mutation candidates to produce when mutation is enabled")
    parser.add_argument("--mutation_fix_cycles", type=int, default=1, help="Maximum number of fixing cycles to perform for each mutation")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel workers for generation, mutation and assembly")
    parser.add_argument("--n_assemble", type=int, default=100, help="Number of assembled candidates to generate from the pool of generated/mutated files for each model")
    parser.add_argument("--n_circuits_per_assembly", type=int, default=2, help="Number of circuits to include in each assembly")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging with code dumps for every execution")
    parser.add_argument("--training_n", type=int, default=5, help="Number of programs to generate for each training round when improving prompts")
    parser.add_argument("--training_threshold", type=float, default=0.5, help="Threshold of fix ratio below which the prompt is considered improved enough to stop training rounds")
    parser.add_argument("--improver_model", type=str, default="anthropic/claude-sonnet-4-5", help="Model to use for improving prompts during training")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Level of reasoning effort to request from the model during generation and improvement (e.g. low, medium, high)")
    parser.add_argument("--improve_prompt", action="store_true", default=False, help="Enable prompt improvement stage")
    parser.add_argument("--max_rounds", type=int, default=3, help="Maximum number of prompt-improvement rounds during training")
    parser.add_argument(
        "--ks_low_threshold",
        type=float,
        default=0.01,
        help="Threshold below which KS-test p-values are flagged as low in production reports.",
    )

    # Parse args
    args, _ = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**(yaml.safe_load(f) or {}))
    # Re-parse CLI args to override config file settings (greater precedence)
    args = parser.parse_args()
    args.language = (args.language or DEFAULT_LANGUAGE).lower()
    if args.language not in SUPPORTED_LANGUAGES:
        parser.error(f"Unsupported language '{args.language}'. Expected one of: {', '.join(SUPPORTED_LANGUAGES)}")

    # Validate arguments to prevent runtime errors
    if args.n_circuits_per_assembly < 1:
        print(f"Warning: n_circuits_per_assembly ({args.n_circuits_per_assembly}) must be >= 1. Setting to 1.")
        args.n_circuits_per_assembly = 1
    
    if args.max_workers < 1:
        print(f"Warning: max_workers ({args.max_workers}) must be >= 1. Setting to 1.")
        args.max_workers = 1

    if args.n_programs < 0:
        print(f"Warning: n_programs ({args.n_programs}) cannot be negative. Setting to 0.")
        args.n_programs = 0

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not args.prompt_dir:
        args.prompt_dir = os.path.join(project_root, "prompts", args.language)
    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "local_saved_circuits")

    run_id = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    common_run_dir = os.path.join(args.output_dir, run_id)
    try:
        os.makedirs(common_run_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create run directory {common_run_dir}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)

    os.environ["QUILLFUZZ_RUN_DIR"] = os.path.abspath(common_run_dir)
    logfile_path = os.path.join(common_run_dir, "execution.log")

    # Start of main training, proudction and mutation loops
    all_stats = []
    all_reports = []
    main_logger = Logger(logfile_path)
    assembled_all_stats = []
    mutation_all_stats = []
    mutation_all_metrics = []
    generated_metrics_by_model = {}
    assembled_metrics_by_model = {}

    for model in args.models:
        # Train
        if args.improve_prompt:
            # The training phase returns a filename, expected to be in args.prompt_dir or created there
            best_prompt = run_training_phase(model, args, common_run_dir, logfile_path)
             
        else:
            # Default to the standard generation prompt
            best_prompt = "generation_prompt.txt"
        
        # Produce
        files, summary, metrics, report_entries = run_production_phase(
            model, best_prompt, args, common_run_dir, logfile_path, main_logger
        )
        all_stats.append(summary)
        generated_metrics_by_model[model] = metrics
        all_reports.append({"model": model, "entries": report_entries})

        # Mutate generated files from production phase, strictly adding to pool of files for assembly
        mutated_files = []
        if args.enable_mutation:
            mutated_files, mutation_summary, mutation_metrics, mutation_reports = run_mutation_phase(
                model, files, args, common_run_dir, logfile_path, main_logger
            )

            if mutation_summary is not None:
                mutation_all_stats.append(mutation_summary)
            else:
                main_logger.log(f"Mutation phase for {model} did not produce a summary.")
            if mutation_metrics:
                mutation_all_metrics.extend(mutation_metrics)
            else:
                main_logger.log(f"Mutation phase for {model} did not produce any metrics.")
            if mutation_reports:
                all_reports.append({"model": f"{model}::mutation", "entries": mutation_reports})
            else:
                main_logger.log(f"Mutation phase for {model} did not produce any report entries.")

        # Union generated and mutated files using absolute paths
        assembly_files = list(files)
        if mutated_files:
            seen_files = set()
            assembly_files = []
            for file_path in list(files) + list(mutated_files):
                abs_path = os.path.abspath(file_path)
                if abs_path in seen_files:
                    continue
                seen_files.add(abs_path)
                assembly_files.append(abs_path)

        # Assemble
        if assembly_files:
            assembly_logfile = os.path.join(common_run_dir, "assembly_execution.log")
            assembly_logger = Logger(assembly_logfile)
            assembled_files, assembled_metrics = assemble_circuits(
                model, assembly_files, args, common_run_dir, assembly_logger
            )

            # Collect assembled metrics for cross-model plots
            if assembled_metrics:
                assembled_metrics_by_model[model] = assembled_metrics
                # Build a minimal summary for assembled results per model
                assembled_all_stats.append({
                    'model': model,
                    'total_cost': 0.0,
                    'total_time': 0.0,
                    'total_programs': len(assembled_metrics),
                    'valid_programs': sum(1 for m in assembled_metrics if m.get('success')),
                    'avg_quality_score': 0.0,
                })

    if all_stats:
        generate_summary_plot(all_stats, os.path.join(common_run_dir, "plots", "performance"))

    generated_complexity_root = os.path.join(common_run_dir, "plots", "complexity")
    for model, model_metrics in generated_metrics_by_model.items():
        if model_metrics:
            generate_complexity_scatter_plots(
                model_metrics,
                os.path.join(generated_complexity_root, sanitize_model_name(model)),
            )

    # Plots for mutation circuits
    if mutation_all_stats:
        generate_summary_plot(mutation_all_stats, os.path.join(common_run_dir, "plots", "mutation_performance"))
    if mutation_all_metrics:
        generate_complexity_scatter_plots(mutation_all_metrics, os.path.join(common_run_dir, "plots", "mutation_complexity"))

    # Plots for assembled circuits (separate folder)
    if assembled_all_stats:
        generate_summary_plot(assembled_all_stats, os.path.join(common_run_dir, "assembled_plots", "performance"))

    assembled_complexity_root = os.path.join(common_run_dir, "assembled_plots", "complexity")
    for model, model_metrics in assembled_metrics_by_model.items():
        if model_metrics:
            generate_complexity_scatter_plots(
                model_metrics,
                os.path.join(assembled_complexity_root, sanitize_model_name(model)),
            )

    performance_summary_path = os.path.join(common_run_dir, "performance_summary.json")
    performance_summary = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": args.language,
        "models": args.models,
        "model_summaries": all_stats,
        "per_file_reports": all_reports,
    }
    with open(performance_summary_path, "w", encoding="utf-8") as f:
        json.dump(performance_summary, f, indent=2)
    print(f"Performance summary JSON: {performance_summary_path}")

    # Generate JSON summary for assembled circuits
    assembled_summary_path = os.path.join(common_run_dir, "assembled_performance_summary.json")
    assembled_summary = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": args.language,
        "models": args.models,
        "model_summaries": assembled_all_stats,
    }
    with open(assembled_summary_path, "w", encoding="utf-8") as f:
        json.dump(assembled_summary, f, indent=2)
    print(f"Assembled performance summary JSON: {assembled_summary_path}")

if __name__ == "__main__":
    main()
