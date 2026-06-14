import concurrent.futures
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from utils.circuit_assembler import assemble
from utils.llm_client import ask_any_model, get_dynamic_prompt, improve_prompt_logic
from utils.execution import compile_generated_program, run_generated_program
from utils.reporting import (
    Logger,
    append_rows_to_csv,
    build_error_details,
    build_metrics_row,
    build_phase_summary,
    format_low_ks_values,
    populate_ks_test_metrics,
    summarize_errors,
)
from utils.utils import save_text_to_file, sanitize_model_name


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
    def __init__(self, index, model, config, logger, start_time, stage="generation", stop_controller=None):
        self.index = index
        self.model = model
        self.config = config
        self.logger = logger
        self.start_time = start_time
        self.stage = stage
        self.stop_controller = stop_controller
        self.filename = f"{sanitize_model_name(self.model)}_{self.stage}output{self.index+1}.py"
        self.stats = GenerationStats()
        self.encountered_errors = []

    def _stop_requested(self):
        return bool(self.stop_controller and self.stop_controller.stop_requested)

    def generate(self, prompt_filename, prompt_kwargs=None):
        if self._stop_requested():
            self.logger.log(f"Skipping generation for {self.filename}: early stop requested.")
            return None

        prompt_kwargs = prompt_kwargs or {}
        prompt_path = prompt_filename if os.path.isabs(prompt_filename) else os.path.join(self.config.prompt_dir, prompt_filename)

        if not os.path.exists(prompt_path):
            self.logger.log(f"{self.stage.capitalize()} prompt not found at {prompt_path}")
            return None

        prompt = get_dynamic_prompt(prompt_path, **prompt_kwargs)

        if prompt is None:
            self.logger.log(f"Error: Prompt for {self.filename} is empty.\n")
            return None

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
        if self._stop_requested():
            self.logger.log(f"Skipping compile check for {self.filename}: early stop requested.")
            return False, "Early stop requested"

        self.logger.log(f"--- {f'[Elapsed: {time.time() - self.start_time:.2f}s]'} Testing {self.stage} {self.filename} ---\n")
        error, _, metrics, wrapped_code = compile_generated_program(code, language=self.config.language)
        metrics = metrics or {}
        self.stats.metrics['compilation'] = metrics

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
        if self._stop_requested():
            self.logger.log(f"Skipping runtime check for {self.filename}: early stop requested.")
            return False, "Early stop requested"

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
            ks_low_threshold=self.config.ks_low_threshold,
        )
        metrics = metrics or {}
        self.stats.metrics['execution'] = metrics

        low_ks_values = populate_ks_test_metrics(metrics, output, self.config.ks_low_threshold)
        if low_ks_values:
            self.logger.log(
                f"{self.filename} LOW KS detected (threshold={self.config.ks_low_threshold}): {format_low_ks_values(low_ks_values)}"
            )

        self.stats.execution_quality_score = metrics.get('quality_score', 0.0)

        if error.strip():
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
            if self._stop_requested():
                self.logger.log(f"Stopping fix loop for {self.filename}: early stop requested.")
                break
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

            compile_ok, compile_err = self.compile_check(fixed_code)

            if not compile_ok:
                self.logger.log(f"Fixed {self.filename} (Cycle {cycle+1}) Compilation Failed.\n")
                current_error = compile_err
                last_failure_type = "compile_fail"
                continue

            self.logger.log(f"Fixed {self.filename} (Cycle {cycle+1}) compiled successfully.\n")

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
        self.config.compile_only = compile_only
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
                    if self._stop_requested():
                        return None, self.stats, list(set(self.encountered_errors)), False
                    save_path = os.path.join(runtime_fail_dir, self.filename)
                    generated_path = os.path.join(generated_dir, self.filename)
                    if os.path.exists(generated_path):
                        os.remove(generated_path)
                    save_text_to_file(code, save_path)
                    return None, self.stats, list(set(self.encountered_errors)), False

            save_path = os.path.join(generated_dir, self.filename)
            save_text_to_file(code, save_path)
            return save_path, self.stats, list(set(self.encountered_errors)), False

        if self._stop_requested():
            return None, self.stats, list(set(self.encountered_errors)), False

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
    total_programs_processed = 0

    for round_idx in range(max_rounds + 1):
        if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
            training_logger.log("Stopping training phase early due to interactive stop request.")
            break

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

        # For each training loop, run generation prompts in parallel until target has been reached for each round
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            submitted = 0
            completed_round = 0
            pbar = tqdm(total=args.training_n, desc=f"Training Round {round_idx}")

            while completed_round < args.training_n:
                while submitted < args.training_n and len(futures) < args.max_workers:
                    if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
                        break
                    processor = ProgramProcessor(
                        submitted,
                        model,
                        args,
                        training_logger,
                        time.time(),
                        stop_controller=getattr(args, "stop_controller", None),
                    )
                    future = executor.submit(processor.process, t_gen_dir, t_fail_dir, prompt_filename, True)
                    futures[future] = submitted
                    submitted += 1

                if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested and not futures:
                    break

                if not futures:
                    break

                done, _ = concurrent.futures.wait(
                    list(futures.keys()),
                    timeout=0.25,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                if not done:
                    continue

                for future in done:
                    completed_round += 1
                    pbar.update(1)
                    futures.pop(future, None)
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

            pbar.close()

        if completed_round == 0:
            training_logger.log(f"Round {round_idx} ended before any programs completed.")
            break

        total_programs_processed += completed_round
        fix_ratio = count_needed_fix / completed_round
        rounds_completed += 1
        training_logger.log(
            f"Round {round_idx} Result: {count_needed_fix}/{completed_round} fixes (Ratio: {fix_ratio:.2f})"
        )

        if fix_ratio < best_fix_ratio:
            best_fix_ratio = fix_ratio
            best_prompt = prompt_filename

        if fix_ratio < args.training_threshold:
            training_logger.log(f"Training success! Proceeding with {prompt_filename}")
            summary_log, _ = build_phase_summary(
                "TRAINING",
                model,
                total_programs_processed,
                time.time() - total_start_time,
                training_stats,
                training_reports,
                args.ks_low_threshold,
            )
            training_logger.log(summary_log)
            return prompt_filename

        if args.improve_prompt and round_idx < max_rounds:
            training_logger.log("Threshold not met. Improving prompt...")

            current_prompt_path = prompt_filename
            if not os.path.isabs(current_prompt_path):
                current_prompt_path = os.path.join(args.prompt_dir, current_prompt_path)

            new_prompt_path = os.path.join(round_dir, "improved_prompt.txt")

            try:
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
        total_programs_processed,
        time.time() - total_start_time,
        training_stats,
        training_reports,
        args.ks_low_threshold,
    )
    training_logger.log(summary_log)
    return best_prompt


def run_production_phase(model, prompt_filename, args, common_run_dir, logfile_path, logger=None):
    """Run the production phase to generate final programs with the (optionally) improved prompt."""

    if logger is None:
        logger = Logger(logfile_path)
    
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

    # Log prompt for versioning and debugging purposes
    try:
        with open(prompt_path, 'r') as f:
            prompt_content = f.read()
        logger.log(f"\n--- Prompt Content ---\n{prompt_content}\n----------------------\n")
    except Exception as e:
        logger.log(f"Could not read prompt file for verbose logging: {e}")

    start_time = time.time()
    stats_list = []
    metrics_rows = []
    report_entries = []
    successful_files = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        submitted = 0
        completed = 0
        pbar = tqdm(total=args.n_programs, desc=f"Production {model}")

        while completed < args.n_programs:
            while submitted < args.n_programs and len(futures) < args.max_workers:
                if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
                    break
                processor = ProgramProcessor(
                    submitted,
                    model,
                    args,
                    logger,
                    time.time(),
                    stop_controller=getattr(args, "stop_controller", None),
                )
                future = executor.submit(processor.process, gen_dir, fail_dir, prompt_filename, False)
                futures[future] = processor.filename
                submitted += 1

            if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested and not futures:
                break

            if not futures:
                break

            done, _ = concurrent.futures.wait(
                list(futures.keys()),
                timeout=0.25,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                continue

            for future in done:
                completed += 1
                pbar.update(1)
                source_filename = futures.pop(future)
                try:
                    save_path, stats, errors, _ = future.result()
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
                        model, file_name, success, execution_metrics, compilation_metrics, getattr(stats, 'cost', 0.0)
                    ))
                except Exception as e:
                    logger.log(f"Error: {e}")
                    error_details = build_error_details([{"error": str(e), "error_full": str(e)}])

                    report_entries.append({
                        "file": source_filename,
                        "success": False,
                        "coverage_percent": 0.0,
                        "low_ks_test_levels": [],
                        "error": error_details["error"],
                        "error_full": error_details["error_full"],
                    })
                    metrics_rows.append(build_metrics_row(model, source_filename, False, {}, {}, 0.0))

        pbar.close()

    if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
        logger.log(
            f"Production phase stopped early at {len(report_entries)}/{args.n_programs} completed programs due to interactive stop request."
        )

    metrics_csv_path = os.path.join(os.path.dirname(logfile_path), "execution_metrics.csv")
    append_rows_to_csv(metrics_csv_path, metrics_rows)
    if metrics_rows:
        logger.log(f"Saved {len(metrics_rows)} run metrics rows to {metrics_csv_path}")

    total_time = time.time() - start_time
    summary_log, summary = build_phase_summary(
        "PERFORMANCE",
        model,
        len(report_entries),
        total_time,
        stats_list,
        report_entries,
        args.ks_low_threshold,
    )
    logger.log(summary_log)

    metrics = [{'model': model, 'metrics': s.metrics} for s in stats_list if s.metrics]

    return successful_files, summary, metrics, report_entries


def run_mutation_phase(model, files, args, common_run_dir, logfile_path, logger=None):
    """Runs mutation phase for given model and valid pool of source files to mutate"""
    
    if logger is None:
        logger = Logger(logfile_path)
    
    mutation_generated_dir = os.path.join(common_run_dir, "generated")
    mutation_failed_dir = os.path.join(common_run_dir, "failed_programs")
    os.makedirs(mutation_generated_dir, exist_ok=True)
    os.makedirs(mutation_failed_dir, exist_ok=True)

    if not files:
        logger.log("Mutation stage enabled but no successful generated programs were available.")
        return [], None, [], []

    mutation_prompt_file = "mutate_prompt_template.txt"
    mutation_prompt_path = os.path.join(args.prompt_dir, mutation_prompt_file)

    try:
        with open(mutation_prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        logger.log(f"\n--- Mutation Prompt Content ---\n{prompt_content}\n-------------------------------\n")
    except Exception as e:
        logger.log(f"Could not read mutation prompt file: {e}")
        print(f"Error: Mutation prompt file '{mutation_prompt_path}' not found. Skipping mutation phase.")
        return [], None, [], []

    n_mutations = getattr(args, "n_mutations", None)
    mutation_count = n_mutations if (n_mutations is not None and n_mutations >= 0) else len(files)
    start_time = time.time()
    stats_list = []
    metrics_rows = []
    report_entries = []
    successful_pool = set(files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        submitted = 0
        completed = 0
        pbar = tqdm(total=mutation_count, desc=f"Mutation {model}")

        while completed < mutation_count:
            while submitted < mutation_count and len(futures) < args.max_workers:
                if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
                    break
                seed_file = random.choice(files)
                try:
                    with open(seed_file, 'r', encoding='utf-8') as seed_handle:
                        seed_code = seed_handle.read()
                except Exception as error:
                    logger.log(f"Failed to read mutation seed {seed_file}: {error}")
                    submitted += 1
                    pbar.update(1)
                    completed += 1
                    continue

                processor = ProgramProcessor(
                    submitted,
                    model,
                    args,
                    logger,
                    time.time(),
                    stage="mutation",
                    stop_controller=getattr(args, "stop_controller", None),
                )
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
                submitted += 1

            if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested and not futures:
                break

            if not futures:
                break

            done, _ = concurrent.futures.wait(
                list(futures.keys()),
                timeout=0.25,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                continue

            for future in done:
                completed += 1
                pbar.update(1)
                source_filename, seed_file = futures.pop(future)
                try:
                    save_path, stats, errors, _ = future.result()
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
                        model, file_name, success, execution_metrics, compilation_metrics, getattr(stats, 'cost', 0.0)
                    ))
                except Exception as e:
                    logger.log(f"Error: {e}")
                    error_details = build_error_details([{"error": str(e), "error_full": str(e)}])

                    report_entries.append({
                        "file": source_filename,
                        "success": False,
                        "coverage_percent": 0.0,
                        "low_ks_test_levels": [],
                        "error": error_details["error"],
                        "error_full": error_details["error_full"],
                    })
                    metrics_rows.append(build_metrics_row(model, source_filename, False, {}, {}, 0.0))

        pbar.close()

    if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
        logger.log(
            f"Mutation phase stopped early at {len(report_entries)}/{mutation_count} completed candidates due to interactive stop request."
        )

    metrics_csv_path = os.path.join(os.path.dirname(logfile_path), "mutation_execution_metrics.csv")
    append_rows_to_csv(metrics_csv_path, metrics_rows)
    if metrics_rows:
        logger.log(f"Saved {len(metrics_rows)} mutation run metrics rows to {metrics_csv_path}")

    total_time = time.time() - start_time
    summary_log, summary = build_phase_summary(
        "MUTATION",
        model,
        len(report_entries),
        total_time,
        stats_list,
        report_entries,
        args.ks_low_threshold,
    )
    logger.log(summary_log)
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
    submitted = 0
    completed = 0
    attempts = 0
    metrics_rows = []
    assembled_metrics = []
    report_entries = []
    pbar = tqdm(total=args.n_assemble, desc=f"Assembling {model}")

    def _run_assembled_candidate(path, code, cid):
        try:
            coverage_artifact_dir = getattr(args, 'coverage_artifacts_dir', None)
            return run_generated_program(
                code,
                language=args.language,
                source_file_path=path,
                circuit_id=cid,
                ks_low_threshold=args.ks_low_threshold,
                coverage_artifact_dir=coverage_artifact_dir,
            )
        except Exception as e:
            return str(e), "", {}, ""

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        while (submitted < args.n_assemble and attempts < 1000) or futures:
            while submitted < args.n_assemble and attempts < 1000 and len(futures) < args.max_workers:
                if not files:
                    break
                if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
                    break

                max_k = max(min(args.n_circuits_per_assembly, len(files)), 1)
                k = random.randint(1, max_k)
                selection = tuple(random.sample(files, k))
                if selection in seen:
                    attempts += 1
                    continue
                seen.add(selection)
                attempts = 0

                out_path = os.path.join(out_dir, f"{model_name}_assembled_{submitted}.py")

                try:
                    logger.log(
                        f"[Assembly {submitted + 1}/{args.n_assemble}] Building candidate from {len(selection)} source file(s): "
                        f"{', '.join(os.path.basename(path) for path in selection)}"
                    )

                    assemble(list(selection), out_path, submitted, args.language)

                    with open(out_path, 'r', encoding='utf-8') as f:
                        assembled_code = f.read()

                    logger.log(f"Preparing to run assembled candidate {os.path.basename(out_path)}...")

                    future = executor.submit(_run_assembled_candidate, out_path, assembled_code, submitted)
                    futures[future] = (out_path, assembled_code, submitted, selection)
                    submitted += 1

                except Exception as e:
                    logger.log(f"Error occurred while preparing {out_path}: {e}")
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                    except Exception:
                        pass
                    continue

            if not futures:
                if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
                    break
                if submitted >= args.n_assemble or attempts >= 1000:
                    break
                continue

            done, _ = concurrent.futures.wait(
                list(futures.keys()),
                timeout=0.25,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )

            if not done:
                continue

            for future in done:
                out_path, assembled_code, cid, selection = futures.pop(future)
                try:
                    result = future.result()
                    if isinstance(result, tuple) and len(result) == 4:
                        error, output, metrics, runtime_code = result
                    else:
                        error, output, metrics, runtime_code = str(result), "", {}, ""
                except Exception as e:
                    error, output, metrics, runtime_code = str(e), "", {}, ""

                logger.log(f"Execution completed for {os.path.basename(out_path)}. Processing results...")

                execution_metrics = metrics or {}
                low_ks_values = populate_ks_test_metrics(execution_metrics, output, args.ks_low_threshold)
                file_name = os.path.basename(out_path)
                source_files = [os.path.basename(p) for p in selection] if selection else []

                compilation_metrics = execution_metrics.get("compilation", {}) if metrics else {}
                runtime_error_full = str(execution_metrics.get("error_full") or error or "").strip()
                runtime_error_summary = str(execution_metrics.get("error_summary") or error or "").strip()
                if not runtime_error_summary and runtime_error_full:
                    summary_lines = [line.strip() for line in runtime_error_full.splitlines() if line.strip()]
                    runtime_error_summary = summary_lines[-1] if summary_lines else runtime_error_full

                assembly_interesting = bool(low_ks_values) or bool(runtime_error_full)
                success = not bool(runtime_error_full)

                metrics_rows.append(build_metrics_row(
                    model, file_name, success, execution_metrics, compilation_metrics, 0.0
                ))

                metrics_for_plot = {"execution": metrics or {}, "compilation": {}}
                assembled_metrics.append({
                    "model": model,
                    "metrics": metrics_for_plot,
                    "success": success,
                    "file": file_name,
                })

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

                completed += 1
                pbar.update(1)

                if not assembly_interesting:
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
                # Build per-file report entry
                error_details = build_error_details([
                    {
                        "error": execution_metrics.get("error_summary")
                            or compilation_metrics.get("error_summary")
                            or runtime_error_summary,
                        "error_full": execution_metrics.get("error_full")
                            or compilation_metrics.get("error_full")
                            or runtime_error_full,
                    }
                ])

                report_entries.append({
                    "file": file_name,
                    "source_files": source_files,
                    "success": success,
                    "coverage_percent": execution_metrics.get("coverage_percent", 0.0) if execution_metrics else 0.0,
                    "low_ks_test_levels": low_ks_values,
                    "error": error_details["error"],
                    "error_full": error_details["error_full"],
                })

    pbar.close()

    if getattr(args, "stop_controller", None) and args.stop_controller.stop_requested:
        logger.log(
            f"Assembly phase stopped early at {completed}/{args.n_assemble} completed candidates due to interactive stop request."
        )

    try:
        if metrics_rows:
            append_rows_to_csv(metrics_csv_path, metrics_rows)
            logger.log(f"Saved {len(metrics_rows)} assembled run metrics to {metrics_csv_path}")
        else:
            logger.log("No metrics collected from assembled runs to save.")
    except Exception as e:
        logger.log(f"Warning: failed to append assembled metrics CSV: {e}")

    assembled_files = []
    try:
        prefix = model_name + "_assembled_"
        for fname in sorted(os.listdir(out_dir)):
            if fname.startswith(prefix) and fname.endswith('.py'):
                assembled_files.append(os.path.join(out_dir, fname))
    except Exception:
        logger.log(f"Warning: failed to list assembled files in {out_dir}")

    return assembled_files, assembled_metrics, report_entries
