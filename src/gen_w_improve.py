import os
import sys
import time
import threading
import concurrent.futures
import random
import argparse
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from tqdm import tqdm

# Add project root to path so we can import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import from local library
from utils.circuit_assembler import assemble
from utils.llm_client import ask_any_model, get_dynamic_prompt
from utils.utils import save_text_to_file, generate_summary_plot, generate_complexity_scatter_plots
from utils.execution import run_generated_program, compile_generated_program

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
        if not stats: return
        self.cost += stats.get('cost', 0.0)
        self.prompt_tokens += stats.get('prompt_tokens', 0)
        self.completion_tokens += stats.get('completion_tokens', 0)
        self.total_tokens += stats.get('total_tokens', 0)
        if 'quality_score' in stats:
            self.quality_score = stats['quality_score']

class Logger:
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path
        self.lock = threading.Lock()

    def log(self, message: str):
        if not self.logfile_path: return
        with self.lock:
            if not message.endswith('\n'):
                message += '\n'
            # Check if file is open elsewhere or just append
            with open(self.logfile_path, "a") as f:
                f.write(message)


class ProgramProcessor:
    def __init__(self, index, model, config, logger, start_time):
        self.index = index
        self.model = model
        self.config = config
        self.logger = logger
        self.start_time = start_time
        self.filename = f"{model.replace('/', '_')}output{index+1}.py"
        self.stats = GenerationStats()
        self.encountered_errors = []

    @property
    def elapsed(self):
        return f"[Elapsed: {time.time() - self.start_time:.2f}s]"

    def log(self, msg):
        self.logger.log(msg)

    def generate(self, prompt_filename):
        if os.path.isabs(prompt_filename):
            prompt_path = prompt_filename
        else:
            prompt_path = os.path.join(self.config.prompt_dir, prompt_filename)
            
        if not os.path.exists(prompt_path):
            self.log(f"Generation prompt not found at {prompt_path}")
            return None

        prompt = get_dynamic_prompt(prompt_path)
        # Pass reasoning_effort from config
        code, stats, err = ask_any_model(self.model, prompt, reasoning_effort=self.config.reasoning_effort)
        
        if code is None:
            self.log(f"Failed to generate {self.filename}. Error: {err}\n")
            self.encountered_errors.append(f"Generation API Error: {err}")
            return None

        self.stats.update(stats)
        self.log(f"{self.filename} Generation Cost: ${stats.get('cost', 0.0):.6f} | "
                 f"Tokens (In/Out/Total): {stats.get('prompt_tokens', 0)}/{stats.get('completion_tokens', 0)}/{stats.get('total_tokens', 0)}")
        return code

    def compile_check(self, code):
        self.log(f"--- {self.elapsed} Testing generated {self.filename} ---\n")
        error, _, metrics, wrapped_code = compile_generated_program(code, language=self.config.language)
        self.stats.metrics['compilation'] = metrics
        
        # Store compiliation quality score as the main quality_score
        if error:
            self.stats.quality_score = 0.0
            self.log(f"{self.filename} Compilation Failed:\n{error}\n")
            self.encountered_errors.append(f"Compilation Error:\n{error}")
            if self.config.verbose:
                self.log(f"--- {self.filename} Code ---\n{wrapped_code}\n-----------------------\n")
            return False, error
        
        self.stats.quality_score = metrics.get('quality_score', 0.0)
        return True, ""

    def run_check(self, code):
        self.log(f"--- {self.elapsed} Running {self.filename} ---\n")
        error, _, metrics, runtime_code = run_generated_program(code, language=self.config.language)
        self.stats.metrics['execution'] = metrics
        
        # Store execution quality score separately
        self.stats.execution_quality_score = metrics.get('quality_score', 0.0)
        self.log(f"{self.filename} Metrics: {self.stats.metrics}\n")

        if error.strip():
            self.log(f"{self.filename} Runtime Error:\n{error}\n")
            self.encountered_errors.append(f"Runtime Error:\n{error}")
            if self.config.verbose:
                self.log(f"--- {self.filename} Code ---\n{runtime_code}\n-----------------------\n")
            return False, error
        
        self.log(f"{self.filename} ran successfully.\n")
        return True, ""

    def fix_loop(self, code, initial_error):
        current_code = code
        current_error = initial_error
        
        for cycle in range(self.config.n_fixing_cycles):
            prompt_path = os.path.join(self.config.prompt_dir, "fixing_prompt_template.txt")
            if not os.path.exists(prompt_path):
                self.log(f"Fixing prompt missing at {prompt_path}\n")
                return None

            prompt = get_dynamic_prompt(prompt_path, faulty_code=current_code, error_message=current_error)
            fixed_code, stats, err = ask_any_model(self.model, prompt, reasoning_effort=self.config.reasoning_effort)

            if not fixed_code:
                self.log(f"Fixing cycle {cycle+1} failed for {self.filename}: {err}\n")
                break
            
            self.stats.update(stats)
            self.log(f"{self.filename} Fixing (Cycle {cycle+1}) Cost: ${stats.get('cost', 0.0):.6f} | "
                     f"Tokens (In/Out/Total): {stats.get('prompt_tokens', 0)}/{stats.get('completion_tokens', 0)}/{stats.get('total_tokens', 0)}")

            # Verify fix (Compile only first)
            compile_ok, compile_err = self.compile_check(fixed_code)
            
            if not compile_ok:
                self.log(f"Fixed {self.filename} (Cycle {cycle+1}) Compilation Failed.\n")
                current_code = fixed_code
                current_error = compile_err
                continue

            self.log(f"Fixed {self.filename} (Cycle {cycle+1}) compiled successfully.\n")
            
            # If we need to run it
            if not self.config.compile_only:
                run_ok, run_err = self.run_check(fixed_code)
                if run_ok:
                    return fixed_code
                else:
                    return fixed_code
            else:
                return fixed_code

        return None

    def process(self, generated_dir, failed_dir, prompt_filename="generation_prompt.txt", compile_only=False):
        self.config.compile_only = compile_only # augment config temporarily
        
        code = self.generate(prompt_filename)
        if not code:
            return None, self.stats, self.encountered_errors, False

        compile_ok, compile_err = self.compile_check(code)
        
        if compile_ok:
             if not compile_only:
                 self.run_check(code)
             
             save_path = os.path.join(generated_dir, self.filename)
             save_text_to_file(code, save_path)
             return save_path, self.stats, list(set(self.encountered_errors)), False

        # If compilation failed, try fixing
        fixed_code = self.fix_loop(code, compile_err)
        
        if fixed_code:
            save_path = os.path.join(generated_dir, self.filename)
            save_text_to_file(fixed_code, save_path)
            return save_path, self.stats, list(set(self.encountered_errors)), True
        else:
            save_path = os.path.join(failed_dir, self.filename)
            save_text_to_file(code, save_path)
            return None, self.stats, list(set(self.encountered_errors)), True


def improve_prompt_logic(improver_model, prompt_path, common_prompt_dir, output_path, error_logs, language, logger=None, reasoning_effort="high"):
    """
    Analyzes errors and rewrites the prompt to improve generation.
    """
    if not os.path.exists(prompt_path):
        if logger:
            logger.log(f"Original prompt file not found at {prompt_path}. Skipping improvement.")
        print(f"Original prompt file not found at {prompt_path}. Skipping improvement.")
        sys.exit(1)

    with open(prompt_path, 'r') as f:
        original_content = f.read()

    unique_errors = list(set(error_logs))[:10]  # Limit errors
    errors_text = "\n---\n".join(unique_errors)
    
    template_path = os.path.join(common_prompt_dir, "prompt_improvement_template.txt")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt improvement template missing at {template_path}")
        
    meta_prompt = get_dynamic_prompt(template_path, language=language, original_content=original_content, errors_text=errors_text)
    
    print(f"\n[Training] requesting prompt improvement from {improver_model}...")
    improved_content, _, err = ask_any_model(improver_model, meta_prompt, reasoning_effort=reasoning_effort)
    
    if improved_content:
        with open(output_path, 'w') as f:
            f.write(improved_content)
        
        if logger:
            logger.log(f"\n--- Improved Prompt Content ({time.ctime()}) ---\n{improved_content}\n-----------------------------------------------\n")

        print(f"[Training] Improved prompt saved to {output_path}")
        return output_path
    else:
        print(f"[Training] Failed to improve prompt: {err}")
        return prompt_path

def run_training_phase(model, args, common_run_dir, main_logfile_path):
    # Main logger for high-level info if needed, but we'll use per-round logs
    # logger = Logger(main_logfile_path) 
    
    prompt_filename = "generation_prompt.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.training_n <= 0:
        return prompt_filename

    max_rounds = 3 

    best_prompt = prompt_filename
    best_fix_ratio = 1.0

    for round_idx in range(max_rounds + 1):
        # Setup directories
        round_dir = os.path.join(common_run_dir, "training_phase", model.replace('/', '_'), f"round_{round_idx}")
        t_gen_dir = os.path.join(round_dir, "generated")
        t_fail_dir = os.path.join(round_dir, "failed")
        os.makedirs(t_gen_dir, exist_ok=True)
        os.makedirs(t_fail_dir, exist_ok=True)

        # Per-round logger
        round_logfile = os.path.join(round_dir, f"round_{round_idx}_execution.log")
        logger = Logger(round_logfile)
        logger.log(f"\n[Training Round {round_idx}] Model: {model} | Prompt: {prompt_filename}")
        logger.log(f"Reasoning Effort: {args.reasoning_effort}")

        training_errors = []
        count_needed_fix = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for i in range(args.training_n):
                processor = ProgramProcessor(i, model, args, logger, time.time())
                futures.append(executor.submit(
                    processor.process, t_gen_dir, t_fail_dir, prompt_filename, True
                ))

            for future in tqdm(concurrent.futures.as_completed(futures), total=args.training_n, desc=f"Training Round {round_idx}"):
                try:
                    _, _, errors, was_fixed = future.result()
                    training_errors.extend(errors)
                    if was_fixed:
                        count_needed_fix += 1
                except Exception as e:
                    logger.log(f"Training error: {e}")

        fix_ratio = count_needed_fix / args.training_n
        logger.log(f"Round {round_idx} Result: {count_needed_fix}/{args.training_n} fixes (Ratio: {fix_ratio:.2f})")

        # Update best prompt if this round is strictly better
        if fix_ratio < best_fix_ratio:
            best_fix_ratio = fix_ratio
            best_prompt = prompt_filename

        if fix_ratio < args.training_threshold:
            logger.log(f"Training success! Proceeding with {prompt_filename}")
            return prompt_filename

        if args.improve_prompt and round_idx < max_rounds:
            logger.log("Threshold not met. Improving prompt...")
            
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
                    os.path.join(script_dir, "Common_prompt_templates"),
                    new_prompt_path, 
                    training_errors, 
                    args.language, 
                    logger,
                    reasoning_effort=args.reasoning_effort
                )
            except Exception as e:
                logger.log(f"Error improving prompt: {e}")
                import traceback
                logger.log(traceback.format_exc())
        else:
            logger.log("Max rounds reached or improvement disabled.")
            
    return best_prompt

def run_production_phase(model, prompt_filename, args, common_run_dir, logfile_path):
    logger = Logger(logfile_path)
    start_time = time.time()
    model_run_dir = common_run_dir
    gen_dir = os.path.join(model_run_dir, "generated")
    fail_dir = os.path.join(model_run_dir, "failed_programs")
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

    successful_files = []
    stats_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for i in range(args.n_programs):
            processor = ProgramProcessor(i, model, args, logger, time.time())
            futures.append(executor.submit(
                processor.process, gen_dir, fail_dir, prompt_filename, False
            ))

        for future in tqdm(concurrent.futures.as_completed(futures), total=args.n_programs, desc=f"Production {model}"):
            try:
                save_path, stats, _, _ = future.result()
                if stats:
                    stats_list.append(stats)
                if save_path:
                    successful_files.append(save_path)
            except Exception as e:
                logger.log(f"Error: {e}")

    # Aggregating Stats
    total_time = time.time() - start_time
    total_cost = sum(s.cost for s in stats_list)
    quality_scores = [s.quality_score for s in stats_list if s.quality_score is not None]
    avg_quality = sum(quality_scores)/len(quality_scores) if quality_scores else 0
    
    total_prompt_tokens = sum(s.prompt_tokens for s in stats_list)
    total_completion_tokens = sum(s.completion_tokens for s in stats_list)
    total_tokens = sum(s.total_tokens for s in stats_list)
    avg_time_per_valid = total_time / len(successful_files) if successful_files else 0

    summary_log = f"""
============================================================
  PERFORMANCE SUMMARY for {model}
------------------------------------------------------------
  Target Number of Programs : {args.n_programs}
  Total Valid Programs     : {len(successful_files)}
  Total Time Taken         : {total_time:.2f} seconds
  Avg Time per Valid Prog  : {avg_time_per_valid:.2f} seconds
  Avg Quality Score        : {avg_quality:.4f}
------------------------------------------------------------
  Total Cost (Estimated)   : ${total_cost:.6f}
  Total Prompt Tokens      : {total_prompt_tokens}
  Total Completion Tokens  : {total_completion_tokens}
  Total Tokens             : {total_tokens}
============================================================
"""
    logger.log(summary_log)

    summary = {
        "model": model,
        "total_cost": total_cost,
        "total_time": total_time,
        "total_programs": args.n_programs,
        "valid_programs": len(successful_files),
        "avg_quality_score": avg_quality,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens
    }
    
    metrics = [{'model': model, 'metrics': s.metrics} for s in stats_list if s.metrics]
    
    return successful_files, summary, metrics

def assemble_circuits(model, files, args, base_dir):
    out_dir = os.path.join(base_dir, "assembled")
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    count = 0
    pbar = tqdm(total=args.n_assemble, desc=f"Assembling {model}")
    
    attempts = 0
    while count < args.n_assemble and attempts < 1000:
        if not files:
            break

        # Ensure we don't try to pick more files than exist, or if n_circuits_per_assembly is somehow < 1
        max_k = min(args.n_circuits_per_assembly, len(files))
        if max_k < 1:
            max_k = 1
            
        k = random.randint(1, max_k)
        selection = tuple(random.sample(files, k))
        if selection in seen:
            attempts += 1
            continue
        seen.add(selection)
        attempts = 0
        
        try:
            assemble(list(selection), os.path.join(out_dir, f"{model.replace('/', '_')}_{count}.py"), count, args.language)
            count += 1
            pbar.update(1)
        except Exception:
            pass
    pbar.close()

def main():
    parser = argparse.ArgumentParser(description="LLM Circuit Generator")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--language", type=str, default="guppy")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--prompt_dir", type=str)
    parser.add_argument("--models", nargs='+', default=["deepseek/deepseek-chat"])
    parser.add_argument("--n_programs", type=int, default=20)
    parser.add_argument("--n_fixing_cycles", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--n_assemble", type=int, default=100)
    parser.add_argument("--n_circuits_per_assembly", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--training_n", type=int, default=5)
    parser.add_argument("--training_threshold", type=float, default=0.5)
    parser.add_argument("--improver_model", type=str, default="anthropic/claude-sonnet-4-5")
    parser.add_argument("--reasoning_effort", type=str, default="high")
    parser.add_argument("--improve_prompt", action="store_true", default=False, help="Enable prompt improvement stage")

    args, _ = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
            args = parser.parse_args()

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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.prompt_dir:
        args.prompt_dir = os.path.join(script_dir, "Guppy_prompt_templates" if args.language == "guppy" else "Qiskit_prompt_templates")
    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(script_dir), "local_saved_circuits")

    run_id = args.run_name or time.strftime("%Y%m%d_%H%M%S_improved")
    common_run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(common_run_dir, exist_ok=True)
    logfile_path = os.path.join(common_run_dir, "execution.log")

    all_stats = []
    all_metrics = []

    for model in args.models:
        # Train
        if args.improve_prompt:
             best_prompt = run_training_phase(model, args, common_run_dir, logfile_path)
             # The training phase returns a filename, expected to be in args.prompt_dir or created there
        else:
             # Default to the standard generation prompt
             best_prompt = "generation_prompt.txt"
        
        # Produce
        files, summary, metrics = run_production_phase(model, best_prompt, args, common_run_dir, logfile_path)
        all_stats.append(summary)
        all_metrics.extend(metrics)

        # Assemble
        if files:
            assemble_circuits(model, files, args, common_run_dir)

    if all_stats:
        generate_summary_plot(all_stats, os.path.join(common_run_dir, "plots", "performance"))
    if all_metrics:
        generate_complexity_scatter_plots(all_metrics, os.path.join(common_run_dir, "plots", "complexity"))



if __name__ == "__main__":
    main()
