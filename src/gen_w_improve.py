import argparse
import json
import os
import sys
import time
import yaml

# Add project root to path so we can import scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gen_workflow import assemble_circuits, run_mutation_phase, run_production_phase, run_training_phase
from utils.execution import (
    build_hourly_coverage_timeline,
    combine_coverage_artifacts,
    format_hourly_coverage_points,
    generate_coverage_report_from_data_file,
    load_compact_coverage_summary,
)
from utils.interactive_stop import GracefulStopController
from utils.reporting import Logger
from utils.utils import generate_complexity_scatter_plots, generate_summary_plot, sanitize_model_name


SUPPORTED_LANGUAGES = ("guppy", "qiskit", "cirq", "pytket", "pennylane")


def main():
    run_start_epoch = time.time()

    # Parsing arguments and cleaning them up with warnings for invalid values, as well as handling config file loading and default paths

    parser = argparse.ArgumentParser(description="LLM Circuit Generator")
    parser.add_argument("--config_file", type=str, help="Relative path to the configuration file.")
    parser.add_argument("--run_name", type=str, help="Name for the current run. Defaults to a timestamp.")
    parser.add_argument("--language", type=str, choices=SUPPORTED_LANGUAGES, help="Language for the generated code.")
    parser.add_argument("--output_dir", type=str, help="Directory for the current run. Defaults to a timestamped folder inside 'local_saved_circuits'.")
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
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (reduces diff_testing shots)")
    parser.add_argument("--simple_mode", action="store_true", default=False, help="Enable simple mode to use simple generation, fixing, and mutation prompts.")
    parser.add_argument(
        "--ks_low_threshold",
        type=float,
        default=0.01,
        help="Threshold below which KS-test p-values are flagged as low in production reports.",
    )
    

    args, _ = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            parser.set_defaults(**(yaml.safe_load(f) or {}))

    args = parser.parse_args()
    
    args.language = args.language if args.language else None
    if args.language:
        args.language = args.language.lower()
    if args.language not in SUPPORTED_LANGUAGES:
        parser.error(f"Unsupported language '{args.language}'. Expected one of: {', '.join(SUPPORTED_LANGUAGES)}")

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
        if args.simple_mode:
            args.prompt_dir = os.path.join(project_root, "prompts", f"{args.language}_simple")
        else:
            args.prompt_dir = os.path.join(project_root, "prompts", args.language)
    run_id = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    if not args.output_dir:
        args.output_dir = os.path.join(project_root, "local_saved_circuits", run_id)

    common_run_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(common_run_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create run directory {common_run_dir}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)

    # Ensure all execution subprocesses save artifacts (e.g., interesting circuits)
    # into this run directory rather than falling back to a default path.
    os.environ["QUILLFUZZ_RUN_DIR"] = common_run_dir

    coverage_artifacts_dir = os.path.join(common_run_dir, "coverage_artifacts")
    os.makedirs(coverage_artifacts_dir, exist_ok=True)
    os.environ["QUILLFUZZ_COVERAGE_ARTIFACT_DIR"] = coverage_artifacts_dir

    # Debug environment variable to enable debug mode in subprocesses (diff_testing test harness)
    os.environ["QUILLFUZZ_DEBUG"] = "1" if args.debug else "0"

    # ======================
    # Start of main workflow
    # ======================
    logfile_path = os.path.join(common_run_dir, "execution.log")

    all_stats = []
    all_reports = []
    main_logger = Logger(logfile_path)

    stop_controller = GracefulStopController(enabled=True, logger=main_logger)
    args.stop_controller = stop_controller
    stop_controller.start()

    assembled_all_stats = []
    assembled_all_reports = []
    mutation_all_stats = []
    mutation_all_metrics = []
    generated_metrics_by_model = {}
    assembled_metrics_by_model = {}

    try:
        for model in args.models:
            if stop_controller.stop_requested:
                main_logger.log("Skipping remaining models due to interactive stop request.")
                break

            if args.improve_prompt:
                best_prompt = run_training_phase(model, args, common_run_dir, logfile_path)
            else:
                best_prompt = "generation_prompt.txt"

            if stop_controller.stop_requested:
                main_logger.log("Stop requested after training phase; skipping production for remaining models.")
                break

            files, summary, metrics, report_entries = run_production_phase(
                model, best_prompt, args, common_run_dir, logfile_path, main_logger
            )
            all_stats.append(summary)
            generated_metrics_by_model[model] = metrics
            all_reports.append({"model": model, "entries": report_entries})

            if stop_controller.stop_requested:
                main_logger.log("Stop requested after production phase; skipping mutation/assembly.")
                break

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

            if stop_controller.stop_requested:
                main_logger.log("Stop requested after mutation phase; skipping assembly.")
                break

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

            if assembly_files:
                assembly_logfile = os.path.join(common_run_dir, "assembly_execution.log")
                assembly_logger = Logger(assembly_logfile)
                assembled_files, assembled_metrics, assembled_reports = assemble_circuits(
                    model, assembly_files, args, common_run_dir, assembly_logger
                )

                if assembled_metrics:
                    assembled_metrics_by_model[model] = assembled_metrics
                    assembled_all_stats.append({
                        "model": model,
                        "total_cost": 0.0,
                        "total_time": 0.0,
                        "total_programs": len(assembled_metrics),
                        "valid_programs": sum(1 for m in assembled_metrics if m.get("success")),
                        "avg_quality_score": 0.0,
                    })
                if assembled_reports:
                    assembled_all_reports.append({"model": f"{model}::assembled", "entries": assembled_reports})

                if stop_controller.stop_requested:
                    main_logger.log("Stop requested during assembly; skipping remaining models.")
                    break
    finally:
        stop_controller.close()

    if all_stats:
        generate_summary_plot(all_stats, os.path.join(common_run_dir, "plots", "performance"))

    generated_complexity_root = os.path.join(common_run_dir, "plots", "complexity")
    for model, model_metrics in generated_metrics_by_model.items():
        if model_metrics:
            generate_complexity_scatter_plots(
                model_metrics,
                os.path.join(generated_complexity_root, sanitize_model_name(model)),
            )

    if mutation_all_stats:
        generate_summary_plot(mutation_all_stats, os.path.join(common_run_dir, "plots", "mutation_performance"))
    if mutation_all_metrics:
        generate_complexity_scatter_plots(mutation_all_metrics, os.path.join(common_run_dir, "plots", "mutation_complexity"))

    if assembled_all_stats:
        generate_summary_plot(assembled_all_stats, os.path.join(common_run_dir, "assembled_plots", "performance"))

    assembled_complexity_root = os.path.join(common_run_dir, "assembled_plots", "complexity")
    for model, model_metrics in assembled_metrics_by_model.items():
        if model_metrics:
            generate_complexity_scatter_plots(
                model_metrics,
                os.path.join(assembled_complexity_root, sanitize_model_name(model)),
            )

    combined_coverage_file = None
    coverage_summary_path = None
    coverage_summary_compact = {}
    coverage_hourly_points = []
    coverage_hourly_point_text = []
    coverage_combine_error = ""
    run_end_epoch = time.time()
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
        main_logger.log("Coverage Hourly Timeline:")
        for line in coverage_hourly_point_text:
            main_logger.log(f"  {line}")

    performance_summary_path = os.path.join(common_run_dir, "performance_summary.json")
    performance_summary = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": args.language,
        "models": args.models,
        "coverage_artifacts_dir": coverage_artifacts_dir,
        "coverage_combined_data_file": combined_coverage_file,
        "coverage_summary_path": coverage_summary_path,
        "coverage_summary_compact": coverage_summary_compact,
        "coverage_hourly_points": coverage_hourly_points,
        "coverage_hourly_points_text": coverage_hourly_point_text,
        "coverage_combine_error": coverage_combine_error,
        "model_summaries": all_stats,
        "per_file_reports": all_reports,
    }
    with open(performance_summary_path, "w", encoding="utf-8") as f:
        json.dump(performance_summary, f, indent=2)
    print(f"Performance summary JSON: {performance_summary_path}")

    assembled_summary_path = os.path.join(common_run_dir, "assembled_performance_summary.json")
    assembled_summary = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": args.language,
        "models": args.models,
        "coverage_artifacts_dir": coverage_artifacts_dir,
        "coverage_combined_data_file": combined_coverage_file,
        "coverage_summary_path": coverage_summary_path,
        "coverage_summary_compact": coverage_summary_compact,
        "coverage_hourly_points": coverage_hourly_points,
        "coverage_hourly_points_text": coverage_hourly_point_text,
        "coverage_combine_error": coverage_combine_error,
        "model_summaries": assembled_all_stats,
        "per_file_reports": assembled_all_reports,
    }
    with open(assembled_summary_path, "w", encoding="utf-8") as f:
        json.dump(assembled_summary, f, indent=2)
    print(f"Assembled performance summary JSON: {assembled_summary_path}")


if __name__ == "__main__":
    main()
