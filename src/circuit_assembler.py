import argparse
import glob
import logging
import math
import os
import random
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble and test quantum circuits.")
    parser.add_argument("input_dir", help="Directory containing input circuit files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save assembled circuits",
    )
    parser.add_argument("--n-generations", type=int, default=1, help="Number of circuits to generate")
    parser.add_argument("--min-files", type=int, default=2, help="Minimum number of files per assembly")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum number of files per assembly")
    parser.add_argument(
        "--language",
        required=True,
        choices=["guppy", "qiskit"],
        help="Language of circuits (guppy or qiskit)",
    )
    return parser.parse_args()


def setup_logging(logfile_path: str) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def permutation_count(n_items: int, k_items: int) -> int:
    if hasattr(math, "perm"):
        return math.perm(n_items, k_items)
    return math.factorial(n_items) // math.factorial(n_items - k_items)


def max_unique_combinations(n_files: int, min_files: int, max_files: int) -> int:
    return sum(permutation_count(n_files, k) for k in range(min_files, max_files + 1))


def main() -> int:
    args = parse_args()

    from utils.circuit_assembler import assemble

    if args.n_generations < 1:
        print("--n-generations must be >= 1")
        return 1

    if args.min_files < 1:
        print("--min-files must be >= 1")
        return 1

    if args.max_files < args.min_files:
        print("--max-files must be >= --min-files")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    logfile_path = os.path.join(args.output_dir, "assembler.log")
    setup_logging(logfile_path)

    logging.info(
        "Starting assembler with configurations: "
        "Input=%s, Output=%s, N=%s, Language=%s",
        args.input_dir,
        args.output_dir,
        args.n_generations,
        args.language,
    )

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.py")))
    if not input_files:
        msg = f"No input files found in {args.input_dir}"
        logging.error(msg)
        print(msg)
        return 1

    effective_max_files = min(args.max_files, len(input_files))
    if args.min_files > len(input_files):
        msg = (
            f"--min-files ({args.min_files}) is greater than available input files "
            f"({len(input_files)})"
        )
        logging.error(msg)
        print(msg)
        return 1

    max_possible = max_unique_combinations(len(input_files), args.min_files, effective_max_files)
    target_generations = min(args.n_generations, max_possible)
    if target_generations < args.n_generations:
        logging.warning(
            "Requested %s generations, but only %s unique ordered combinations are possible. "
            "Capping generations to %s.",
            args.n_generations,
            max_possible,
            target_generations,
        )

    logging.info("Found %s input files. Starting assembly...", len(input_files))
    print(f"Found {len(input_files)} input files. Starting assembly...")

    seen_combinations = set()
    generated_count = 0
    failed_count = 0
    attempts_without_progress = 0
    max_attempts_without_progress = max(1000, target_generations * 25)

    with tqdm(total=target_generations, desc="Assembling") as pbar:
        while generated_count < target_generations and attempts_without_progress < max_attempts_without_progress:
            k = random.randint(args.min_files, effective_max_files)
            selected_files = random.sample(input_files, k)
            combo_key = tuple(selected_files)

            if combo_key in seen_combinations:
                attempts_without_progress += 1
                continue

            seen_combinations.add(combo_key)
            output_file = os.path.join(args.output_dir, f"assembled_circuit_{generated_count}.py")

            try:
                assemble(selected_files, output_file, generated_count, language=args.language)
                generated_count += 1
                attempts_without_progress = 0
                pbar.update(1)
            except Exception as exc:
                failed_count += 1
                attempts_without_progress += 1
                logging.error("Failed to assemble combination %s: %s", selected_files, exc)

    if generated_count < target_generations:
        logging.warning(
            "Stopped early after %s generations due to repeated duplicate or failed attempts.",
            generated_count,
        )

    logging.info(
        "Assembly complete. Generated=%s, Failed=%s, Requested=%s",
        generated_count,
        failed_count,
        args.n_generations,
    )
    return 0 if generated_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
