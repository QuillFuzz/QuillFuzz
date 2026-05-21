from typing import Tuple, Any, Optional
from collections import Counter
from itertools import zip_longest
from scipy.stats import ks_2samp
from numpy.typing import NDArray
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import traceback
import pathlib
import shutil


class Base():
    OUTPUT_DIR = (pathlib.Path(__file__).parent.parent.parent / "local_saved_circuits").resolve()
    TIMEOUT_SECONDS = 200
    KS_THRESHOLD = 0.01

    def __init__(self):
        super().__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--plot', action='store_true', help='Plot results after running circuit')
        self.args = self.parser.parse_args()
        self.plot: bool = self.args.plot
        self.run_output_dir: pathlib.Path = self._resolve_run_output_dir()

    def _resolve_run_output_dir(self) -> pathlib.Path:
        env_run_dir = os.getenv("QUILLFUZZ_RUN_DIR")
        if env_run_dir:
            resolved_env_dir = pathlib.Path(env_run_dir).expanduser().resolve()
            resolved_env_dir.mkdir(parents=True, exist_ok=True)
            return resolved_env_dir

        output_dir = pathlib.Path(self.OUTPUT_DIR).resolve()

        if output_dir.name.startswith("Complete_run"):
            return output_dir

        try:
            complete_run_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("Complete_run")]
            if complete_run_dirs:
                return max(complete_run_dirs, key=lambda p: p.stat().st_mtime)
        except Exception:
            pass

        return output_dir

    def get_interesting_circuits_dir(self) -> pathlib.Path:
        return self.run_output_dir / "interesting_circuits"

    def _ensure_qnexus_login(self) -> None:
        if not self.qnexus_check_login_status():
            self.qnexus_login()

    def _render_guppy_error(self, error: Exception) -> bool:
        try:
            from guppylang_internals.error import GuppyError
            if isinstance(error, GuppyError):
                from guppylang_internals.diagnostic import DiagnosticsRenderer
                from guppylang_internals.engine import DEF_STORE

                renderer = DiagnosticsRenderer(DEF_STORE.sources)
                renderer.render_diagnostic(error.error)
                sys.stderr.write("\n".join(renderer.buffer))
                sys.stderr.write("\n\nGuppy compilation failed due to 1 previous error\n")
                return True
        except Exception:
            return False

        return False

    def _counts_from_qsys_raw(self, raw_counts: Counter[Tuple[Any, ...], int]) -> Counter[int, int]:
        counts = Counter()
        for key, value in raw_counts.items():
            pieces = []
            for measurement in key:
                try:
                    piece = measurement[1]
                except Exception:
                    piece = measurement
                pieces.append(str(piece))
            joined = ''.join(pieces)
            counts[joined] = value
        return self.preprocess_counts(counts)

    def preprocess_counts(self, counts: Counter[Tuple[str, ...], int]) -> Counter[int, int]:
        out: Counter[int, int] = {}
        for k in counts.keys():
            if isinstance(k, tuple):
                key_str = ''.join(str(x) for x in k).replace(' ', '')
            else:
                key_str = str(k).replace(' ', '')

            if not key_str:
                print(f"Warning: Empty key string encounterd in counts: {k}. Skipping.")
                continue

            try:
                out[int(key_str, 2)] = counts[k]
            except ValueError as e:
                print(f"Error processing count key: '{k}' -> '{key_str}'. Error: {e}")
                raise e

        return dict(sorted(out.items()))

    def ks_test(self, counts1: Counter[int, int], counts2: Counter[int, int], total_shots: int) -> float:
        sample1, sample2 = [], []

        for p1, p2 in zip_longest(counts1.items(), counts2.items(), fillvalue=None):
            if p1:
                sample1 += p1[1] * [p1[0]]
            if p2:
                sample2 += p2[1] * [p2[0]]

        assert (len(sample1) == total_shots) and (len(sample2) == total_shots), "Sample size does not match number of shots"

        ks_stat, p_value = ks_2samp(sorted(sample1), sorted(sample2), method='asymp')
        return p_value

    def compare_statevectors(self, sv1: NDArray[np.complex128], sv2: NDArray[np.complex128], precision: int = 6) -> float:
        return np.round(abs(np.vdot(sv1, sv2)), precision)

    def save_interesting_circuit(self, circuit_number: int, interesting_dir: Optional[pathlib.Path] = None) -> None:
        if interesting_dir is None:
            interesting_dir = self.get_interesting_circuits_dir()

        interesting_dir.mkdir(parents=True, exist_ok=True)

        circuit_source_path = None
        source_file_env = os.getenv("QUILLFUZZ_SOURCE_FILE")
        if source_file_env:
            candidate = pathlib.Path(source_file_env).expanduser().resolve()
            if candidate.exists() and candidate.suffix == ".py":
                circuit_source_path = candidate

        if circuit_source_path is not None:
            base_name = circuit_source_path.name
        else:
            base_name = f"circuit{circuit_number}.py"

        circuit_dest_path = interesting_dir / base_name
        if circuit_dest_path.exists() and circuit_source_path is not None and circuit_dest_path.resolve() != circuit_source_path.resolve():
            print(f"Info: File {circuit_dest_path} has already been flagged as interesting. Skipping")
            return

        if circuit_source_path is not None and circuit_source_path.exists():
            try:
                shutil.copy2(circuit_source_path, circuit_dest_path)
                print(f"Interesting circuit saved to: {circuit_dest_path}")
            except Exception as e:
                print(f"Error copying circuit file: {e}")
        else:
            print(
                "Warning: Circuit source file could not be resolved. "
                f"Set QUILLFUZZ_SOURCE_FILE to the tested program path. Current value: {source_file_env}"
            )

    def plot_histogram(self, res: Counter[int, int], title: str, compilation_level: int, circuit_number: int = 0):
        plots_dir = self.OUTPUT_DIR / f"circuit{circuit_number}"
        if not plots_dir.exists():
            plots_dir.mkdir(parents=True, exist_ok=True)

        plots_path = plots_dir / f"output{circuit_number}_{title}{compilation_level if compilation_level else 'uncompiled'}.png"
        values = list(res.keys())
        freqs = list(res.values())

        bar_width = 0.5
        plt.bar(values, freqs, width=bar_width, edgecolor='black')
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
        plt.xlabel("Possible results")
        plt.ylabel("Number of occurances")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plots_path)
        plt.close()
