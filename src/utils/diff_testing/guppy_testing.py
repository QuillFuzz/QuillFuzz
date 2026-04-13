from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import Counter
import random
import traceback

from guppylang import guppy
from guppylang import enable_experimental_features
from guppylang.std.quantum import *
from guppylang.std.qsystem import *
from guppylang.std.builtins import result, array
enable_experimental_features()
from selene_sim import build, Quest
from hugr.qsystem.result import QsysResult
from tket.passes import NormalizeGuppy, PytketHugrPass, PassResult
from hugr.hugr.base import Hugr
from pytket.circuit import Circuit
from pytket.passes import *
from pytket.passes import RemoveRedundancies, SquashRzPhasedX
from pytket.extensions.qiskit import AerBackend

from .base import Base


class guppyTesting(Base):
    def __init__(self):
        super().__init__()

    def ks_diff_test(self, circuit: Any, circuit_number: int, n_qubits: int = 10) -> None:
        hugr = None

        def compile_circuit():
            return circuit.compile()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(compile_circuit)
                hugr = future.result(timeout=self.TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            print(f"Compilation timed out after {self.TIMEOUT_SECONDS} seconds")
            self.save_interesting_circuit(circuit_number)
            return
        except Exception as e:
            if self._render_guppy_error(e):
                return
            print("Error during compilation:", e)
            print("Exception :", traceback.format_exc())
            self.save_interesting_circuit(circuit_number)
            return

        if hugr is None:
            return

        try:
            runner = build(hugr.to_bytes())
            results = QsysResult(runner.run_shots(Quest(), n_qubits=n_qubits, n_shots=1000))
            raw_counts = results.collated_counts()
            if not raw_counts:
                print(f"Warning: No counts collected for circuit {circuit_number}")
            elif len(raw_counts) > 0 and not list(raw_counts.keys())[0]:
                print(f"Warning: Empty keys in counts for circuit {circuit_number}. Measurements may not be returned/output.")

            counts_base = self._counts_from_qsys_raw(raw_counts)
            if not counts_base:
                print(f"Warning: No valid counts after preprocessing for circuit {circuit_number}. Skipping test.")
                return

            print(f"Uncompiled circuit{circuit_number} succesfully run.")
        except Exception as e:
            print(f"Error running uncompiled circuit: {e}")
            self.save_interesting_circuit(circuit_number)
            return

        pass_name = random.choice(["redundant_cx", "squash_rz", "normalize"])
        print(f"Applying pass: {pass_name}")

        try:
            main_function: Hugr = circuit.compile_function().modules[0]
            hugr_opt = None

            if pass_name == "redundant_cx":
                rr_pass = PytketHugrPass(RemoveRedundancies())
                pass_result: PassResult = rr_pass.run(main_function)
                hugr_opt = pass_result.hugr
            elif pass_name == "squash_rz":
                squash_pass = PytketHugrPass(SquashRzPhasedX())
                hugr_opt = squash_pass(main_function)
            elif pass_name == "normalize":
                normalize = NormalizeGuppy()
                hugr_opt = normalize(main_function)

            print(f"Pass {pass_name} applied successfully.")

            runner_opt = build(hugr_opt.to_bytes())
            results_opt = QsysResult(runner_opt.run_shots(Quest(), n_qubits=n_qubits, n_shots=1000))
            raw_counts_opt = results_opt.collated_counts()
            counts_opt = self._counts_from_qsys_raw(raw_counts_opt)

            if not counts_opt:
                print(f"Warning: No valid counts after preprocessing for optimized circuit {circuit_number}.")
                if counts_base:
                    print(f"Interesting discrepancy: Optimized circuit lost all outputs.")
                    self.save_interesting_circuit(circuit_number)
                return

            ks_value = self.ks_test(counts_base, counts_opt, 1000)
            print(f"Pass {pass_name} ks-test p-value: {ks_value}")

            if ks_value < self.KS_THRESHOLD:
                print(f"Interesting circuit found (Low KS): {circuit_number}")
                self.save_interesting_circuit(circuit_number)

            if self.plot:
                self.plot_histogram(counts_base, f"Guppy Base Results", 0, circuit_number)
                self.plot_histogram(counts_opt, f"Guppy {pass_name} Results", 1, circuit_number)

        except Exception as e:
            print(f"Error executing pass {pass_name} or running result: {e}")
            traceback.print_exc()
            self.save_interesting_circuit(circuit_number)
