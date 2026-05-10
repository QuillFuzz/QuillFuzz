from typing import Any, Optional, Callable, Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import copy
import random
import traceback

from guppylang import enable_experimental_features
enable_experimental_features()
from guppylang.std.quantum import *
from guppylang.std.qsystem import *
from selene_sim import build, Quest
from hugr.qsystem.result import QsysResult
from tket.passes import NormalizeGuppy, PytketHugrPass, PassResult
from hugr.hugr.base import Hugr
from pytket.passes import *

from .base import Base


class guppyTesting(Base):
    
    # Update this for list of passes to test
    def _curated_pass_registry(self) -> Dict[str, Callable[[], BasePass]]:
        return {
            "RemoveRedundancies": RemoveRedundancies,
            "SquashRzPhasedX": SquashRzPhasedX,
            "CliffordSimp": CliffordSimp,
            "CommuteThroughMultis": CommuteThroughMultis,
            "PeepholeOptimise2Q": PeepholeOptimise2Q,
            "FullPeepholeOptimise": FullPeepholeOptimise,
            "SynthesiseTket": SynthesiseTket,
            "RemoveBarriers": RemoveBarriers,
            "RemovePhaseOps": RemovePhaseOps,
            "SquashTK1": SquashTK1,
            "DecomposeMultiQubitsCX": DecomposeMultiQubitsCX,
            "RebaseTket": RebaseTket,
            "FlattenRegisters": FlattenRegisters,
        }
    
    def _get_selected_passes(self, random_n: Optional[int]) -> List[Tuple[str, Callable[[], BasePass]]]:
        
        curated = self._curated_pass_registry()
        selected = dict(curated)

        if random_n is not None:
            selected = dict(random.sample(list(selected.items()), min(random_n, len(selected))))

        return list(selected.items())

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
                print(
                    f"Warning: Empty keys in counts for circuit {circuit_number}. Measurements may not be returned/output."
                )

            counts_base = self._counts_from_qsys_raw(raw_counts)
            if not counts_base:
                print(f"Warning: No valid counts after preprocessing for circuit {circuit_number}. Skipping test.")
                return

            print(f"Uncompiled circuit {circuit_number} successfully run.")
        except Exception as e:
            print(f"Error running uncompiled circuit: {e}")
            self.save_interesting_circuit(circuit_number)
            return

        # Now compile with pytket/tket passes, but first normalize Hugr
        try:
            print("Normalizing Hugr for pass application...")
            normalize = NormalizeGuppy()
            main_function: Hugr = circuit.compile_function().modules[0]
            hugr_base = normalize(main_function)
        except Exception as e:
            print(f"Error during Hugr normalization: {e}")
            return

        def apply_pass_with_fallback(pass_name: str, pass_factory: Callable[[], Any], source_hugr: Hugr) -> Hugr:
            pass_instance = pass_factory()
            pass_input = copy.deepcopy(source_hugr)

            try:
                wrapped_pass = PytketHugrPass(pass_instance)
                pass_result: PassResult = wrapped_pass.run(pass_input)
                if pass_result.hugr is None:
                    raise RuntimeError(f"{pass_name} produced no HUGR result via PytketHugrPass")
                return pass_result.hugr
            except Exception:
                try:
                    wrapped_pass = PytketHugrPass(pass_instance)
                    pass_result = wrapped_pass(pass_input)
                except Exception as e:
                    raise RuntimeError(f"Error applying pass {pass_name} : {e}")
                    
        # Now run all the curated passes
        selected_passes = self._get_selected_passes(random_n=None)
        if not selected_passes:
            print("No passes selected for guppy ks_diff_test; returning after baseline run.")
            return
        for pass_name, pass_factory in selected_passes:
            print(f"Applying pass: {pass_name}")

            try:
                hugr_opt = apply_pass_with_fallback(pass_name, pass_factory, hugr_base)

                runner_opt = build(hugr_opt.to_bytes())
                results_opt = QsysResult(runner_opt.run_shots(Quest(), n_qubits=n_qubits, n_shots=1000))
                raw_counts_opt = results_opt.collated_counts()
                counts_opt = self._counts_from_qsys_raw(raw_counts_opt)

                if not counts_opt:
                    print(f"Warning: No valid counts after preprocessing for optimized circuit {circuit_number}.")
                    if counts_base:
                        print("Interesting discrepancy: Optimized circuit lost all outputs.")
                        self.save_interesting_circuit(circuit_number)
                    continue

                ks_value = self.ks_test(counts_base, counts_opt, 1000)
                print(f"Pass {pass_name} ks-test p-value: {ks_value}")

                if ks_value < self.KS_THRESHOLD:
                    print(f"Interesting circuit found (Low KS): {circuit_number}")
                    self.save_interesting_circuit(circuit_number)

                if self.plot:
                    self.plot_histogram(counts_base, "Guppy Base Results", 0, circuit_number)
                    self.plot_histogram(counts_opt, f"Guppy {pass_name} Results", 1, circuit_number)

            except Exception as e:
                print(f"Error executing pass {pass_name} or running result: {e}")
                traceback.print_exc()
                self.save_interesting_circuit(circuit_number)
                continue
