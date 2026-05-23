from typing import Any, Optional, Callable, Dict, Tuple, List
from collections import Counter
import inspect
import random
import traceback
from numpy.typing import NDArray
import numpy as np

from pytket.circuit import Circuit
from pytket.passes import *
from pytket.extensions.qiskit import AerBackend, AerStateBackend
from guppylang import guppy, enable_experimental_features
from guppylang.std.quantum import *
from guppylang.std.qsystem import *
from guppylang.std.builtins import result
from selene_sim import build, Quest
from hugr.qsystem.result import QsysResult
from tket.passes import NormalizeGuppy, PytketHugrPass, PassResult
from hugr.hugr.base import Hugr

# Base testing class
from .base import Base
enable_experimental_features()


class pytketTesting(Base):
    def __init__(self):
        super().__init__()

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

    def _try_convert_pytket_circuit_to_hugr(self, circuit: Circuit) -> Tuple[Optional[Hugr], str]:
        attempts = []

        try:
            if hasattr(circuit, "to_hugr"):
                return circuit.to_hugr(), "Circuit.to_hugr"
            attempts.append("Circuit.to_hugr not available")
        except Exception as exc:
            attempts.append(f"Circuit.to_hugr failed: {exc}")

        try:
            from tket.interop.pytket import circuit_to_hugr  # type: ignore
            return circuit_to_hugr(circuit), "tket.interop.pytket.circuit_to_hugr"
        except Exception as exc:
            attempts.append(f"tket.interop.pytket.circuit_to_hugr unavailable/failed: {exc}")

        try:
            from tket.interop.pytket import to_hugr  # type: ignore
            return to_hugr(circuit), "tket.interop.pytket.to_hugr"
        except Exception as exc:
            attempts.append(f"tket.interop.pytket.to_hugr unavailable/failed: {exc}")

        return None, "; ".join(attempts)

    def ks_diff_test(
        self,
        circuit: Circuit,
        circuit_number: int,
        random_n: Optional[int] = None,
        shots: int = 10000,
    ) -> float:
        ks_value = 1.0
        backend = AerBackend()

        try:
            shots_local = self.shots(shots)
            baseline_circ = backend.get_compiled_circuit(circuit.copy(), optimisation_level=0)
            baseline_handle = backend.process_circuit(baseline_circ, n_shots=shots_local)
            baseline_result = backend.get_result(baseline_handle)
            counts1 = self.preprocess_counts(baseline_result.get_counts())

            selected_passes = self._get_selected_passes(random_n)
            if not selected_passes:
                print("No passes selected for ks_diff_test; returning baseline p-value 1.0")
                return ks_value

            for i, (pass_name, pass_factory) in enumerate(selected_passes):
                compiled_circ = circuit.copy()
                tket_pass = pass_factory()
                tket_pass.apply(compiled_circ)
                backend_circ = backend.get_compiled_circuit(compiled_circ, optimisation_level=0)
                handle2 = backend.process_circuit(backend_circ, n_shots=shots_local)
                result2 = backend.get_result(handle2)
                counts2 = self.preprocess_counts(result2.get_counts())

                ks_value = self.ks_test(counts1, counts2, shots_local)
                print(f"{pass_name} (#{i+1}) ks-test p-value: {ks_value}")

                if ks_value < self.KS_THRESHOLD:
                    print(f"Interesting circuit found: {circuit_number}")
                    self.save_interesting_circuit(circuit_number)

                if self.plot:
                    self.plot_histogram(counts1, "Uncompiled Circuit Results", 0, circuit_number)
                    self.plot_histogram(counts2, f"Tket_{pass_name}_Results", i + 1, circuit_number)

        except Exception as e:
            print("Error during pytket differential testing:", e)
            print("Exception :", traceback.format_exc())
            self.save_interesting_circuit(circuit_number)

        return ks_value

    def ks_diff_test_tket2(
        self,
        circuit: Any,
        circuit_number: int,
        random_n: Optional[int] = None,
        shots: int = 1000,
    ) -> float:
        ks_value = 1.0
        hugr_base: Optional[Hugr] = None
        conversion_path = ""

        try:
            if isinstance(circuit, Hugr):
                hugr_base = circuit
                conversion_path = "input is already Hugr"
            elif isinstance(circuit, Circuit):
                hugr_base, conversion_path = self._try_convert_pytket_circuit_to_hugr(circuit)
            elif hasattr(circuit, "compile_function"):
                hugr_base = circuit.compile_function().modules[0]
                conversion_path = "compile_function().modules[0]"

            if hugr_base is None:
                print("Could not prepare HUGR input for ks_diff_test_tket2.")
                print(f"Conversion attempts: {conversion_path or 'none'}")
                return ks_value

            print(f"ks_diff_test_tket2 using path: {conversion_path}")

            n_qubits = getattr(circuit, "n_qubits", 10)

            try:
                hugr_base = NormalizeGuppy()(hugr_base, inplace=False)
            except Exception:
                pass

            shots_local = self.shots(shots)
            runner = build(hugr_base.to_bytes())
            results = QsysResult(runner.run_shots(Quest(), n_qubits=n_qubits, n_shots=shots_local))
            counts_base = self._counts_from_qsys_raw(results.collated_counts())
            if not counts_base:
                print("No baseline counts generated for ks_diff_test_tket2.")
                return ks_value

            selected_passes = self._get_selected_passes(random_n)
            if not selected_passes:
                print("No passes selected for ks_diff_test_tket2; returning baseline p-value 1.0")
                return ks_value

            for i, (pass_name, pass_factory) in enumerate(selected_passes):
                try:
                    wrapped_pass = PytketHugrPass(pass_factory())
                    pass_result: PassResult = wrapped_pass.run(hugr_base, inplace=False)
                    hugr_opt = pass_result.hugr

                    runner_opt = build(hugr_opt.to_bytes())
                    results_opt = QsysResult(runner_opt.run_shots(Quest(), n_qubits=n_qubits, n_shots=shots_local))
                    counts_opt = self._counts_from_qsys_raw(results_opt.collated_counts())
                    if not counts_opt:
                        print(f"Skipping {pass_name}: no counts after optimization")
                        continue

                    ks_value = self.ks_test(counts_base, counts_opt, shots_local)
                    print(f"tket2:{pass_name} (#{i+1}) ks-test p-value: {ks_value}")

                    if ks_value < self.KS_THRESHOLD:
                        print(f"Interesting circuit found: {circuit_number}")
                        self.save_interesting_circuit(circuit_number)

                    if self.plot:
                        self.plot_histogram(counts_base, "Tket2 Base Results", 0, circuit_number)
                        self.plot_histogram(counts_opt, f"Tket2_{pass_name}_Results", i + 1, circuit_number)
                except Exception as pass_exc:
                    print(f"Skipping pass {pass_name} due to error: {pass_exc}")

        except Exception as e:
            print("Error during tket2 differential testing:", e)
            print("Exception :", traceback.format_exc())
            self.save_interesting_circuit(circuit_number)

        return ks_value

    def run_circ_statevector(self, circuit: Circuit, circuit_number: int) -> NDArray[np.complex128]:
        try:
            backend = AerStateBackend()
            uncompiled_circ = backend.get_compiled_circuit(circuit.copy(), optimisation_level=0)
            no_pass_statevector = uncompiled_circ.get_statevector()

            for i in range(3):
                compiled_circ = backend.get_compiled_circuit(circuit.copy(), optimisation_level=i + 1)
                pass_statevector = compiled_circ.get_statevector()
                dot_prod = self.compare_statevectors(no_pass_statevector, pass_statevector, 6)

                if dot_prod == 1:
                    print("Statevectors are the same\n")
                else:
                    print("Statevectors not the same")
                    print("Dot product: ", dot_prod)

        except Exception:
            print("Exception :", traceback.format_exc())

    # Convert pytket circuit to guppy, run in guppy, and compare results with pytket ran in Aer Simulator
    def run_guppy_pytket_diff(self, circuit: Circuit, circuit_number: int, qubit_defs_list: list[int], bit_defs_list: list[int]) -> None:
        pytket_circ_copy = circuit.copy()
        from pytket import OpType
        guppy_gateset = {OpType.CX, OpType.CZ, OpType.CY, OpType.X, OpType.Y, OpType.Z, OpType.H, OpType.T,
                         OpType.Tdg, OpType.Rx, OpType.Ry, OpType.Rz, OpType.S, OpType.Sdg, OpType.CCX,
                         OpType.V, OpType.Vdg, OpType.CRz}
        DecomposeBoxes().apply(circuit)
        AutoRebase(guppy_gateset).apply(circuit)

        guppy_circuit = guppy.load_pytket("guppy_circuit", circuit)

        qubit_defs_list_sorted = [x[1] for x in sorted(enumerate(qubit_defs_list), key=lambda x: (x[1] == 0, x[0]))]
        bit_defs_list_sorted = [x for x in bit_defs_list if x == 0] + [x for x in bit_defs_list if x != 0]
        if len(bit_defs_list_sorted) == 1 and bit_defs_list_sorted[0] != 0:
            bit_defs_list_sorted = [1] * bit_defs_list_sorted[0]

        @guppy.comptime
        def main() -> None:
            qubit_variables = []
            for qubit_def in qubit_defs_list_sorted:
                qubit_array = [qubit() for _ in range(qubit_def)] if qubit_def > 0 else [qubit()]
                qubit_variables.append(qubit_array)
            creg_results = guppy_circuit(*qubit_variables)

            for i in range(len(bit_defs_list_sorted)):
                if bit_defs_list_sorted[i] == 0:
                    result(f"b{i}", creg_results[i])
            for r in range(len(qubit_variables)):
                if isinstance(qubit_variables[r], list):
                    result(f"q{r}", measure_array(qubit_variables[r]))
            if creg_results is not None and hasattr(creg_results, '__len__'):
                for i in range(len(bit_defs_list_sorted)):
                    if bit_defs_list_sorted[i] != 0:
                        result(f"creg{i}", creg_results[i])

        try:
            compiled_circ = main.compile()
            shots_local = self.shots(10000)
            runner = build(compiled_circ)
            results = QsysResult(runner.run_shots(Quest(), n_qubits=circuit.n_qubits, n_shots=shots_local))
            counts_guppy = results.collated_counts()
            counts_guppy = Counter({''.join([measurement[1] for measurement in key]): value for key, value in counts_guppy.items()})
            counts_guppy = self.preprocess_counts(counts_guppy)
            print("Processed guppy counts:", counts_guppy)

            backend = AerBackend()
            pytket_circ_copy.measure_all()
            uncompiled_pytket_circ = backend.get_compiled_circuit(pytket_circ_copy, optimisation_level=0)
            handle = backend.process_circuit(uncompiled_pytket_circ, n_shots=shots_local)
            result_pytket = backend.get_result(handle)
            counts_pytket = self.preprocess_counts(result_pytket.get_counts())
            print("Pytket counts:", counts_pytket)

            ks_value = self.ks_test(counts_guppy, counts_pytket, shots_local)
            print(f"Guppy vs Pytket ks-test p-value: {ks_value}")

            if ks_value < self.KS_THRESHOLD:
                print(f"Interesting circuit found: {circuit_number}")
                self.save_interesting_circuit(circuit_number)

            if self.plot:
                self.plot_histogram(counts_guppy, "Guppy Circuit Results", 0, circuit_number)
                self.plot_histogram(counts_pytket, "Pytket Circuit Results", 0, circuit_number)

        except Exception as e:
            if self._render_guppy_error(e):
                pass
            else:
                print("Error during compilation:", e)
                print("Exception :", traceback.format_exc())
