from pytket.circuit import Circuit
from qiskit import transpile
import traceback

from .base import Base


class qiskitTesting(Base):
    def __init__(self):
        super().__init__()

    def ks_diff_test(self, circuit: Circuit, circuit_number: int) -> float:
        ks_value = 1.0
        try:
            from qiskit_aer import AerSimulator
            backend = AerSimulator()

            circuit.measure_all()
            uncompiled_circ = transpile(circuit, backend, optimization_level=0)
            counts1 = self.preprocess_counts(backend.run(uncompiled_circ, shots=10000).result().get_counts())

            for i in range(3):
                compiled_circ = transpile(circuit, backend, optimization_level=i + 1)
                counts2 = self.preprocess_counts(backend.run(compiled_circ, shots=10000).result().get_counts())

                ks_value = self.ks_test(counts1, counts2, 10000)
                print(f"Optimisation level {i+1} ks-test p-value: {ks_value}")

                if ks_value < self.KS_THRESHOLD:
                    print(f"Interesting circuit found: {circuit_number}")
                    self.save_interesting_circuit(circuit_number)

                if self.plot:
                    self.plot_histogram(counts1, "Uncompiled Circuit Results", 0, circuit_number)
                    self.plot_histogram(counts2, "Compiled Circuit Results", i + 1, circuit_number)

        except Exception as e:
            print("Error during qiskit differential testing:", e)
            print("Exception :", traceback.format_exc())
            self.save_interesting_circuit(circuit_number)

        return ks_value
