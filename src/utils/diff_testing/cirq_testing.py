from collections import Counter
from typing import Any, Iterable

import cirq

from .base import Base


class cirqTesting(Base):
    def __init__(self):
        super().__init__()

    def _extract_measurement_counts(self, result: cirq.Result) -> Counter[str]:
        counts: Counter[str] = Counter()
        measurements = result.measurements
        if not measurements:
            return counts

        measurement_keys = sorted(measurements.keys())
        sample_count = len(measurements[measurement_keys[0]])

        for sample_index in range(sample_count):
            bitstring_parts = []
            for measurement_key in measurement_keys:
                sample = measurements[measurement_key][sample_index]
                bitstring_parts.append(''.join(str(int(bit)) for bit in sample))
            counts[''.join(bitstring_parts)] += 1
        return counts

    def _counts_for_circuit(self, circuit: cirq.Circuit, shots: int) -> Counter[int]:
        simulator = cirq.Simulator()
        measured_circuit = circuit.copy()
        measurement_keys = sorted(measured_circuit.all_measurement_key_names())
        if not measurement_keys:
            measured_circuit.append(cirq.measure(*sorted(measured_circuit.all_qubits()), key='m'))

        result = simulator.run(measured_circuit, repetitions=shots)
        counts = self._extract_measurement_counts(result)
        return self.preprocess_counts(counts)

    def _transforms(self) -> Iterable[tuple[str, Any]]:
        return (
            ('drop_empty_moments', cirq.drop_empty_moments),
            ('eject_z', cirq.eject_z),
            ('merge_single_qubit_gates_to_phxz', cirq.merge_single_qubit_gates_to_phxz),
        )

    def ks_diff_test(self, circuit: cirq.Circuit, circuit_number: int) -> float:
        ks_value = 1.0

        try:
            if circuit is None:
                print('No Cirq circuit provided.')
                return ks_value

            shots_local = self.shots(1000)
            baseline_counts = self._counts_for_circuit(circuit, shots_local)

            for transform_name, transform in self._transforms():
                transformed = transform(circuit.copy())
                transformed_counts = self._counts_for_circuit(transformed, shots_local)
                ks_value = self.ks_test(baseline_counts, transformed_counts, shots_local)
                print(f'Cirq transform {transform_name} ks-test p-value: {ks_value}')

                if ks_value < self.KS_THRESHOLD:
                    print(f'Interesting circuit found: {circuit_number}')
                    self.save_interesting_circuit(circuit_number)

                if self.plot:
                    self.plot_histogram(baseline_counts, 'Cirq Baseline Results', 0, circuit_number)
                    self.plot_histogram(transformed_counts, f'Cirq {transform_name} Results', 1, circuit_number)

        except Exception as error:
            print('Error during cirq differential testing:', error)
            self.save_interesting_circuit(circuit_number)

        return ks_value
