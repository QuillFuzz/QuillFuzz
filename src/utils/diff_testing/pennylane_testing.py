from __future__ import annotations
import inspect
from collections import Counter
from typing import Any
import numpy as np
import pennylane as qml
from .base import Base


class pennylaneTesting(Base):
    def __init__(self):
        super().__init__()

    def _stable_code(self, value: Any) -> int:
        text = repr(value)
        total = 0
        for index, character in enumerate(text):
            total += (index + 1) * ord(character)
        return total % 10_000

    def _build_call_args(self, circuit: Any) -> list[Any]:
        try:
            signature = inspect.signature(circuit)
        except Exception:
            return []

        call_args: list[Any] = []
        for index, parameter in enumerate(signature.parameters.values()):
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if parameter.default is not inspect._empty:
                call_args.append(parameter.default)
            else:
                # Generate parameter arrays instead of scalars to support generated code
                # that tries to subscript parameters like params[i]. Using a 4-element
                # array to match typical quantum circuit parameter counts.
                param_value = 0.125 * (index + 1)
                call_args.append(np.array([param_value, param_value * 2, param_value * 3, param_value * 4]))
        return call_args

    def _call_circuit(self, circuit: Any, seed: int) -> Any:
        if circuit is None:
            raise ValueError("PennyLane circuit main() returned None")

        if isinstance(circuit, (list, tuple)):
            for item in circuit:
                if callable(item):
                    circuit = item
                    break
            else:
                circuit = circuit[0] if circuit else circuit

        np.random.seed(seed)
        call_args = self._build_call_args(circuit)
        try:
            return circuit(*call_args)
        except ZeroDivisionError as e:
            # Help diagnose modulo-by-zero in generated code
            raise ZeroDivisionError(
                f"Modulo by zero in circuit execution. This typically occurs when generated code uses "
                f"`index % len(param)` but len(param) is 0, or divides by a parameter that is 0. "
                f"Parameters passed: {call_args}. Original error: {e}"
            ) from e
        except Exception as e:
            # Re-raise with circuit call context
            raise type(e)(
                f"Error executing circuit with parameters {[type(arg).__name__ for arg in call_args]}: {e}"
            ) from e

    def _flatten_numeric_output(self, value: Any) -> list[float]:
        if isinstance(value, dict):
            flattened: list[float] = []
            for key, count in value.items():
                flattened.extend([float(self._stable_code(key))] * int(count))
            return flattened

        if isinstance(value, (list, tuple)):
            flattened: list[float] = []
            for item in value:
                flattened.extend(self._flatten_numeric_output(item))
            return flattened

        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [float(value.item())]
            if value.dtype.kind in {"i", "u", "b"}:
                return [float(item) for item in value.reshape(-1).tolist()]

            # Handle complex arrays (e.g., statevectors) by taking magnitude
            if value.dtype.kind == 'c':
                flat = np.abs(value).reshape(-1)
            else:
                flat = value.astype(float).reshape(-1)
            
            # Check if this is a valid probability distribution
            flat_sum = float(flat.sum()) if flat.size > 0 else 0.0
            if flat.size > 0 and np.all(flat >= 0.0) and np.isclose(flat_sum, 1.0, atol=1e-3) and flat_sum > 0.0:
                rng = np.random.default_rng(0)
                # Normalize to ensure valid probabilities
                normalized = flat / flat_sum
                choices = rng.choice(np.arange(flat.size), size=max(1, flat.size * 16), p=normalized)
                return [float(choice) for choice in choices.tolist()]
            return [float(item) for item in flat.tolist()]

        if isinstance(value, (np.integer, int, np.bool_)):
            return [float(value)]

        if isinstance(value, (np.floating, float)):
            return [float(value)]

        return [float(self._stable_code(value))]

    def _counts_from_output(self, output: Any, seed: int) -> Counter[int]:
        flattened = self._flatten_numeric_output(output)
        if not flattened:
            return Counter({0: 1})

        counts: Counter[int] = Counter()
        for item in flattened:
            if abs(item) <= 1.0:
                counts[int(round(item * 1000))] += 1
            else:
                counts[int(round(item))] += 1

        if not counts:
            counts[int(seed)] += 1
        return counts

    def _create_alt_device_circuit(self, circuit: Any) -> Any:
        """
        Create an alternative QNode using a different device or execution mode.
        This enables differential testing: comparing execution on different backends
        or with different decomposition strategies.
        """
        if not callable(circuit):
            return circuit

        try:
            qnode = circuit
            circuit_func = None
            original_device = None

            # Try multiple strategies to extract circuit function and device
            # Strategy 1: Check for .func and .device attributes (standard QNode)
            if hasattr(qnode, 'func') and hasattr(qnode, 'device'):
                circuit_func = qnode.func
                original_device = qnode.device
            # Strategy 2: Check for _circuit attribute (some wrapped versions)
            elif hasattr(qnode, '_circuit'):
                circuit_func = qnode._circuit
                if hasattr(qnode, '_device'):
                    original_device = qnode._device
            # Strategy 3: Try calling it and see if it's already a QNode-like callable
            elif callable(qnode) and hasattr(qnode, '__self__'):
                # Bound method or similar; try to extract circuit
                if hasattr(qnode, '__self__') and hasattr(qnode.__self__, 'func'):
                    circuit_func = qnode.__self__.func
                    original_device = qnode.__self__.device if hasattr(qnode.__self__, 'device') else None

            # If we couldn't extract the function or device, return original
            if circuit_func is None or original_device is None:
                return circuit

            # Extract device configuration
            n_wires = original_device.num_wires
            # Check if original device has shots (sampling mode)
            original_shots = getattr(original_device, 'shots', None)

            # Try to create an alternative device with different configuration
            try:
                # Attempt to use lightning.qubit for potentially different simulation
                # Always use shots to support qml.sample() measurements
                if original_shots is not None:
                    alt_device = qml.device('lightning.qubit', wires=n_wires, shots=original_shots)
                else:
                    alt_device = qml.device('lightning.qubit', wires=n_wires, shots=1000)
                alt_qnode = qml.QNode(circuit_func, alt_device)
                return alt_qnode
            except Exception:
                pass

            # Fallback: create another default.qubit device with shots enabled
            try:
                if original_shots is not None:
                    alt_device = qml.device('default.qubit', wires=n_wires, shots=original_shots)
                else:
                    alt_device = qml.device('default.qubit', wires=n_wires, shots=1000)
                alt_qnode = qml.QNode(circuit_func, alt_device)
                return alt_qnode
            except Exception:
                pass

            return circuit
        except Exception:
            return circuit

    def ks_diff_test(self, circuit: Any, circuit_number: int, shots: int = 1000) -> float:
        if circuit is None:
            print("No PennyLane circuit provided.")
            return 0.0

        try:
            baseline = circuit
            comparison = self._create_alt_device_circuit(circuit)
            
            # Check if we got a different circuit; if not, both will be identical
            if comparison is circuit:
                print(f"Warning: Could not create alternative device; using same circuit for both baseline and comparison")

            baseline_output = self._call_circuit(baseline, seed=0)
            comparison_output = self._call_circuit(comparison, seed=1)

            counts1 = self._counts_from_output(baseline_output, seed=0)
            counts2 = self._counts_from_output(comparison_output, seed=1)

            total_shots = max(sum(counts1.values()), sum(counts2.values()))
            if total_shots == 0:
                print(f"Warning: No samples collected for PennyLane circuit {circuit_number}")
                return 1.0

            ks_value = self.ks_test(counts1, counts2, total_shots)
            print(f"PennyLane ks-test p-value: {ks_value}")

            if ks_value < self.KS_THRESHOLD:
                print(f"Interesting circuit found: {circuit_number}")
                self.save_interesting_circuit(circuit_number)

            if self.plot:
                self.plot_histogram(counts1, "PennyLane Baseline Results", 0, circuit_number)
                self.plot_histogram(counts2, "PennyLane Transformed Results", 1, circuit_number)

            return ks_value
        except Exception as error:
            error_msg = str(error)
            # Check if this is a measurement mode compatibility issue
            if "not accepted for analytic simulation" in error_msg or "analytic" in error_msg.lower():
                print(f"Measurement mode incompatibility detected: {error}")
                print("Hint: Try using qml.expval() instead of qml.sample(observable), or add shots to the device.")
            else:
                print("Error during pennylane differential testing:", error)
            self.save_interesting_circuit(circuit_number)
            return 0.0