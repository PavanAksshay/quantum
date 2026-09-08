import os
import numpy as np
import pennylane as qml


class FastQuantumKernel:

    def __init__(
        self,
        n_qubits=4,
        n_layers=1
    ):

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        self.circuit = self._build_circuit()

        # Cache already-computed states
        self.state_cache = {}

    def _build_circuit(self):

        @qml.qnode(
            self.device
        )
        def circuit(x):

            # Feature encoding

            for i in range(
                self.n_qubits
            ):

                qml.Hadamard(
                    wires=i
                )

                qml.RY(
                    x[i],
                    wires=i
                )

                qml.RZ(
                    x[i],
                    wires=i
                )

            # Entanglement

            for layer in range(
                self.n_layers
            ):

                for i in range(
                    self.n_qubits - 1
                ):

                    qml.CNOT(
                        wires=[
                            i,
                            i + 1
                        ]
                    )

                qml.CNOT(
                    wires=[
                        self.n_qubits - 1,
                        0
                    ]
                )

            return qml.state()

        return circuit

    def _key(self, x):

        return tuple(
            np.round(
                x,
                decimals=10
            )
        )

    def state(self, x):

        key = self._key(x)

        if key not in self.state_cache:

            self.state_cache[key] = (
                np.asarray(
                    self.circuit(x)
                )
            )

        return self.state_cache[key]

    def kernel_value(
        self,
        x1,
        x2
    ):

        state1 = self.state(x1)

        state2 = self.state(x2)

        overlap = np.vdot(
            state1,
            state2
        )

        return float(
            np.abs(overlap) ** 2
        )

    def matrix(
        self,
        X,
        Y=None
    ):

        X = np.asarray(X)

        if Y is None:

            Y = X

        else:

            Y = np.asarray(Y)

        matrix = np.zeros(
            (len(X), len(Y))
        )

        # Precompute states once

        X_states = [
            self.state(x)
            for x in X
        ]

        Y_states = [
            self.state(y)
            for y in Y
        ]

        for i in range(
            len(X)
        ):

            for j in range(
                len(Y)
            ):

                overlap = np.vdot(
                    X_states[i],
                    Y_states[j]
                )

                matrix[i, j] = (
                    np.abs(overlap) ** 2
                )

        return matrix