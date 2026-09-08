import numpy as np
import pennylane as qml


class QuantumKernel:

    def __init__(
        self,
        n_qubits=4,
        n_layers=2
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        self.feature_map = self._build_feature_map()

    def _build_feature_map(self):

        @qml.qnode(self.device)
        def circuit(x):

            # ---------------------------------
            # Initial state
            # ---------------------------------

            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # ---------------------------------
            # Data encoding + entanglement
            # ---------------------------------

            for layer in range(self.n_layers):

                for i in range(self.n_qubits):

                    qml.RY(
                        x[i],
                        wires=i
                    )

                # Ring entanglement
                for i in range(self.n_qubits - 1):

                    qml.CNOT(
                        wires=[i, i + 1]
                    )

                qml.CNOT(
                    wires=[
                        self.n_qubits - 1,
                        0
                    ]
                )

            return qml.state()

        return circuit

    def state_vector(self, x):

        return np.asarray(
            self.feature_map(x)
        )

    def kernel_value(self, x1, x2):

        state_1 = self.state_vector(x1)
        state_2 = self.state_vector(x2)

        overlap = np.vdot(
            state_1,
            state_2
        )

        return float(
            np.abs(overlap) ** 2
        )

    def matrix(self, X, Y=None):

        X = np.asarray(X)

        if Y is None:
            Y = X

        Y = np.asarray(Y)

        matrix = np.zeros(
            (len(X), len(Y))
        )

        for i in range(len(X)):

            for j in range(len(Y)):

                matrix[i, j] = (
                    self.kernel_value(
                        X[i],
                        Y[j]
                    )
                )

        return matrix