import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class HybridVQC:

    def __init__(
        self,
        n_qubits=4,
        n_layers=2,
        learning_rate=0.02,
        epochs=30,
        batch_size=None,
        seed=42
    ):

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.history = []

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        self.weights = None

        # Classical layer parameters
        self.classical_weights = None
        self.classical_bias = None

        self.qnode = self._build_circuit()

    # ==========================================
    # Quantum circuit
    # ==========================================

    def _build_circuit(self):

        @qml.qnode(
            self.device,
            interface="autograd"
        )
        def circuit(
            x,
            weights
        ):

            # ----------------------------------
            # Feature encoding
            # ----------------------------------

            for i in range(
                self.n_qubits
            ):

                qml.RY(
                    x[i],
                    wires=i
                )

                qml.RZ(
                    x[i],
                    wires=i
                )

            # ----------------------------------
            # Variational layers
            # ----------------------------------

            for layer in range(
                self.n_layers
            ):

                for i in range(
                    self.n_qubits
                ):

                    qml.RX(
                        weights[
                            layer,
                            i,
                            0
                        ],
                        wires=i
                    )

                    qml.RY(
                        weights[
                            layer,
                            i,
                            1
                        ],
                        wires=i
                    )

                    qml.RZ(
                        weights[
                            layer,
                            i,
                            2
                        ],
                        wires=i
                    )

                # ----------------------------------
                # Entanglement
                # ----------------------------------

                for i in range(
                    self.n_qubits - 1
                ):

                    qml.CNOT(
                        wires=[
                            i,
                            i + 1
                        ]
                    )

                # Ring connection

                qml.CNOT(
                    wires=[
                        self.n_qubits - 1,
                        0
                    ]
                )

            # ----------------------------------
            # Measure every qubit
            # ----------------------------------

            return [
                qml.expval(
                    qml.PauliZ(i)
                )
                for i in range(
                    self.n_qubits
                )
            ]

        return circuit

    # ==========================================
    # Feature preparation
    # ==========================================

    def _prepare_features(
        self,
        X
    ):

        X = np.asarray(X)

        X = X[
            :,
            :self.n_qubits
        ]

        # Bound PCA values

        X = np.tanh(X)

        # Map to angle range

        X = X * np.pi

        return X

    # ==========================================
    # Initialize parameters
    # ==========================================

    def _initialize_parameters(self):

        rng = np.random.default_rng(
            self.seed
        )

        quantum_weights = (
            rng.normal(
                0,
                0.1,
                size=(
                    self.n_layers,
                    self.n_qubits,
                    3
                )
            )
        )

        self.weights = pnp.array(
            quantum_weights,
            requires_grad=True
        )

        # Classical readout layer

        classical_weights = (
            rng.normal(
                0,
                0.1,
                size=self.n_qubits
            )
        )

        self.classical_weights = (
            pnp.array(
                classical_weights,
                requires_grad=True
            )
        )

        self.classical_bias = (
            pnp.array(
                0.0,
                requires_grad=True
            )
        )

    # ==========================================
    # Quantum feature extraction
    # ==========================================

    def _quantum_features(
        self,
        X,
        weights
    ):

        outputs = []

        for x in X:

            result = self.qnode(
                x,
                weights
            )

            result = pnp.stack(
                result
            )

            outputs.append(
                result
            )

        return pnp.stack(
            outputs
        )

    # ==========================================
    # Sigmoid
    # ==========================================

    def _sigmoid(
        self,
        x
    ):

        return 1 / (
            1 + qml.math.exp(-x)
        )

    # ==========================================
    # Forward pass
    # ==========================================

    def _forward(
        self,
        X,
        quantum_weights,
        classical_weights,
        classical_bias
    ):

        quantum_features = (
            self._quantum_features(
                X,
                quantum_weights
            )
        )

        logits = (
            qml.math.dot(
                quantum_features,
                classical_weights
            )
            + classical_bias
        )

        return self._sigmoid(
            logits
        )

    # ==========================================
    # Loss
    # ==========================================

    def _loss(
        self,
        quantum_weights,
        classical_weights,
        classical_bias,
        X,
        y
    ):

        predictions = self._forward(
            X,
            quantum_weights,
            classical_weights,
            classical_bias
        )

        predictions = qml.math.clip(
            predictions,
            1e-7,
            1 - 1e-7
        )

        loss = -qml.math.mean(

            y * qml.math.log(
                predictions
            )

            +

            (1 - y)
            *
            qml.math.log(
                1 - predictions
            )
        )

        return loss

    # ==========================================
    # Training
    # ==========================================

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        patience=None
    ):

        X = self._prepare_features(
            X
        )

        y = pnp.array(
            y,
            dtype=float,
            requires_grad=False
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_prep = self._prepare_features(X_val)
            y_val_arr = pnp.array(
                y_val,
                dtype=float,
                requires_grad=False
            )

        self._initialize_parameters()

        optimizer = qml.AdamOptimizer(
            stepsize=self.learning_rate
        )

        self.history = []

        n_samples = len(X)
        rng = np.random.default_rng(self.seed)

        best_val_loss = float("inf")
        best_weights = None
        best_classical_weights = None
        best_classical_bias = None
        patience_counter = 0

        print(
            "\nTraining Hybrid VQC..."
        )

        for epoch in range(
            self.epochs
        ):

            if self.batch_size is not None and self.batch_size < n_samples:
                permutation = rng.permutation(n_samples)
                epoch_loss = 0.0
                num_batches = int(np.ceil(n_samples / self.batch_size))

                for b in range(num_batches):
                    batch_idx = permutation[
                        b * self.batch_size : (b + 1) * self.batch_size
                    ]
                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]

                    params, b_loss = optimizer.step_and_cost(
                        lambda qw, cw, cb: self._loss(
                            qw,
                            cw,
                            cb,
                            X_batch,
                            y_batch
                        ),
                        self.weights,
                        self.classical_weights,
                        self.classical_bias,
                    )

                    (
                        self.weights,
                        self.classical_weights,
                        self.classical_bias,
                    ) = params

                    epoch_loss += float(b_loss)

                train_loss = epoch_loss / num_batches
            else:
                params, loss = optimizer.step_and_cost(
                    lambda qw, cw, cb: self._loss(
                        qw,
                        cw,
                        cb,
                        X,
                        y
                    ),
                    self.weights,
                    self.classical_weights,
                    self.classical_bias,
                )

                (
                    self.weights,
                    self.classical_weights,
                    self.classical_bias,
                ) = params

                train_loss = float(loss)

            history_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss
            }

            log_msg = (
                f"Epoch "
                f"{epoch + 1:02d}/"
                f"{self.epochs} "
                f"| Train Loss: "
                f"{train_loss:.6f}"
            )

            if has_val:
                val_loss = float(
                    self._loss(
                        self.weights,
                        self.classical_weights,
                        self.classical_bias,
                        X_val_prep,
                        y_val_arr
                    )
                )
                history_entry["val_loss"] = val_loss
                log_msg += f" | Val Loss: {val_loss:.6f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = pnp.copy(self.weights)
                    best_classical_weights = pnp.copy(self.classical_weights)
                    best_classical_bias = pnp.copy(self.classical_bias)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience is not None and patience_counter >= patience:
                        print(
                            f"{log_msg} -> Early stopping at epoch {epoch + 1}"
                        )
                        self.history.append(history_entry)
                        self.weights = best_weights
                        self.classical_weights = best_classical_weights
                        self.classical_bias = best_classical_bias
                        break

            self.history.append(history_entry)

            if (
                epoch == 0
                or (epoch + 1) % 5 == 0
            ):
                print(log_msg)

        if has_val and best_weights is not None and (patience is None or patience_counter < patience):
            self.weights = best_weights
            self.classical_weights = best_classical_weights
            self.classical_bias = best_classical_bias

        return self

    # ==========================================
    # Probability prediction
    # ==========================================

    def predict_proba(
        self,
        X
    ):

        X = self._prepare_features(
            X
        )

        probabilities = (
            self._forward(
                X,
                self.weights,
                self.classical_weights,
                self.classical_bias
            )
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        return np.column_stack(
            [
                1 - probabilities,
                probabilities
            ]
        )

    # ==========================================
    # Prediction
    # ==========================================

    def predict(
        self,
        X,
        threshold=0.5
    ):

        probabilities = (
            self.predict_proba(X)
            [:, 1]
        )

        return (
            probabilities >= threshold
        ).astype(int)