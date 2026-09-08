import os
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "results/roberta_quantum_features"

OUTPUT_DIR = "results/roberta_pca_features"

DIMENSIONS = [2, 4, 6, 8]


# ============================================================
# LOAD RAW ROBERTA EMBEDDINGS
# ============================================================

print("=" * 80)
print("ROBERTA → PCA DIMENSIONALITY REDUCTION")
print("=" * 80)

X_train = np.load(
    f"{INPUT_DIR}/X_train.npy"
)

y_train = np.load(
    f"{INPUT_DIR}/y_train.npy"
)

X_val = np.load(
    f"{INPUT_DIR}/X_val.npy"
)

y_val = np.load(
    f"{INPUT_DIR}/y_val.npy"
)

X_test = np.load(
    f"{INPUT_DIR}/X_test.npy"
)

y_test = np.load(
    f"{INPUT_DIR}/y_test.npy"
)


print("\nOriginal shapes:")
print("Training:   ", X_train.shape)
print("Validation: ", X_val.shape)
print("Testing:    ", X_test.shape)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n")
print("=" * 80)
print("FITTING STANDARD SCALER")
print("=" * 80)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)

print("Scaling complete.")


# ============================================================
# PCA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for dimensions in DIMENSIONS:

    print("\n")
    print("=" * 80)

    print(
        f"PCA DIMENSIONS = {dimensions}"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Fit PCA ONLY on training data
    # --------------------------------------------------------

    pca = PCA(
        n_components=dimensions,
        random_state=42
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    # --------------------------------------------------------
    # Apply same PCA to validation/test
    # --------------------------------------------------------

    X_val_pca = pca.transform(
        X_val_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Explained variance
    # --------------------------------------------------------

    explained_variance = (
        pca.explained_variance_ratio_
    )

    total_variance = (
        explained_variance.sum()
    )

    print(
        f"Explained variance:"
        f" {total_variance:.6f}"
    )

    print(
        "Training shape:",
        X_train_pca.shape
    )

    print(
        "Validation shape:",
        X_val_pca.shape
    )

    print(
        "Testing shape:",
        X_test_pca.shape
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dimension_dir = os.path.join(
        OUTPUT_DIR,
        f"pca_{dimensions}"
    )

    os.makedirs(
        dimension_dir,
        exist_ok=True
    )

    np.save(
        f"{dimension_dir}/X_train.npy",
        X_train_pca
    )

    np.save(
        f"{dimension_dir}/y_train.npy",
        y_train
    )

    np.save(
        f"{dimension_dir}/X_val.npy",
        X_val_pca
    )

    np.save(
        f"{dimension_dir}/y_val.npy",
        y_val
    )

    np.save(
        f"{dimension_dir}/X_test.npy",
        X_test_pca
    )

    np.save(
        f"{dimension_dir}/y_test.npy",
        y_test
    )

    # Save PCA information

    np.save(
        f"{dimension_dir}/explained_variance.npy",
        explained_variance
    )

    print(
        f"Saved to: {dimension_dir}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("PCA PROCESS COMPLETE")
print("=" * 80)

print(
    "\nCreated:"
)

for dimensions in DIMENSIONS:

    print(
        f"  {OUTPUT_DIR}/pca_{dimensions}/"
    )