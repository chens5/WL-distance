"""Krein-space SVM for indefinite kernels.

This is a direct Python implementation of Algorithm 1 in

    Loosli, Canu, and Ong, "Learning SVM in Krein Spaces",
    IEEE TPAMI 38(6), 2016.

For binary labels y in {-1, +1}, the algorithm eigendecomposes the
label-weighted Gram matrix G = Y K Y, solves an ordinary SVM after replacing
its spectrum by its absolute value, maps the dual solution back with
U sign(D) U^T, and predicts with the original (possibly indefinite) kernel.
Multiclass classification uses a one-vs-one decomposition.

The public repository previously used an eigenvalue-clipping helper from WTK.
Clipping produces a positive-semidefinite approximation and is not the KSVM
algorithm described above.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


PAPER_GRID = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])


@dataclass
class PreparedBinaryProblem:
    kernel: np.ndarray
    labels: np.ndarray
    eigenvectors: np.ndarray
    eigenvalue_signs: np.ndarray
    auxiliary_kernel: np.ndarray


@dataclass
class BinaryKreinModel:
    classes: tuple
    train_columns: np.ndarray
    original_coefficients: np.ndarray
    intercept: float

    def decision_function(self, test_kernel: np.ndarray) -> np.ndarray:
        return (
            np.asarray(test_kernel, dtype=np.float64)
            @ self.original_coefficients
            + self.intercept
        )

    def predict(self, test_kernel: np.ndarray) -> np.ndarray:
        decision = self.decision_function(test_kernel)
        return np.where(decision > 0.0, self.classes[1], self.classes[0])


@dataclass
class OneVsOneKreinModel:
    classes: np.ndarray
    pair_models: list[BinaryKreinModel]

    def predict(self, test_kernel: np.ndarray) -> np.ndarray:
        test_kernel = np.asarray(test_kernel, dtype=np.float64)
        votes = np.zeros((test_kernel.shape[0], len(self.classes)), dtype=np.int64)
        class_to_column = {
            value: index for index, value in enumerate(self.classes.tolist())
        }
        for model in self.pair_models:
            pair_prediction = model.predict(test_kernel[:, model.train_columns])
            for class_value in model.classes:
                votes[pair_prediction == class_value, class_to_column[class_value]] += 1
        # np.argmax resolves voting ties in favor of the first sorted class,
        # matching libsvm's deterministic one-vs-one behavior.
        return self.classes[np.argmax(votes, axis=1)]


def _validate_training_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"training kernel must be square, got {kernel.shape}")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("training kernel contains non-finite entries")
    asymmetry = float(np.max(np.abs(kernel - kernel.T)))
    if asymmetry > 1e-8:
        raise ValueError(
            f"training kernel is not symmetric (maximum error {asymmetry})"
        )
    return 0.5 * (kernel + kernel.T)


def prepare_binary_problem(
    kernel: np.ndarray, signed_labels: np.ndarray
) -> PreparedBinaryProblem:
    """Construct the positive-semidefinite auxiliary problem in Algorithm 1."""

    kernel = _validate_training_kernel(kernel)
    labels = np.asarray(signed_labels, dtype=np.float64)
    if kernel.shape[0] != len(labels):
        raise ValueError("kernel and labels have inconsistent sizes")
    if set(np.unique(labels).tolist()) != {-1.0, 1.0}:
        raise ValueError("binary labels must be encoded as -1 and +1")

    weighted_gram = labels[:, None] * kernel * labels[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_gram)

    # Algorithm 1 specifies S = sign(D).  In particular, do not clip negative
    # eigenvalues or truncate small nonzero modes: either change breaks the
    # exact map from the auxiliary solution back to the original kernel.
    signs = np.sign(eigenvalues)
    absolute_gram = (eigenvectors * np.abs(eigenvalues)) @ eigenvectors.T
    auxiliary_kernel = labels[:, None] * absolute_gram * labels[None, :]
    auxiliary_kernel = 0.5 * (auxiliary_kernel + auxiliary_kernel.T)
    return PreparedBinaryProblem(
        kernel=kernel,
        labels=labels,
        eigenvectors=eigenvectors,
        eigenvalue_signs=signs,
        auxiliary_kernel=auxiliary_kernel,
    )


def fit_prepared_binary(
    prepared: PreparedBinaryProblem,
    c_value: float,
    *,
    classes: tuple = (-1, 1),
    train_columns: np.ndarray | None = None,
) -> BinaryKreinModel:
    """Solve the auxiliary SVM and map its dual solution back to Krein space."""

    svc = SVC(kernel="precomputed", C=float(c_value), max_iter=-1)
    svc.fit(prepared.auxiliary_kernel, prepared.labels)

    # sklearn stores y_i * alpha_tilde_i for support vectors.
    alpha_tilde = np.zeros(len(prepared.labels), dtype=np.float64)
    alpha_tilde[svc.support_] = (
        svc.dual_coef_[0] * prepared.labels[svc.support_]
    )
    alpha = (
        prepared.eigenvectors * prepared.eigenvalue_signs
    ) @ (prepared.eigenvectors.T @ alpha_tilde)
    original_coefficients = prepared.labels * alpha
    intercept = float(svc.intercept_[0])

    # Verify the identity that distinguishes Algorithm 1 from clipping:
    # predictions mapped back to K must equal the auxiliary SVM predictions.
    auxiliary_decision = (
        prepared.auxiliary_kernel @ (prepared.labels * alpha_tilde) + intercept
    )
    original_decision = prepared.kernel @ original_coefficients + intercept
    maximum_error = float(np.max(np.abs(original_decision - auxiliary_decision)))
    if maximum_error > 1e-6:
        raise RuntimeError(
            "Loosli KSVM map-back consistency check failed "
            f"(maximum decision error {maximum_error})"
        )

    if train_columns is None:
        train_columns = np.arange(len(prepared.labels), dtype=np.int64)
    return BinaryKreinModel(
        classes=classes,
        train_columns=np.asarray(train_columns, dtype=np.int64),
        original_coefficients=original_coefficients,
        intercept=intercept,
    )


def prepare_one_vs_one(
    kernel: np.ndarray, labels: np.ndarray
) -> list[tuple[PreparedBinaryProblem, tuple, np.ndarray]]:
    """Prepare every binary subproblem used for multiclass classification."""

    kernel = _validate_training_kernel(kernel)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("at least two classes are required")

    prepared_pairs = []
    for class_a, class_b in combinations(classes.tolist(), 2):
        columns = np.flatnonzero((labels == class_a) | (labels == class_b))
        signed = np.where(labels[columns] == class_b, 1.0, -1.0)
        pair_kernel = kernel[np.ix_(columns, columns)]
        prepared_pairs.append(
            (prepare_binary_problem(pair_kernel, signed),
             (class_a, class_b), columns)
        )
    return prepared_pairs


def fit_prepared_one_vs_one(
    prepared_pairs: list[tuple[PreparedBinaryProblem, tuple, np.ndarray]],
    labels: np.ndarray,
    c_value: float,
) -> OneVsOneKreinModel:
    pair_models = [
        fit_prepared_binary(
            problem, c_value, classes=classes, train_columns=columns
        )
        for problem, classes, columns in prepared_pairs
    ]
    return OneVsOneKreinModel(classes=np.unique(labels), pair_models=pair_models)


def fit_loosli_ksvm(
    kernel: np.ndarray, labels: np.ndarray, c_value: float
) -> OneVsOneKreinModel:
    """Fit Algorithm 1 to a precomputed kernel matrix."""

    return fit_prepared_one_vs_one(
        prepare_one_vs_one(kernel, labels), labels, c_value
    )


class KreinSVC(BaseEstimator, ClassifierMixin):
    """KSVM estimator for a precomputed distance matrix.

    ``fit`` expects a square training distance matrix and ``predict`` expects
    test-to-training distances.  Both are converted to ``exp(-gamma * D)``.
    """

    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        distance = np.asarray(X, dtype=np.float64)
        kernel = np.exp(-float(self.gamma) * distance)
        self.model_ = fit_loosli_ksvm(kernel, np.asarray(y), float(self.C))
        self.classes_ = self.model_.classes
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("KreinSVC must be fitted before prediction")
        kernel = np.exp(-float(self.gamma) * np.asarray(X, dtype=np.float64))
        return self.model_.predict(kernel)


def _choose_parameters(
    distance: np.ndarray,
    labels: np.ndarray,
    gammas: np.ndarray,
    c_values: np.ndarray,
    cv: int,
) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=cv, shuffle=False)
    scores = np.zeros((len(gammas), len(c_values)), dtype=np.float64)
    for train, validation in splitter.split(distance, labels):
        distance_train = distance[np.ix_(train, train)]
        distance_validation = distance[np.ix_(validation, train)]
        y_train = labels[train]
        y_validation = labels[validation]
        for gamma_index, gamma in enumerate(gammas):
            train_kernel = np.exp(-gamma * distance_train)
            validation_kernel = np.exp(-gamma * distance_validation)
            prepared = prepare_one_vs_one(train_kernel, y_train)
            for c_index, c_value in enumerate(c_values):
                model = fit_prepared_one_vs_one(prepared, y_train, c_value)
                scores[gamma_index, c_index] += accuracy_score(
                    y_validation, model.predict(validation_kernel)
                )
    scores /= cv
    gamma_index, c_index = np.unravel_index(
        int(np.argmax(scores)), scores.shape
    )
    return float(gammas[gamma_index]), float(c_values[c_index])


def krein_svm_grid_search(
    D_train: np.ndarray,
    D_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    param_grid: dict | None = None,
    gammas: np.ndarray | None = None,
    cv: int = 10,
):
    """Select C and gamma by nested CV, then evaluate the Loosli KSVM.

    The default grid is the one reported in the paper: both C and gamma range
    over 10^-3, ..., 10^3.  The return shape matches the historical WTK helper.
    """

    distance_train = np.asarray(D_train, dtype=np.float64)
    distance_test = np.asarray(D_test, dtype=np.float64)
    labels_train = np.asarray(y_train)
    labels_test = np.asarray(y_test)
    if param_grid is None:
        c_values = PAPER_GRID
    else:
        c_values = np.asarray(param_grid.get("C", PAPER_GRID), dtype=np.float64)
    if gammas is None:
        gammas = PAPER_GRID
    else:
        gammas = np.asarray(gammas, dtype=np.float64)

    gamma, c_value = _choose_parameters(
        distance_train, labels_train, gammas, c_values, cv
    )
    model = KreinSVC(C=c_value, gamma=gamma).fit(
        distance_train, labels_train
    )
    prediction = model.predict(distance_test)
    model.best_params_ = {"C": c_value, "gamma": gamma}
    return model, float(accuracy_score(labels_test, prediction))
