#!/usr/bin/env python3
"""Exact (full-rank) Krein SVM from Loosli, Canu, and Ong (2016).

The paper defines the label-weighted Gram matrix G_ij = y_i y_j K_ij,
solves an ordinary SVM after replacing its spectrum by |D|, and maps the
dual variables back with U sign(D) U^T.  This module implements that algorithm
without modifying the test kernel.  Multiclass classification uses the same
one-vs-one decomposition as C-SVC/libsvm.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.svm import SVC


@dataclass
class PreparedBinaryProblem:
    kernel: np.ndarray
    labels: np.ndarray
    eigenvectors: np.ndarray
    eigenvalue_signs: np.ndarray
    auxiliary_kernel: np.ndarray
    min_eigenvalue: float
    negative_eigenvalue_count: int
    negative_spectral_mass_fraction: float


@dataclass
class BinaryKreinModel:
    classes: tuple[int | float, int | float]
    train_columns: np.ndarray
    signed_labels: np.ndarray
    alpha_tilde: np.ndarray
    alpha: np.ndarray
    original_coefficients: np.ndarray
    intercept: float
    min_eigenvalue: float
    negative_eigenvalue_count: int
    negative_spectral_mass_fraction: float
    max_auxiliary_decision_error: float
    max_original_decision_error: float
    dual_equality_error: float
    fit_status: int

    def decision_function(self, test_kernel: np.ndarray) -> np.ndarray:
        return np.asarray(test_kernel, dtype=np.float64) @ self.original_coefficients + self.intercept

    def predict(self, test_kernel: np.ndarray) -> np.ndarray:
        decision = self.decision_function(test_kernel)
        return np.where(decision > 0.0, self.classes[1], self.classes[0])


@dataclass
class OneVsOneKreinModel:
    classes: np.ndarray
    pair_models: list[BinaryKreinModel]

    def predict_with_metadata(self, test_kernel: np.ndarray) -> tuple[np.ndarray, dict]:
        test_kernel = np.asarray(test_kernel, dtype=np.float64)
        votes = np.zeros((test_kernel.shape[0], len(self.classes)), dtype=np.int64)
        class_to_column = {value: index for index, value in enumerate(self.classes.tolist())}
        for model in self.pair_models:
            pair_kernel = test_kernel[:, model.train_columns]
            pair_prediction = model.predict(pair_kernel)
            for class_value in model.classes:
                mask = pair_prediction == class_value
                votes[mask, class_to_column[class_value]] += 1
        maxima = votes.max(axis=1, keepdims=True)
        tie_count = int(np.sum((votes == maxima).sum(axis=1) > 1))
        # np.argmax chooses the first class on a voting tie, as libsvm does.
        prediction = self.classes[np.argmax(votes, axis=1)]
        return prediction, {"ovo_voting_ties": tie_count}

    def predict(self, test_kernel: np.ndarray) -> np.ndarray:
        return self.predict_with_metadata(test_kernel)[0]


def _validate_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"training kernel must be square, got {kernel.shape}")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("training kernel contains non-finite entries")
    asymmetry = float(np.max(np.abs(kernel - kernel.T)))
    if asymmetry > 1e-8:
        raise ValueError(f"training kernel is not symmetric (max error {asymmetry})")
    return 0.5 * (kernel + kernel.T)


def prepare_binary_problem(
    kernel: np.ndarray,
    signed_labels: np.ndarray,
    *,
    eigenvalue_tolerance: float = 1e-10,
) -> PreparedBinaryProblem:
    kernel = _validate_kernel(kernel)
    labels = np.asarray(signed_labels, dtype=np.float64)
    if kernel.shape[0] != len(labels):
        raise ValueError("kernel and labels have inconsistent sizes")
    if set(np.unique(labels).tolist()) != {-1.0, 1.0}:
        raise ValueError("binary labels must be encoded as -1 and +1")

    weighted_gram = labels[:, None] * kernel * labels[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_gram)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    threshold = eigenvalue_tolerance * scale
    # Algorithm 1 uses S=sign(D).  Do not truncate numerically small nonzero
    # modes here: doing so while retaining |D| in the stabilized Gram matrix
    # breaks the exact map-back identity G U S U^T alpha_tilde = G_tilde alpha_tilde.
    signs = np.sign(eigenvalues)
    absolute_gram = (eigenvectors * np.abs(eigenvalues)) @ eigenvectors.T
    auxiliary_kernel = labels[:, None] * absolute_gram * labels[None, :]
    auxiliary_kernel = 0.5 * (auxiliary_kernel + auxiliary_kernel.T)
    negative = eigenvalues[eigenvalues < -threshold]
    spectral_mass = max(float(np.abs(eigenvalues).sum()), np.finfo(float).eps)
    return PreparedBinaryProblem(
        kernel=kernel,
        labels=labels,
        eigenvectors=eigenvectors,
        eigenvalue_signs=signs,
        auxiliary_kernel=auxiliary_kernel,
        min_eigenvalue=float(eigenvalues[0]),
        negative_eigenvalue_count=int(len(negative)),
        negative_spectral_mass_fraction=float(np.abs(negative).sum() / spectral_mass),
    )


def fit_prepared_binary(
    prepared: PreparedBinaryProblem,
    c_value: float,
    *,
    classes: tuple[int | float, int | float] = (-1, 1),
    train_columns: np.ndarray | None = None,
) -> BinaryKreinModel:
    svc = SVC(kernel="precomputed", C=float(c_value), max_iter=-1)
    svc.fit(prepared.auxiliary_kernel, prepared.labels)

    alpha_tilde = np.zeros(len(prepared.labels), dtype=np.float64)
    alpha_tilde[svc.support_] = svc.dual_coef_[0] * prepared.labels[svc.support_]
    projection = prepared.eigenvectors.T @ alpha_tilde
    alpha = (prepared.eigenvectors * prepared.eigenvalue_signs) @ projection
    original_coefficients = prepared.labels * alpha
    intercept = float(svc.intercept_[0])

    auxiliary_decision = prepared.auxiliary_kernel @ (prepared.labels * alpha_tilde) + intercept
    sklearn_decision = np.asarray(svc.decision_function(prepared.auxiliary_kernel))
    original_decision = prepared.kernel @ original_coefficients + intercept
    max_auxiliary_error = float(np.max(np.abs(auxiliary_decision - sklearn_decision)))
    max_original_error = float(np.max(np.abs(original_decision - auxiliary_decision)))
    if max_auxiliary_error > 1e-7 or max_original_error > 1e-6:
        raise RuntimeError(
            "Loosli transform consistency check failed: "
            f"aux={max_auxiliary_error}, original={max_original_error}"
        )

    if train_columns is None:
        train_columns = np.arange(len(prepared.labels), dtype=np.int64)
    return BinaryKreinModel(
        classes=classes,
        train_columns=np.asarray(train_columns, dtype=np.int64),
        signed_labels=prepared.labels,
        alpha_tilde=alpha_tilde,
        alpha=alpha,
        original_coefficients=original_coefficients,
        intercept=intercept,
        min_eigenvalue=prepared.min_eigenvalue,
        negative_eigenvalue_count=prepared.negative_eigenvalue_count,
        negative_spectral_mass_fraction=prepared.negative_spectral_mass_fraction,
        max_auxiliary_decision_error=max_auxiliary_error,
        max_original_decision_error=max_original_error,
        dual_equality_error=float(abs(prepared.labels @ alpha_tilde)),
        fit_status=int(svc.fit_status_),
    )


def prepare_one_vs_one(kernel: np.ndarray, labels: np.ndarray) -> list[tuple[PreparedBinaryProblem, tuple, np.ndarray]]:
    kernel = _validate_kernel(kernel)
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
            (prepare_binary_problem(pair_kernel, signed), (class_a, class_b), columns)
        )
    return prepared_pairs


def fit_prepared_one_vs_one(
    prepared_pairs: list[tuple[PreparedBinaryProblem, tuple, np.ndarray]],
    labels: np.ndarray,
    c_value: float,
) -> OneVsOneKreinModel:
    pair_models = [
        fit_prepared_binary(problem, c_value, classes=classes, train_columns=columns)
        for problem, classes, columns in prepared_pairs
    ]
    return OneVsOneKreinModel(classes=np.unique(labels), pair_models=pair_models)


def fit_loosli_ksvm(kernel: np.ndarray, labels: np.ndarray, c_value: float) -> OneVsOneKreinModel:
    return fit_prepared_one_vs_one(prepare_one_vs_one(kernel, labels), labels, c_value)


def model_diagnostics(model: OneVsOneKreinModel) -> dict:
    pairs = []
    for pair_model in model.pair_models:
        pairs.append(
            {
                "classes": list(pair_model.classes),
                "training_size": int(len(pair_model.signed_labels)),
                "min_eigenvalue": pair_model.min_eigenvalue,
                "negative_eigenvalue_count": pair_model.negative_eigenvalue_count,
                "negative_spectral_mass_fraction": pair_model.negative_spectral_mass_fraction,
                "max_auxiliary_decision_error": pair_model.max_auxiliary_decision_error,
                "max_original_decision_error": pair_model.max_original_decision_error,
                "dual_equality_error": pair_model.dual_equality_error,
                "fit_status": pair_model.fit_status,
            }
        )
    return {"pair_models": pairs}
