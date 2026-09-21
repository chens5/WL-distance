#!/usr/bin/env python3
"""Nested-CV evaluation of a distance matrix with the full-rank Loosli KSVM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from loosli_ksvm import (
    fit_prepared_one_vs_one,
    model_diagnostics,
    prepare_one_vs_one,
)


PAPER_GRID = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
EXPECTED_LABEL_FUNCTION = {
    "f1": "f1(G,v)=degree_G(v)",
    "f2": "f2(G,v)=degree_G(v)+1/|V_G|",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_labels(archive_path: Path, dataset: str) -> np.ndarray:
    suffix = f"{dataset}_graph_labels.txt"
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {suffix} in {archive_path}, found {candidates}")
        with archive.open(candidates[0]) as handle:
            return np.loadtxt(handle, dtype=np.int64)


def validate_provenance(
    summary_path: Path,
    *,
    dataset: str,
    label_mode: str,
    depth: int,
    distance_kind: str,
) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != dataset:
        raise RuntimeError(f"source dataset mismatch in {summary_path}")
    if int(summary.get("k", -1)) != depth:
        raise RuntimeError(f"source depth mismatch in {summary_path}")
    expected_label = EXPECTED_LABEL_FUNCTION[label_mode]
    if summary.get("label_function") != expected_label:
        raise RuntimeError(
            f"refusing source with wrong {label_mode} definition: "
            f"{summary.get('label_function')!r} != {expected_label!r}"
        )
    expected_distance_phrase = "full WL" if distance_kind == "dwl" else "WLLB"
    if expected_distance_phrase.lower() not in summary.get("distance_definition", "").lower():
        raise RuntimeError(f"source distance definition mismatch in {summary_path}")
    return summary


def validate_distance(distance: np.ndarray, labels: np.ndarray) -> dict:
    if distance.shape != (len(labels), len(labels)):
        raise RuntimeError(f"distance shape {distance.shape} does not match {len(labels)} labels")
    if not np.all(np.isfinite(distance)):
        raise RuntimeError("distance matrix contains non-finite entries")
    symmetry_error = float(np.max(np.abs(distance - distance.T)))
    diagonal_error = float(np.max(np.abs(np.diag(distance))))
    minimum = float(distance.min())
    if symmetry_error > 1e-8 or diagonal_error > 1e-8 or minimum < -1e-8:
        raise RuntimeError(
            f"invalid distance matrix: symmetry={symmetry_error}, diagonal={diagonal_error}, min={minimum}"
        )
    return {
        "shape": list(distance.shape),
        "symmetry_error": symmetry_error,
        "diagonal_error": diagonal_error,
        "minimum": minimum,
        "maximum": float(distance.max()),
    }


def score_one_inner_gamma(
    distance_train: np.ndarray,
    labels_train: np.ndarray,
    inner_train: np.ndarray,
    inner_valid: np.ndarray,
    gamma: float,
    c_values: np.ndarray,
) -> np.ndarray:
    kernel_train = np.exp(-gamma * distance_train[np.ix_(inner_train, inner_train)])
    kernel_valid = np.exp(-gamma * distance_train[np.ix_(inner_valid, inner_train)])
    y_inner_train = labels_train[inner_train]
    y_inner_valid = labels_train[inner_valid]
    prepared = prepare_one_vs_one(kernel_train, y_inner_train)
    scores = []
    for c_value in c_values:
        model = fit_prepared_one_vs_one(prepared, y_inner_train, float(c_value))
        scores.append(float(accuracy_score(y_inner_valid, model.predict(kernel_valid))))
    return np.asarray(scores)


def select_parameters(
    distance_train: np.ndarray,
    labels_train: np.ndarray,
    *,
    inner_splits: int,
    workers: int,
) -> tuple[dict, np.ndarray]:
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=False)
    splits = list(splitter.split(distance_train, labels_train))
    tasks = [
        (fold, gamma_index, inner_train, inner_valid, float(gamma))
        for fold, (inner_train, inner_valid) in enumerate(splits)
        for gamma_index, gamma in enumerate(PAPER_GRID)
    ]
    results = joblib.Parallel(n_jobs=workers, prefer="threads")(
        joblib.delayed(score_one_inner_gamma)(
            distance_train,
            labels_train,
            inner_train,
            inner_valid,
            gamma,
            PAPER_GRID,
        )
        for _, _, inner_train, inner_valid, gamma in tasks
    )
    score_cube = np.empty((inner_splits, len(PAPER_GRID), len(PAPER_GRID)), dtype=np.float64)
    for (fold, gamma_index, _, _, _), scores in zip(tasks, results):
        score_cube[fold, gamma_index] = scores
    mean_scores = score_cube.mean(axis=0)
    # Flattening is gamma-major then C-major, matching the released loop order.
    gamma_index, c_index = np.unravel_index(int(np.argmax(mean_scores)), mean_scores.shape)
    return (
        {
            "gamma": float(PAPER_GRID[gamma_index]),
            "C": float(PAPER_GRID[c_index]),
            "inner_cv_accuracy": float(mean_scores[gamma_index, c_index]),
            "tie_break": "first maximum in ascending gamma-major, C-minor grid order",
        },
        mean_scores,
    )


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-path", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--distance-kind", choices=("wllb", "dwl"), required=True)
    parser.add_argument("--label-mode", choices=("f1", "f2"), required=True)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--q", type=float, default=0.6)
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_summary = validate_provenance(
        args.source_summary,
        dataset=args.dataset,
        label_mode=args.label_mode,
        depth=args.depth,
        distance_kind=args.distance_kind,
    )
    labels = load_labels(args.data_archive, args.dataset)
    distance = np.load(args.distance_path)
    matrix_checks = validate_distance(distance, labels)
    outer = StratifiedKFold(n_splits=args.outer_splits, shuffle=True, random_state=args.seed)

    rows = []
    run_started = time.time()
    for fold, (train, test) in enumerate(outer.split(distance, labels)):
        fold_started = time.time()
        distance_train = distance[np.ix_(train, train)]
        labels_train = labels[train]
        best, score_grid = select_parameters(
            distance_train,
            labels_train,
            inner_splits=args.inner_splits,
            workers=args.workers,
        )
        train_kernel = np.exp(-best["gamma"] * distance_train)
        test_kernel = np.exp(-best["gamma"] * distance[np.ix_(test, train)])
        prepared = prepare_one_vs_one(train_kernel, labels_train)
        model = fit_prepared_one_vs_one(prepared, labels_train, best["C"])
        prediction, prediction_metadata = model.predict_with_metadata(test_kernel)
        diagnostics = model_diagnostics(model)
        row = {
            "fold": fold,
            **best,
            "accuracy": float(accuracy_score(labels[test], prediction)),
            "confusion_matrix": confusion_matrix(labels[test], prediction, labels=np.unique(labels)).tolist(),
            "prediction_counts": {
                str(int(key)): int(value)
                for key, value in zip(*np.unique(prediction, return_counts=True))
            },
            **prediction_metadata,
            "fit_seconds": time.time() - fold_started,
            "model_diagnostics": diagnostics,
            "inner_score_grid": score_grid.tolist(),
        }
        rows.append(row)
        print(json.dumps({"event": "outer_fold_complete", **row}, default=json_safe), flush=True)

    accuracies = np.asarray([row["accuracy"] for row in rows])
    summary = {
        "schema": "wl-distance-loosli-ksvm-v1",
        "algorithm": {
            "name": "full-rank KSVM",
            "reference": "Loosli, Canu, and Ong, Learning SVM in Krein Spaces",
            "steps": "Algorithm 1: eigendecompose label-weighted G; solve SVM with |spectrum|; map alpha back with U sign(D) U^T; predict with original kernel",
            "multiclass": "one-vs-one C-SVC/libsvm decomposition; first class wins a vote tie",
            "test_kernel": "original indefinite exp(-gamma * distance), never clipped or projected",
        },
        "paper_settings": {
            "dataset": args.dataset,
            "distance_kind": args.distance_kind,
            "label_mode": args.label_mode,
            "label_function": EXPECTED_LABEL_FUNCTION[args.label_mode],
            "depth": args.depth,
            "q": args.q,
            "C_grid": PAPER_GRID.tolist(),
            "gamma_grid": PAPER_GRID.tolist(),
        },
        "reproduction_choices_not_stated_in_manuscript": {
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "outer_shuffle": True,
            "inner_shuffle": False,
            "seed": args.seed,
            "rationale": "matches the released vanilla-SVM nested 10-fold structure while adding a fixed seed",
        },
        "source": {
            "distance_path": str(args.distance_path),
            "distance_sha256": file_sha256(args.distance_path),
            "source_summary_path": str(args.source_summary),
            "source_summary_sha256": file_sha256(args.source_summary),
            "data_archive": str(args.data_archive),
            "data_archive_sha256": file_sha256(args.data_archive),
            "source_summary": source_summary,
        },
        "matrix_checks": matrix_checks,
        "graph_count": int(len(labels)),
        "class_counts": {
            str(int(key)): int(value) for key, value in zip(*np.unique(labels, return_counts=True))
        },
        "aggregate": {
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": float(accuracies.std()),
            "fold_accuracies": accuracies.tolist(),
        },
        "folds": rows,
        "runtime": {
            "elapsed_seconds": time.time() - run_started,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")

    csv_path = args.output_dir / "fold_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "fold", "gamma", "C", "inner_cv_accuracy", "accuracy",
            "ovo_voting_ties", "fit_seconds", "prediction_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(json.dumps({"event": "complete", "summary": str(summary_path), **summary["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
