#!/usr/bin/env python3
"""Reproduce WWL, WL, and WL-OA baselines under an explicit CV protocol."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import igraph as ig
import numpy as np
import ot
from grakel import Graph as GrakelGraph
from grakel.kernels import VertexHistogram, WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from wwl.propagation_scheme import ContinuousWeisfeilerLehman, WeisfeilerLehman as WWLPropagation

from diagnose_wllb_ksvm import parse_tu_dataset


_WWL_REPRESENTATIONS: list[np.ndarray] = []
_WWL_MAX_K = 4
_WWL_RELEASED_REPRESENTATIONS: list[list[np.ndarray]] = []
_WWL_RELEASED_METRICS: list[str] = []


def _wwl_pair(task: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    i, j = task
    first = _WWL_REPRESENTATIONS[i]
    second = _WWL_REPRESENTATIONS[j]
    values = []
    for k in range(1, _WWL_MAX_K + 1):
        costs = ot.dist(first[:, : k + 1], second[:, : k + 1], metric="euclidean")
        values.append(float(ot.emd2([], [], costs, numItermax=1_000_000)))
    return i, j, np.asarray(values)


def _wwl_released_pair(task: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    i, j = task
    values = []
    for representations, metric in zip(_WWL_RELEASED_REPRESENTATIONS, _WWL_RELEASED_METRICS):
        costs = ot.dist(representations[i], representations[j], metric=metric)
        values.append(float(ot.emd2([], [], costs, numItermax=1_000_000)))
    return i, j, np.asarray(values)


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def compute_wwl_distances(
    adjacencies: list[np.ndarray], output_dir: Path, *, workers: int, max_k: int = 4
) -> list[np.ndarray]:
    global _WWL_REPRESENTATIONS, _WWL_MAX_K
    _WWL_MAX_K = max_k
    graphs = [ig.Graph.Adjacency((adjacency > 0).tolist(), mode="undirected") for adjacency in adjacencies]
    scheme = ContinuousWeisfeilerLehman()
    _WWL_REPRESENTATIONS = scheme.fit_transform(graphs, num_iterations=max_k)
    n = len(graphs)
    paths = [output_dir / f"wwl_distance_k{k}.npy" for k in range(1, max_k + 1)]
    matrices = []
    for path in paths:
        if path.exists():
            matrix = np.load(path)
        else:
            matrix = np.full((n, n), np.nan, dtype=np.float64)
            np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    tasks = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if not all(np.isfinite(matrix[i, j]) for matrix in matrices)
    ]
    started = time.time()
    print(json.dumps({"event": "wwl_distance_start", "graphs": n, "pairs": len(tasks),
                      "workers": workers}), flush=True)
    if tasks:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            for offset, (i, j, values) in enumerate(pool.imap_unordered(_wwl_pair, tasks, chunksize=16), start=1):
                for depth, value in enumerate(values):
                    matrices[depth][i, j] = value
                    matrices[depth][j, i] = value
                if offset % 5000 == 0:
                    for path, matrix in zip(paths, matrices):
                        atomic_save(path, matrix)
                    wall = time.time() - started
                    print(json.dumps({"event": "wwl_distance_progress", "pairs_done": offset,
                                      "pairs_total": len(tasks),
                                      "pairs_per_second": offset / max(wall, 1e-9)}), flush=True)
    for path, matrix in zip(paths, matrices):
        atomic_save(path, matrix)
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise RuntimeError("incomplete WWL distance matrix")
    print(json.dumps({"event": "wwl_distance_done", "wall_seconds": time.time() - started}), flush=True)
    return matrices


def compute_wwl_released_loop_distances(
    adjacencies: list[np.ndarray], output_dir: Path, *, workers: int
) -> list[np.ndarray]:
    """Emulate the public nearest_neighbor.py loop, including graph mutation.

    The first WWL call sees unlabeled graphs and runs continuous propagation.
    That call writes degree labels onto the igraph objects. Subsequent calls on
    the same graph list therefore switch to categorical propagation.
    """
    global _WWL_RELEASED_REPRESENTATIONS, _WWL_RELEASED_METRICS
    graphs = [ig.Graph.Adjacency((adjacency > 0).tolist()) for adjacency in adjacencies]
    continuous = ContinuousWeisfeilerLehman()
    representations = [continuous.fit_transform(graphs, num_iterations=1)]
    metrics = ["euclidean"]
    for k in range(2, 5):
        categorical = WWLPropagation()
        representations.append(categorical.fit_transform(graphs, num_iterations=k))
        metrics.append("hamming")
    _WWL_RELEASED_REPRESENTATIONS = representations
    _WWL_RELEASED_METRICS = metrics

    n = len(graphs)
    paths = [output_dir / f"wwl_released_loop_distance_k{k}.npy" for k in range(1, 5)]
    matrices = []
    for path in paths:
        if path.exists():
            matrix = np.load(path)
        else:
            matrix = np.full((n, n), np.nan, dtype=np.float64)
            np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)
    tasks = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if not all(np.isfinite(matrix[i, j]) for matrix in matrices)
    ]
    started = time.time()
    print(json.dumps({"event": "wwl_released_loop_start", "graphs": n,
                      "pairs": len(tasks), "workers": workers}), flush=True)
    if tasks:
        with mp.get_context("fork").Pool(processes=workers) as pool:
            for offset, (i, j, values) in enumerate(
                pool.imap_unordered(_wwl_released_pair, tasks, chunksize=16), start=1
            ):
                for depth, value in enumerate(values):
                    matrices[depth][i, j] = value
                    matrices[depth][j, i] = value
                if offset % 5000 == 0:
                    for path, matrix in zip(paths, matrices):
                        atomic_save(path, matrix)
    for path, matrix in zip(paths, matrices):
        atomic_save(path, matrix)
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise RuntimeError("incomplete released-loop WWL distance matrix")
    print(json.dumps({"event": "wwl_released_loop_done",
                      "wall_seconds": time.time() - started}), flush=True)
    return matrices


def compute_grakel_kernels(adjacencies: list[np.ndarray], max_k: int = 4) -> dict[str, list[np.ndarray]]:
    graphs = []
    for adjacency in adjacencies:
        labels = {index: str(int(degree)) for index, degree in enumerate(adjacency.sum(axis=1))}
        graphs.append(GrakelGraph(adjacency, node_labels=labels))
    result = {"wl": [], "wloa": []}
    for k in range(1, max_k + 1):
        wl = WeisfeilerLehman(n_iter=k, normalize=True, base_graph_kernel=VertexHistogram)
        wloa = WeisfeilerLehmanOptimalAssignment(n_iter=k, normalize=True)
        result["wl"].append(np.asarray(wl.fit_transform(graphs), dtype=np.float64))
        result["wloa"].append(np.asarray(wloa.fit_transform(graphs), dtype=np.float64))
        print(json.dumps({"event": "grakel_kernel_done", "k": k}), flush=True)
    return result


def one_nearest_neighbor(distance_matrices: list[np.ndarray], labels: np.ndarray) -> dict:
    per_k = []
    indices = np.arange(len(labels))
    for k, distance in enumerate(distance_matrices, start=1):
        scores = []
        for seed in range(10):
            train, test = train_test_split(indices, test_size=0.1, random_state=seed)
            model = KNeighborsClassifier(n_neighbors=1, metric="precomputed")
            model.fit(distance[np.ix_(train, train)], labels[train])
            scores.append(float(accuracy_score(labels[test], model.predict(distance[np.ix_(test, train)]))))
        per_k.append({"k": k, "mean_accuracy": float(np.mean(scores)),
                      "std_accuracy": float(np.std(scores)), "accuracies": scores})
    return {"per_k": per_k, "best": max(per_k, key=lambda row: row["mean_accuracy"])}


def _cv_score(kernel: np.ndarray, labels: np.ndarray, c_value: float,
              splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores = []
    for train, valid in splits:
        model = SVC(kernel="precomputed", C=c_value, max_iter=-1)
        model.fit(kernel[np.ix_(train, train)], labels[train])
        scores.append(float(accuracy_score(labels[valid], model.predict(kernel[np.ix_(valid, train)]))))
    return float(np.mean(scores))


def nested_kernel_svm(
    kernels: list[np.ndarray], labels: np.ndarray, *, seed: int, gamma_grid: list[float] | None
) -> dict:
    c_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    rows = []
    for fold, (train, test) in enumerate(outer.split(kernels[0], labels)):
        y_train = labels[train]
        y_test = labels[test]
        inner = list(StratifiedKFold(n_splits=10, shuffle=False).split(train, y_train))
        best = None
        for k, base_kernel in enumerate(kernels, start=1):
            train_kernel = base_kernel[np.ix_(train, train)]
            test_kernel = base_kernel[np.ix_(test, train)]
            gammas = gamma_grid if gamma_grid is not None else [None]
            for gamma in gammas:
                candidate_train = np.exp(-gamma * train_kernel) if gamma is not None else train_kernel
                candidate_test = np.exp(-gamma * test_kernel) if gamma is not None else test_kernel
                for c_value in c_grid:
                    score = _cv_score(candidate_train, y_train, c_value, inner)
                    candidate = {"k": k, "gamma": gamma, "C": c_value,
                                 "cv_accuracy": score, "train": candidate_train,
                                 "test": candidate_test}
                    if best is None or score > best["cv_accuracy"]:
                        best = candidate
        assert best is not None
        model = SVC(kernel="precomputed", C=best["C"], max_iter=-1)
        model.fit(best["train"], y_train)
        prediction = model.predict(best["test"])
        row = {key: value for key, value in best.items() if key not in {"train", "test"}}
        row.update({"fold": fold, "accuracy": float(accuracy_score(y_test, prediction))})
        rows.append(row)
        print(json.dumps({"event": "baseline_fold", **row}), flush=True)
    scores = np.asarray([row["accuracy"] for row in rows])
    return {"rows": rows, "mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adjacencies, labels = parse_tu_dataset(args.dataset_dir, args.dataset_name)
    wwl_distances = compute_wwl_distances(adjacencies, args.output_dir, workers=args.workers, max_k=5)
    wwl_released_loop_distances = compute_wwl_released_loop_distances(
        adjacencies, args.output_dir, workers=args.workers
    )
    grakel_kernels = compute_grakel_kernels(adjacencies)
    for name, kernels in grakel_kernels.items():
        for k, kernel in enumerate(kernels, start=1):
            atomic_save(args.output_dir / f"{name}_kernel_k{k}.npy", kernel)

    gamma_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    summary = {
        "schema": "wl-distance-baseline-reproduction-v1",
        "dataset": args.dataset_name,
        "graph_count": len(labels),
        "seed": args.seed,
        "protocol_note": "both the manuscript-stated protocol and the released-code protocol are reported because they disagree",
        "wwl_source_commit": "107a8dfe3d97d8996753dbdc695f4577514cacbf",
        "wwl_1nn_manuscript": one_nearest_neighbor(wwl_distances[:4], labels),
        "wwl_1nn_released_loop": one_nearest_neighbor(wwl_released_loop_distances, labels),
        "wwl_svm_manuscript": nested_kernel_svm(wwl_distances[:4], labels, seed=args.seed, gamma_grid=gamma_grid),
        "wwl_svm_released_fixed_k5": nested_kernel_svm([wwl_distances[4]], labels, seed=args.seed, gamma_grid=gamma_grid),
        "wl_svm": nested_kernel_svm(grakel_kernels["wl"], labels, seed=args.seed, gamma_grid=None),
        "wloa_svm": nested_kernel_svm(grakel_kernels["wloa"], labels, seed=args.seed, gamma_grid=None),
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "summary": str(path)}), flush=True)


if __name__ == "__main__":
    main()
