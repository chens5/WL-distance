#!/usr/bin/env python3
"""Recompute one WLLB/f2 setting and diagnose the released KSVM pipeline.

This script intentionally implements the paper definition
    f2(G, v) = degree_G(v) + 1 / |V_G|
directly instead of using the released ``sz_degree_mapping`` helper, whose
membership test and dictionary keys are inconsistent.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


@dataclass
class GraphDescriptor:
    n: int
    degree: np.ndarray
    mu: np.ndarray
    label_values: np.ndarray
    pushed_rows: np.ndarray


_DESCRIPTORS: list[GraphDescriptor] = []


def parse_tu_dataset(dataset_dir: Path, dataset_name: str) -> tuple[list[np.ndarray], np.ndarray]:
    prefix = dataset_dir / dataset_name
    indicator = np.loadtxt(prefix.with_name(prefix.name + "_graph_indicator.txt"), dtype=np.int64)
    labels = np.loadtxt(prefix.with_name(prefix.name + "_graph_labels.txt"), dtype=np.int64)
    edges = np.loadtxt(prefix.with_name(prefix.name + "_A.txt"), delimiter=",", dtype=np.int64)

    graph_count = int(indicator.max())
    node_ids = [np.flatnonzero(indicator == graph_id) for graph_id in range(1, graph_count + 1)]
    local_index: dict[int, tuple[int, int]] = {}
    adjacencies: list[np.ndarray] = []
    for graph_idx, ids in enumerate(node_ids):
        adjacencies.append(np.zeros((len(ids), len(ids)), dtype=np.float64))
        for local_idx, global_idx in enumerate(ids):
            local_index[int(global_idx)] = (graph_idx, local_idx)

    for raw_u, raw_v in np.atleast_2d(edges):
        u = int(raw_u) - 1
        v = int(raw_v) - 1
        graph_u, local_u = local_index[u]
        graph_v, local_v = local_index[v]
        if graph_u != graph_v:
            raise ValueError(f"cross-graph edge encountered: {raw_u}, {raw_v}")
        adjacencies[graph_u][local_u, local_v] = 1.0
        adjacencies[graph_u][local_v, local_u] = 1.0

    if len(labels) != graph_count:
        raise ValueError(f"expected {graph_count} graph labels, found {len(labels)}")
    return adjacencies, labels


def make_descriptor(adjacency: np.ndarray, *, q: float, k: int) -> GraphDescriptor:
    n = int(adjacency.shape[0])
    degree = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    nonisolated = degree > 0
    transition[nonisolated] = (1.0 - q) * adjacency[nonisolated] / degree[nonisolated, None]
    transition += q * np.eye(n)
    transition[~nonisolated, ~nonisolated] = 1.0
    transition_k = np.linalg.matrix_power(transition, k)

    degree_sum = float(degree.sum())
    mu = degree / degree_sum if degree_sum > 0 else np.full(n, 1.0 / n)
    labels = degree + 1.0 / n
    label_values, inverse = np.unique(labels, return_inverse=True)
    pushed = np.zeros((n, len(label_values)), dtype=np.float64)
    for node_idx, label_idx in enumerate(inverse):
        pushed[:, label_idx] += transition_k[:, node_idx]
    return GraphDescriptor(n, degree, mu, label_values, pushed)


def row_wasserstein_cost(g: GraphDescriptor, h: GraphDescriptor) -> np.ndarray:
    support = np.union1d(g.label_values, h.label_values)
    hist_g = np.zeros((g.n, len(support)), dtype=np.float64)
    hist_h = np.zeros((h.n, len(support)), dtype=np.float64)
    hist_g[:, np.searchsorted(support, g.label_values)] = g.pushed_rows
    hist_h[:, np.searchsorted(support, h.label_values)] = h.pushed_rows
    if len(support) == 1:
        return np.zeros((g.n, h.n), dtype=np.float64)
    gaps = np.diff(support)
    cdf_g = np.cumsum(hist_g, axis=1)[:, :-1]
    cdf_h = np.cumsum(hist_h, axis=1)[:, :-1]
    return np.sum(np.abs(cdf_g[:, None, :] - cdf_h[None, :, :]) * gaps[None, None, :], axis=2)


def wllb_pair(task: tuple[int, int]) -> tuple[int, int, float]:
    i, j = task
    g = _DESCRIPTORS[i]
    h = _DESCRIPTORS[j]
    node_cost = row_wasserstein_cost(g, h)
    distance = float(ot.emd2(g.mu, h.mu, node_cost, numItermax=1_000_000))
    return i, j, distance


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def compute_distance_matrix(
    descriptors: list[GraphDescriptor], output_path: Path, *, workers: int, checkpoint_every: int
) -> np.ndarray:
    global _DESCRIPTORS
    _DESCRIPTORS = descriptors
    n_graphs = len(descriptors)
    if output_path.exists():
        distances = np.load(output_path)
        if distances.shape != (n_graphs, n_graphs):
            raise ValueError(f"checkpoint has shape {distances.shape}, expected {(n_graphs, n_graphs)}")
    else:
        distances = np.full((n_graphs, n_graphs), np.nan, dtype=np.float64)
        np.fill_diagonal(distances, 0.0)

    tasks = [
        (i, j)
        for i in range(n_graphs)
        for j in range(i + 1, n_graphs)
        if not np.isfinite(distances[i, j])
    ]
    total_pairs = n_graphs * (n_graphs - 1) // 2
    completed_before = total_pairs - len(tasks)
    print(json.dumps({"event": "distance_start", "graphs": n_graphs, "pairs_total": total_pairs,
                      "pairs_resumed": completed_before, "workers": workers}), flush=True)
    started = time.time()
    if tasks:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            for offset, (i, j, value) in enumerate(pool.imap_unordered(wllb_pair, tasks, chunksize=64), start=1):
                distances[i, j] = value
                distances[j, i] = value
                if offset % checkpoint_every == 0:
                    atomic_save_npy(output_path, distances)
                    elapsed = time.time() - started
                    done = completed_before + offset
                    print(json.dumps({"event": "distance_progress", "pairs_done": done,
                                      "pairs_total": total_pairs, "elapsed_sec": elapsed,
                                      "pairs_per_sec": offset / max(elapsed, 1e-9)}), flush=True)
    atomic_save_npy(output_path, distances)
    if not np.all(np.isfinite(distances)):
        raise RuntimeError("distance matrix contains non-finite entries after computation")
    print(json.dumps({"event": "distance_done", "elapsed_sec": time.time() - started,
                      "min": float(distances.min()), "max": float(distances.max())}), flush=True)
    return distances


def psd_clip(kernel: np.ndarray, tolerance: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    symmetric = 0.5 * (kernel + kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, 0.0)
    corrected = (eigenvectors * clipped[None, :]) @ eigenvectors.T
    positive = eigenvalues > tolerance
    projector = eigenvectors[:, positive] @ eigenvectors[:, positive].T
    return corrected, projector, eigenvalues


def cv_score_precomputed(kernel: np.ndarray, labels: np.ndarray, c_value: float, splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
    scores: list[float] = []
    for train, valid in splits:
        model = SVC(kernel="precomputed", C=c_value, max_iter=-1)
        model.fit(kernel[np.ix_(train, train)], labels[train])
        prediction = model.predict(kernel[np.ix_(valid, train)])
        scores.append(float(accuracy_score(labels[valid], prediction)))
    return float(np.mean(scores))


def select_raw(
    distance_train: np.ndarray, labels: np.ndarray, gammas: np.ndarray, c_values: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    best: dict | None = None
    for gamma in gammas:
        kernel = np.exp(-float(gamma) * distance_train)
        for c_value in c_values:
            score = cv_score_precomputed(kernel, labels, float(c_value), splits)
            row = {"gamma": float(gamma), "C": float(c_value), "cv_accuracy": score}
            if best is None or score > best["cv_accuracy"]:
                best = row
    assert best is not None
    return best


def select_wtk(
    distance_train: np.ndarray, labels: np.ndarray, gammas: np.ndarray, c_values: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    best: dict | None = None
    for gamma in gammas:
        raw_kernel = np.exp(-float(gamma) * distance_train)
        corrected, _, eigenvalues = psd_clip(raw_kernel)
        negative = eigenvalues[eigenvalues < -1e-8]
        spectral = {
            "min_eigenvalue": float(eigenvalues[0]),
            "negative_count": int(len(negative)),
            "negative_mass_fraction": float(np.abs(negative).sum() / max(np.abs(eigenvalues).sum(), 1e-15)),
        }
        for c_value in c_values:
            score = cv_score_precomputed(corrected, labels, float(c_value), splits)
            row = {"gamma": float(gamma), "C": float(c_value), "cv_accuracy": score, **spectral}
            if best is None or score > best["cv_accuracy"]:
                best = row
    assert best is not None
    return best


def prediction_payload(labels: np.ndarray, prediction: np.ndarray) -> dict:
    classes, counts = np.unique(prediction, return_counts=True)
    payload = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "confusion_matrix": confusion_matrix(labels, prediction).tolist(),
        "prediction_counts": {str(int(k)): int(v) for k, v in zip(classes, counts)},
    }
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:
        flipped = np.where(prediction == unique_labels[0], unique_labels[1], unique_labels[0])
        payload["binary_flipped_accuracy"] = float(accuracy_score(labels, flipped))
    return payload


def evaluate(
    distance: np.ndarray,
    labels: np.ndarray,
    *,
    outer_splits: int,
    raw_inner_splits: int,
    wtk_inner_splits: int,
    seed: int,
) -> dict:
    paper_gammas = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
    paper_cs = np.asarray([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
    wtk_gammas = np.logspace(-4, 1, num=6)
    wtk_cs = np.logspace(-3, 5, num=9)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for fold, (train, test) in enumerate(outer.split(distance, labels)):
        d_train = distance[np.ix_(train, train)]
        d_test = distance[np.ix_(test, train)]
        y_train = labels[train]
        y_test = labels[test]
        raw_inner = StratifiedKFold(n_splits=raw_inner_splits, shuffle=False)
        raw_splits = list(raw_inner.split(d_train, y_train))
        wtk_inner = StratifiedKFold(n_splits=wtk_inner_splits, shuffle=False)
        wtk_splits = list(wtk_inner.split(d_train, y_train))

        raw_best = select_raw(d_train, y_train, paper_gammas, paper_cs, raw_splits)
        raw_train = np.exp(-raw_best["gamma"] * d_train)
        raw_test = np.exp(-raw_best["gamma"] * d_test)
        raw_model = SVC(kernel="precomputed", C=raw_best["C"], max_iter=-1)
        raw_model.fit(raw_train, y_train)
        raw_prediction = raw_model.predict(raw_test)
        rows.append({"fold": fold, "method": "standard_svm_paper_grid", **raw_best,
                     **prediction_payload(y_test, raw_prediction)})

        for method_name, gammas, c_values in [
            ("wtk_released_default_grid", wtk_gammas, wtk_cs),
            ("wtk_paper_grid", paper_gammas, paper_cs),
        ]:
            best = select_wtk(d_train, y_train, gammas, c_values, wtk_splits)
            train_raw = np.exp(-best["gamma"] * d_train)
            test_raw = np.exp(-best["gamma"] * d_test)
            train_corrected, projector, eigenvalues = psd_clip(train_raw)
            model = SVC(kernel="precomputed", C=best["C"], max_iter=-1)
            model.fit(train_corrected, y_train)

            mismatch_prediction = model.predict(test_raw)
            mismatch = prediction_payload(y_test, mismatch_prediction)
            rows.append({"fold": fold, "method": method_name + "_raw_test", **best, **mismatch,
                         "test_projection_relative_frobenius": float(
                             np.linalg.norm(test_raw - test_raw @ projector, ord="fro")
                             / max(np.linalg.norm(test_raw, ord="fro"), 1e-15)
                         )})

            projected_test = test_raw @ projector
            projected_prediction = model.predict(projected_test)
            rows.append({"fold": fold, "method": method_name + "_projected_test", **best,
                         **prediction_payload(y_test, projected_prediction),
                         "test_projection_relative_frobenius": float(
                             np.linalg.norm(test_raw - projected_test, ord="fro")
                             / max(np.linalg.norm(test_raw, ord="fro"), 1e-15)
                         )})

        print(json.dumps({"event": "evaluation_fold", "fold": fold,
                          "rows": [row for row in rows if row["fold"] == fold]}), flush=True)

    aggregate: dict[str, dict] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        accuracies = np.asarray([row["accuracy"] for row in method_rows])
        aggregate[method] = {
            "mean_accuracy": float(accuracies.mean()),
            "std_accuracy": float(accuracies.std()),
            "fold_accuracies": accuracies.tolist(),
        }
        flipped = [row.get("binary_flipped_accuracy") for row in method_rows]
        if all(value is not None for value in flipped):
            aggregate[method]["mean_binary_flipped_accuracy"] = float(np.mean(flipped))
    return {"rows": rows, "aggregate": aggregate}


def evaluate_one_nearest_neighbor(distance: np.ndarray, labels: np.ndarray, repetitions: int = 10) -> dict:
    accuracies: list[float] = []
    graph_indices = np.arange(len(labels))
    for seed in range(repetitions):
        train, test = train_test_split(graph_indices, test_size=0.1, random_state=seed)
        model = KNeighborsClassifier(n_neighbors=1, metric="precomputed")
        model.fit(distance[np.ix_(train, train)], labels[train])
        prediction = model.predict(distance[np.ix_(test, train)])
        accuracies.append(float(accuracy_score(labels[test], prediction)))
    values = np.asarray(accuracies)
    return {
        "repetitions": repetitions,
        "mean_accuracy": float(values.mean()),
        "std_accuracy": float(values.std()),
        "accuracies": values.tolist(),
    }


def repo_f2_static_audit(adjacencies: list[np.ndarray]) -> dict:
    retained_fractions = []
    for adjacency in adjacencies:
        degrees = adjacency.sum(axis=1).astype(int)
        retained_fractions.append(len(np.unique(degrees)) / len(degrees))
    return {
        "released_helper_membership_test": "tests integer degree but stores degree + 1/n",
        "consequence": "repeated-degree entries are overwritten; equal-size second-graph keys can overwrite first-graph keys",
        "mean_unique_degree_fraction": float(np.mean(retained_fractions)),
        "min_unique_degree_fraction": float(np.min(retained_fractions)),
        "max_unique_degree_fraction": float(np.max(retained_fractions)),
    }


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    scalar_keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="IMDB-BINARY")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--q", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--raw-inner-splits", type=int, default=3)
    parser.add_argument("--wtk-inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adjacencies, labels = parse_tu_dataset(args.dataset_dir, args.dataset_name)
    descriptors = [make_descriptor(adjacency, q=args.q, k=args.k) for adjacency in adjacencies]
    mapping_audit = repo_f2_static_audit(adjacencies)
    distance_path = args.output_dir / f"{args.dataset_name}_wllb_f2_k{args.k}.npy"
    distance = compute_distance_matrix(descriptors, distance_path, workers=args.workers,
                                       checkpoint_every=args.checkpoint_every)
    evaluation = evaluate(
        distance,
        labels,
        outer_splits=args.outer_splits,
        raw_inner_splits=args.raw_inner_splits,
        wtk_inner_splits=args.wtk_inner_splits,
        seed=args.seed,
    )
    nearest_neighbor = evaluate_one_nearest_neighbor(distance, labels)
    summary = {
        "schema": "wl-distance-ksvm-diagnostic-v1",
        "dataset": args.dataset_name,
        "graph_count": len(labels),
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "q": args.q,
        "k": args.k,
        "label_function": "f2(G,v)=degree_G(v)+1/|V_G|",
        "distance_definition": "paper-faithful WLLB using exact EMD",
        "seed": args.seed,
        "outer_splits": args.outer_splits,
        "raw_inner_splits": args.raw_inner_splits,
        "wtk_inner_splits": args.wtk_inner_splits,
        "released_repo_f2_audit": mapping_audit,
        "one_nearest_neighbor": nearest_neighbor,
        "evaluation": evaluation,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rows_csv(args.output_dir / "fold_metrics.csv", evaluation["rows"])
    print(json.dumps({"event": "complete", "summary": str(summary_path),
                      "aggregate": evaluation["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
