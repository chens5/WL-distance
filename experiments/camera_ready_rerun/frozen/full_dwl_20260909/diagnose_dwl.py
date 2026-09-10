#!/usr/bin/env python3
"""Compute paper-faithful full WL distances and reuse the diagnostic evaluators."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import ot

from diagnose_wllb_ksvm import (
    GraphDescriptor,
    evaluate,
    evaluate_one_nearest_neighbor,
    make_descriptor,
    parse_tu_dataset,
    row_wasserstein_cost,
    write_rows_csv,
)


_DESCRIPTORS: list[GraphDescriptor] = []
_MAX_K = 4


def _positive_support(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(row > 0)
    weights = row[indices]
    return indices, weights / weights.sum()


def dwl_pair(task: tuple[int, int]) -> tuple[int, int, np.ndarray, float]:
    pair_started = time.perf_counter()
    graph_i, graph_j = task
    g = _DESCRIPTORS[graph_i]
    h = _DESCRIPTORS[graph_j]

    # At depth one, the recursive transport cost is one-dimensional because
    # the ground cost is absolute label difference.
    current = row_wasserstein_cost(g, h)
    distances = [float(ot.emd2(g.mu, h.mu, current, numItermax=1_000_000))]

    g_supports = [_positive_support(row) for row in g.transition]
    h_supports = [_positive_support(row) for row in h.transition]
    for _depth in range(2, _MAX_K + 1):
        next_cost = np.empty((g.n, h.n), dtype=np.float64)
        for node_i, (support_i, weights_i) in enumerate(g_supports):
            for node_j, (support_j, weights_j) in enumerate(h_supports):
                ground = current[np.ix_(support_i, support_j)]
                next_cost[node_i, node_j] = ot.emd2(
                    weights_i,
                    weights_j,
                    ground,
                    numItermax=1_000_000,
                )
        current = next_cost
        distances.append(float(ot.emd2(g.mu, h.mu, current, numItermax=1_000_000)))
    return graph_i, graph_j, np.asarray(distances), time.perf_counter() - pair_started


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def compute_all_depths(
    descriptors: list[GraphDescriptor],
    output_dir: Path,
    dataset_name: str,
    label_mode: str,
    *,
    max_k: int,
    workers: int,
    checkpoint_every: int,
    benchmark_pairs: int | None,
    benchmark_seed: int,
) -> tuple[list[np.ndarray], dict]:
    global _DESCRIPTORS, _MAX_K
    _DESCRIPTORS = descriptors
    _MAX_K = max_k
    graph_count = len(descriptors)
    paths = [
        output_dir / f"{dataset_name}_dwl_{label_mode}_k{k}.npy"
        for k in range(1, max_k + 1)
    ]
    matrices = []
    for path in paths:
        if path.exists():
            matrix = np.load(path)
        else:
            matrix = np.full((graph_count, graph_count), np.nan, dtype=np.float64)
            np.fill_diagonal(matrix, 0.0)
        matrices.append(matrix)

    tasks = [
        (i, j)
        for i in range(graph_count)
        for j in range(i + 1, graph_count)
        if not all(np.isfinite(matrix[i, j]) for matrix in matrices)
    ]
    if benchmark_pairs is not None:
        generator = np.random.default_rng(benchmark_seed)
        selection = generator.choice(len(tasks), size=min(benchmark_pairs, len(tasks)), replace=False)
        tasks = [tasks[int(index)] for index in selection]
    print(json.dumps({"event": "distance_start", "graphs": graph_count,
                      "pairs_selected": len(tasks), "max_k": max_k,
                      "workers": workers, "benchmark_pairs": benchmark_pairs}), flush=True)

    started = time.time()
    pair_seconds: list[float] = []
    if tasks:
        context = mp.get_context("fork")
        with context.Pool(processes=workers) as pool:
            for offset, (i, j, values, elapsed_pair) in enumerate(
                pool.imap_unordered(dwl_pair, tasks, chunksize=1), start=1
            ):
                pair_seconds.append(elapsed_pair)
                for depth, value in enumerate(values):
                    matrices[depth][i, j] = value
                    matrices[depth][j, i] = value
                if benchmark_pairs is None and offset % checkpoint_every == 0:
                    for path, matrix in zip(paths, matrices):
                        atomic_save(path, matrix)
                if offset % max(1, min(100, checkpoint_every)) == 0:
                    wall = time.time() - started
                    print(json.dumps({"event": "distance_progress", "pairs_done": offset,
                                      "pairs_selected": len(tasks), "wall_sec": wall,
                                      "pairs_per_wall_sec": offset / max(wall, 1e-9)}), flush=True)

    if benchmark_pairs is None:
        for path, matrix in zip(paths, matrices):
            atomic_save(path, matrix)
        if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
            raise RuntimeError("one or more full-WL matrices remain incomplete")

    wall_seconds = time.time() - started
    timing = {
        "selected_pairs": len(tasks),
        "wall_seconds": wall_seconds,
        "pairs_per_wall_second": len(tasks) / max(wall_seconds, 1e-9),
        "mean_worker_pair_seconds": float(np.mean(pair_seconds)) if pair_seconds else 0.0,
        "median_worker_pair_seconds": float(np.median(pair_seconds)) if pair_seconds else 0.0,
        "max_worker_pair_seconds": float(np.max(pair_seconds)) if pair_seconds else 0.0,
    }
    print(json.dumps({"event": "distance_done", **timing}), flush=True)
    return matrices, timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-mode", choices=("f1", "f2"), required=True)
    parser.add_argument("--q", type=float, default=0.6)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--benchmark-pairs", type=int)
    parser.add_argument("--benchmark-seed", type=int, default=20260909)
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--raw-inner-splits", type=int, default=10)
    parser.add_argument("--wtk-inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adjacencies, labels = parse_tu_dataset(args.dataset_dir, args.dataset_name)
    descriptors = [
        make_descriptor(
            adjacency,
            q=args.q,
            k=1,
            label_mode=args.label_mode,
        )
        for adjacency in adjacencies
    ]
    matrices, timing = compute_all_depths(
        descriptors,
        args.output_dir,
        args.dataset_name,
        args.label_mode,
        max_k=args.max_k,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        benchmark_pairs=args.benchmark_pairs,
        benchmark_seed=args.benchmark_seed,
    )
    timing_path = args.output_dir / "timing.json"
    timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.benchmark_pairs is not None:
        return

    for k, matrix in enumerate(matrices, start=1):
        evaluation = evaluate(
            matrix,
            labels,
            outer_splits=args.outer_splits,
            raw_inner_splits=args.raw_inner_splits,
            wtk_inner_splits=args.wtk_inner_splits,
            seed=args.seed,
        )
        nearest_neighbor = evaluate_one_nearest_neighbor(matrix, labels)
        summary = {
            "schema": "wl-distance-full-diagnostic-v1",
            "dataset": args.dataset_name,
            "graph_count": len(labels),
            "class_counts": {str(int(key)): int(value) for key, value in zip(*np.unique(labels, return_counts=True))},
            "q": args.q,
            "k": k,
            "label_function": (
                "f1(G,v)=degree_G(v)"
                if args.label_mode == "f1"
                else "f2(G,v)=degree_G(v)+1/|V_G|"
            ),
            "distance_definition": "paper-faithful full WL distance using exact EMD",
            "seed": args.seed,
            "outer_splits": args.outer_splits,
            "raw_inner_splits": args.raw_inner_splits,
            "wtk_inner_splits": args.wtk_inner_splits,
            "distance_timing": timing,
            "one_nearest_neighbor": nearest_neighbor,
            "evaluation": evaluation,
        }
        summary_path = args.output_dir / f"summary_k{k}.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_rows_csv(args.output_dir / f"fold_metrics_k{k}.csv", evaluation["rows"])
        print(json.dumps({"event": "evaluation_complete", "k": k,
                          "summary": str(summary_path),
                          "aggregate": evaluation["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
