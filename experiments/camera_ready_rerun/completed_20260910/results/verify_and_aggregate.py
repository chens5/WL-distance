#!/usr/bin/env python3
"""Verify the completed 112-setting Loosli-KSVM rerun and aggregate its results."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REMOTE = HERE / "remote" / "loosli_ksvm_20260909"
PARTIAL = HERE.parent / "partial_results"
DATASETS = [
    ("MUTAG", "MUTAG", "mutag"),
    ("PROTEINS", "PROTEINS", "proteins"),
    ("PTC-FM", "PTC_FM", "ptc-fm"),
    ("PTC-MR", "PTC_MR", "ptc-mr"),
    ("IMDB-B", "IMDB-BINARY", "imdb-binary"),
    ("IMDB-M", "IMDB-MULTI", "imdb-multi"),
    ("COX2", "COX2", "cox2"),
]
F2 = "f2(G,v)=degree_G(v)+1/|V_G|"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_summary(kind: str, mode: str, slug: str, depth: int) -> Path:
    if kind == "dwl":
        return PARTIAL / "standard_dwl" / f"{slug}_{mode}" / f"summary_k{depth}.json"
    return PARTIAL / f"standard_wllb_{mode}" / f"{slug}_k{depth}" / "summary.json"


def main() -> None:
    rows: list[dict] = []
    errors: list[str] = []
    total_pair_models = 0
    converged_pair_models = 0
    negative_pair_models = 0
    max_dual_equality_error = 0.0
    max_auxiliary_decision_error = 0.0
    max_original_decision_error = 0.0
    max_symmetry_error = 0.0
    max_diagonal_error = 0.0
    min_distance = float("inf")
    csv_fold_rows = 0

    for kind in ("dwl", "wllb"):
        for display, canonical, slug in DATASETS:
            for mode in ("f1", "f2"):
                for depth in range(1, 5):
                    output_dir = REMOTE / "outputs" / kind / f"{slug}_{mode}_k{depth}"
                    summary_path = output_dir / "summary.json"
                    fold_path = output_dir / "fold_metrics.csv"
                    if not summary_path.exists() or not fold_path.exists():
                        errors.append(f"missing output: {output_dir}")
                        continue
                    summary = json.loads(summary_path.read_text())
                    folds = summary.get("folds", [])
                    with fold_path.open(newline="", encoding="utf-8") as stream:
                        csv_rows = list(csv.DictReader(stream))
                    csv_fold_rows += len(csv_rows)
                    if len(folds) != 10 or len(csv_rows) != 10:
                        errors.append(f"wrong fold count: {summary_path}")
                    json_acc = [float(fold["accuracy"]) for fold in folds]
                    csv_acc = [float(fold["accuracy"]) for fold in csv_rows]
                    if json_acc != csv_acc:
                        errors.append(f"fold CSV/JSON mismatch: {summary_path}")

                    algorithm = summary.get("algorithm", {})
                    if algorithm.get("name") != "full-rank KSVM":
                        errors.append(f"wrong algorithm: {summary_path}")
                    if "original indefinite" not in algorithm.get("test_kernel", ""):
                        errors.append(f"wrong test-kernel path: {summary_path}")
                    settings = summary.get("paper_settings", {})
                    if settings.get("dataset") != canonical:
                        errors.append(f"dataset mismatch: {summary_path}")
                    if settings.get("distance_kind") != kind or settings.get("label_mode") != mode:
                        errors.append(f"setting mismatch: {summary_path}")
                    if int(settings.get("depth", -1)) != depth:
                        errors.append(f"depth mismatch: {summary_path}")
                    if mode == "f2" and settings.get("label_function") != F2:
                        errors.append(f"wrong f2 label: {summary_path}")

                    source = source_summary(kind, mode, slug, depth)
                    if not source.exists():
                        errors.append(f"missing synchronized source summary: {source}")
                    elif summary["source"]["source_summary_sha256"] != sha256(source):
                        errors.append(f"source SHA mismatch: {summary_path}")
                    if mode == "f2" and summary["source"]["source_summary"].get("label_function") != F2:
                        errors.append(f"wrong source f2 provenance: {summary_path}")

                    checks = summary.get("matrix_checks", {})
                    max_symmetry_error = max(max_symmetry_error, float(checks.get("symmetry_error", float("inf"))))
                    max_diagonal_error = max(max_diagonal_error, float(checks.get("diagonal_error", float("inf"))))
                    min_distance = min(min_distance, float(checks.get("minimum", float("-inf"))))
                    if max_symmetry_error > 1e-8 or max_diagonal_error > 1e-8 or min_distance < -1e-8:
                        errors.append(f"invalid distance checks: {summary_path}")

                    vote_ties = 0
                    setting_pair_models = 0
                    setting_negative_pair_models = 0
                    for fold in folds:
                        vote_ties += int(fold.get("ovo_voting_ties", 0))
                        for model in fold.get("model_diagnostics", {}).get("pair_models", []):
                            total_pair_models += 1
                            setting_pair_models += 1
                            fit_status = int(model.get("fit_status", -1))
                            converged_pair_models += int(fit_status == 0)
                            if fit_status != 0:
                                errors.append(f"nonconverged pair model: {summary_path} fold {fold['fold']}")
                            negative = int(model.get("negative_eigenvalue_count", 0)) > 0
                            negative_pair_models += int(negative)
                            setting_negative_pair_models += int(negative)
                            max_dual_equality_error = max(
                                max_dual_equality_error,
                                float(model.get("dual_equality_error", float("inf"))),
                            )
                            max_auxiliary_decision_error = max(
                                max_auxiliary_decision_error,
                                float(model.get("max_auxiliary_decision_error", float("inf"))),
                            )
                            max_original_decision_error = max(
                                max_original_decision_error,
                                float(model.get("max_original_decision_error", float("inf"))),
                            )

                    aggregate = summary["aggregate"]
                    rows.append(
                        {
                            "distance_kind": kind,
                            "dataset": display,
                            "dataset_slug": slug,
                            "label_mode": mode,
                            "depth": depth,
                            "mean_accuracy": float(aggregate["mean_accuracy"]),
                            "std_accuracy": float(aggregate["std_accuracy"]),
                            "mean_accuracy_percent": 100.0 * float(aggregate["mean_accuracy"]),
                            "std_accuracy_percent": 100.0 * float(aggregate["std_accuracy"]),
                            "folds": len(folds),
                            "pair_models": setting_pair_models,
                            "pair_models_with_negative_eigenvalues": setting_negative_pair_models,
                            "ovo_vote_ties": vote_ties,
                            "summary_path": str(summary_path.relative_to(HERE)),
                            "summary_sha256": sha256(summary_path),
                            "fold_metrics_path": str(fold_path.relative_to(HERE)),
                            "fold_metrics_sha256": sha256(fold_path),
                        }
                    )

    if max_dual_equality_error > 1e-8:
        errors.append(f"dual equality error too large: {max_dual_equality_error}")
    if max_auxiliary_decision_error > 1e-7:
        errors.append(f"auxiliary decision identity error too large: {max_auxiliary_decision_error}")
    if max_original_decision_error > 1e-7:
        errors.append(f"original decision identity error too large: {max_original_decision_error}")

    csv_path = HERE / "ksvm_results_112_settings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    inventory_rows = []
    for path in sorted((HERE / "remote").rglob("*")):
        if path.is_file():
            inventory_rows.append(
                {
                    "path": str(path.relative_to(HERE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    inventory_path = HERE / "artifact_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory_rows)

    verification = {
        "schema": "wl-distance-loosli-ksvm-verification-v1",
        "status": "pass" if not errors and len(rows) == 112 else "fail",
        "settings_expected": 112,
        "settings_verified": len(rows),
        "fold_rows_verified": csv_fold_rows,
        "pair_models": total_pair_models,
        "converged_pair_models": converged_pair_models,
        "pair_models_with_negative_eigenvalues": negative_pair_models,
        "max_dual_equality_error": max_dual_equality_error,
        "max_auxiliary_decision_error": max_auxiliary_decision_error,
        "max_original_decision_error": max_original_decision_error,
        "max_distance_symmetry_error": max_symmetry_error,
        "max_distance_diagonal_error": max_diagonal_error,
        "minimum_distance_entry": min_distance,
        "f2_definition": F2,
        "algorithm": "full-rank Loosli Algorithm 1; original indefinite test kernel",
        "errors": errors,
        "synced_artifact_files_in_inventory": len(inventory_rows),
        "outputs": {
            "aggregate_csv": str(csv_path.relative_to(HERE.parent.parent.parent.parent)),
            "artifact_inventory_csv": str(inventory_path.relative_to(HERE.parent.parent.parent.parent)),
        },
    }
    (HERE / "verification_report.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    if verification["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
