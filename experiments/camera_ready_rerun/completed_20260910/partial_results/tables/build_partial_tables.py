#!/usr/bin/env python3
"""Build paper-style partial rerun tables from synced, completed artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TABLE_DIR = Path(__file__).resolve().parent
ROOT = TABLE_DIR.parent
OUTPUT_PDF = Path("/Users/mozhengchao/research_os/output/pdf/wl_distance_partial_results_tables_2026-09-10.pdf")

DATASETS = [
    ("MUTAG", "mutag"),
    ("PROTEINS", "proteins"),
    ("PTC-FM", "ptc-fm"),
    ("PTC-MR", "ptc-mr"),
    ("IMDB-B", "imdb-binary"),
    ("IMDB-M", "imdb-multi"),
    ("COX2", "cox2"),
]

F2_NAME = "f2(G,v)=degree_G(v)+1/|V_G|"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value(mean: float, std: float) -> tuple[str, float, float]:
    return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}", mean, std


def pending() -> tuple[str, None, None]:
    return "\\textnormal{\\textit{pending}}", None, None


records: list[dict] = []


def record(
    table: str,
    family: str,
    classifier: str,
    k: int | None,
    label_mode: str | None,
    dataset: str,
    source: Path | None,
    mean: float | None,
    std: float | None,
) -> str:
    if source is None or mean is None or std is None:
        cell, mean, std = pending()
        status = "pending"
        source_text = ""
        source_digest = ""
    else:
        cell, mean, std = value(mean, std)
        status = "finished"
        source_text = str(source.relative_to(ROOT))
        source_digest = sha256(source)
    records.append(
        {
            "table": table,
            "family": family,
            "classifier": classifier,
            "k": "" if k is None else k,
            "label_mode": "" if label_mode is None else label_mode,
            "dataset": dataset,
            "status": status,
            "mean_accuracy": "" if mean is None else f"{mean:.17g}",
            "std_accuracy": "" if std is None else f"{std:.17g}",
            "source_path": source_text,
            "source_sha256": source_digest,
        }
    )
    return cell


def standard_summary(kind: str, label_mode: str, slug: str, k: int) -> Path:
    if kind == "dwl":
        return ROOT / "standard_dwl" / f"{slug}_{label_mode}" / f"summary_k{k}.json"
    return ROOT / f"standard_wllb_{label_mode}" / f"{slug}_k{k}" / "summary.json"


def genuine_summary(kind: str, label_mode: str, slug: str, k: int) -> Path:
    return ROOT / f"loosli_{kind}" / f"{slug}_{label_mode}_k{k}" / "summary.json"


def get_standard(kind: str, label_mode: str, slug: str, k: int, metric: str):
    path = standard_summary(kind, label_mode, slug, k)
    if not path.exists():
        return None, None, None
    data = load(path)
    if label_mode == "f2":
        assert data["label_function"] == F2_NAME, path
    if metric == "1nn":
        aggregate = data["one_nearest_neighbor"]
    else:
        aggregate = data["evaluation"]["aggregate"]["standard_svm_paper_grid"]
    return path, aggregate["mean_accuracy"], aggregate["std_accuracy"]


def get_genuine(kind: str, label_mode: str, slug: str, k: int):
    path = genuine_summary(kind, label_mode, slug, k)
    if not path.exists():
        return None, None, None
    data = load(path)
    assert data["algorithm"]["name"] == "full-rank KSVM", path
    assert "original indefinite" in data["algorithm"]["test_kernel"], path
    assert data["matrix_checks"]["symmetry_error"] <= 1e-10, path
    assert data["matrix_checks"]["diagonal_error"] <= 1e-10, path
    if label_mode == "f2":
        assert data["paper_settings"]["label_function"] == F2_NAME, path
        source = data["source"]
        assert source["source_summary"]["label_function"] == F2_NAME, path
        expected = standard_summary(kind, label_mode, slug, k)
        assert expected.exists(), expected
        assert source["source_summary_sha256"] == sha256(expected), path
    aggregate = data["aggregate"]
    return path, aggregate["mean_accuracy"], aggregate["std_accuracy"]


def baseline(dataset_slug: str, field: str, subfield: str | None = None):
    path = ROOT / "baselines" / dataset_slug / "summary.json"
    if not path.exists():
        return None, None, None
    data = load(path)[field]
    if subfield:
        data = data[subfield]
    return path, data["mean_accuracy"], data["std_accuracy"]


def row_label(kind: str, k: int, label_mode: str, classifier: str = "standard") -> str:
    symbol = "\\mathrm{WL}" if kind == "dwl" else "\\mathrm{WLLB}"
    suffix = ", K" if classifier == "ksvm" else ""
    label_tex = "f_1" if label_mode == "f1" else "f_2"
    return f"$d_{{{symbol}}}^{{\\scriptscriptstyle{{({k})}}}}$, ${label_tex}${suffix}"


def make_row(table: str, row_name: str, getter, family: str, classifier: str, k, label_mode) -> str:
    cells = []
    for dataset, slug in DATASETS:
        path, mean, std = getter(slug)
        cells.append(record(table, family, classifier, k, label_mode, dataset, path, mean, std))
    return row_name + " & " + " & ".join(cells) + "\\\\"


def table_header() -> list[str]:
    return [
        "\\begin{center}",
        "\\begin{scriptsize}",
        "\\begin{sc}",
        "\\setlength{\\tabcolsep}{3.7pt}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & MUTAG & PROTEINS & PTC-FM & PTC-MR & IMDB-B & IMDB-M & COX2\\\\",
    ]


def table_footer() -> list[str]:
    return [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{sc}",
        "\\end{scriptsize}",
        "\\end{center}",
    ]


def build_nn_table(snapshot: str) -> str:
    lines = [
        "\\begin{table*}[htbp!]",
        "\\caption{\\textbf{Partial 1-Nearest Neighbor rerun accuracy.} "
        f"Paper-layout snapshot at {snapshot}. Values are mean $\\pm$ standard deviation in percent; "
        "\\textit{pending} cells have not completed.}",
        "\\label{tab:partial-rerun-nn}",
        *table_header(),
    ]
    for kind in ["dwl", "wllb"]:
        for label_mode in ["f1", "f2"]:
            lines.append("\\midrule")
            for k in range(1, 5):
                getter = lambda slug, kind=kind, label_mode=label_mode, k=k: get_standard(
                    kind, label_mode, slug, k, "1nn"
                )
                lines.append(
                    make_row(
                        "1NN",
                        row_label(kind, k, label_mode),
                        getter,
                        kind,
                        "1NN",
                        k,
                        label_mode,
                    )
                )
    lines.append("\\midrule")
    lines.append(
        make_row(
            "1NN",
            "WWL",
            lambda slug: baseline(slug, "wwl_1nn_manuscript", "best"),
            "WWL",
            "1NN",
            None,
            None,
        )
    )
    lines.extend(table_footer())
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def build_svm_table(snapshot: str) -> str:
    lines = [
        "\\begin{table*}[htbp!]",
        "\\caption{\\textbf{Partial SVM rerun accuracy.} "
        f"Paper-layout snapshot at {snapshot}. K denotes the full-rank Algorithm~1 KSVM; "
        "values are mean $\\pm$ standard deviation in percent, and \\textit{pending} cells have not completed.}",
        "\\label{tab:partial-rerun-svm}",
        *table_header(),
    ]
    for kind in ["dwl", "wllb"]:
        for label_mode in ["f1", "f2"]:
            for classifier in ["standard", "ksvm"]:
                lines.append("\\midrule")
                for k in range(1, 5):
                    if classifier == "standard":
                        getter = lambda slug, kind=kind, label_mode=label_mode, k=k: get_standard(
                            kind, label_mode, slug, k, "svm"
                        )
                    else:
                        getter = lambda slug, kind=kind, label_mode=label_mode, k=k: get_genuine(
                            kind, label_mode, slug, k
                        )
                    lines.append(
                        make_row(
                            "SVM",
                            row_label(kind, k, label_mode, classifier),
                            getter,
                            kind,
                            classifier,
                            k,
                            label_mode,
                        )
                    )
    lines.append("\\midrule")
    for name, field in [("WWL", "wwl_svm_manuscript"), ("WL", "wl_svm"), ("WL-OA", "wloa_svm")]:
        lines.append(
            make_row(
                "SVM",
                name,
                lambda slug, field=field: baseline(slug, field),
                name,
                "SVM",
                None,
                None,
            )
        )
    lines.extend(table_footer())
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> None:
    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nn = build_nn_table(snapshot)
    svm = build_svm_table(snapshot)
    (TABLE_DIR / "partial_1nn_table.tex").write_text(nn)
    (TABLE_DIR / "partial_svm_table.tex").write_text(svm)

    with (TABLE_DIR / "partial_table_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    counts: dict[str, dict[str, int]] = {}
    for table in ["1NN", "SVM"]:
        subset = [r for r in records if r["table"] == table]
        counts[table] = {
            "finished_cells": sum(r["status"] == "finished" for r in subset),
            "pending_cells": sum(r["status"] == "pending" for r in subset),
            "total_cells": len(subset),
        }
    manifest = {
        "schema": "wl-distance-paper-style-partial-tables-v1",
        "snapshot_utc": snapshot,
        "counts": counts,
        "selection": {
            "standard_svm": "evaluation.aggregate.standard_svm_paper_grid",
            "ksvm": "full-rank Loosli Algorithm 1 aggregate; original indefinite test kernel",
            "baseline": "manuscript-stated protocol fields",
            "rounding": "accuracy fractions multiplied by 100 and rounded to one decimal",
            "pending": "no completed summary artifact was present in the synchronized snapshot",
        },
        "f2_provenance_check": "Every completed genuine WLLB-f2 KSVM cell matched the SHA-256 of its synchronized corrected source summary and used degree_G(v)+1/|V_G|.",
        "output_pdf": str(OUTPUT_PDF),
    }
    (TABLE_DIR / "partial_table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    wrapper = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.55in]{geometry}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=5pt}
\renewcommand{\arraystretch}{1.02}
\pagestyle{plain}
\begin{document}
\input{partial_1nn_table.tex}
\clearpage
\input{partial_svm_table.tex}
\end{document}
"""
    (TABLE_DIR / "paper_style_partial_tables_preview.tex").write_text(wrapper)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
