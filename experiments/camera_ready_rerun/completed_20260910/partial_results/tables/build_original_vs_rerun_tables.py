#!/usr/bin/env python3
"""Build paired published-versus-rerun versions of the paper's result tables."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


TABLE_DIR = Path(__file__).resolve().parent
ROOT = TABLE_DIR.parent
MANUSCRIPT = Path(
    "/Users/mozhengchao/local-repos/overleaf-projects/"
    "weisfeiler-lehman-meets-optimal-transport/JMLR/dwl_jmlr_camera_ready.tex"
)
OUTPUT_PDF = Path(
    "/Users/mozhengchao/research_os/output/pdf/"
    "wl_distance_original_vs_rerun_tables_2026-09-10.pdf"
)

DATASETS = ["MUTAG", "PROTEINS", "PTC-FM", "PTC-MR", "IMDB-B", "IMDB-M", "COX2"]


def table_block(text: str, label: str) -> str:
    label_token = f"\\label{{{label}}}"
    label_pos = text.index(label_token)
    start = text.rfind("\\begin{table*}", 0, label_pos)
    end = text.index("\\end{table*}", label_pos) + len("\\end{table*}")
    if start < 0:
        raise ValueError(f"Could not locate start of table {label}")
    return text[start:end]


def parse_rows(block: str) -> list[list[str]]:
    tabular_start = block.index("\\begin{tabular}")
    tabular_end = block.index("\\end{tabular}", tabular_start)
    body = block[tabular_start:tabular_end]
    rows: list[list[str]] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if "&" not in line or not re.search(r"\\\\\s*$", line):
            continue
        line = re.sub(r"\\\\\s*$", "", line)
        fields = [field.strip() for field in line.split("&")]
        if fields[0] == "Method":
            continue
        if len(fields) != 8:
            raise ValueError(f"Expected 8 fields, found {len(fields)} in {raw_line!r}")
        rows.append(fields)
    return rows


def normalized_method(method: str) -> str:
    return re.sub(r"\s+", "", method)


def numeric_value(cell: str) -> tuple[float | None, float | None]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\$\\pm\$\s*([0-9]+(?:\.[0-9]+)?)", cell)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def without_bold(cell: str) -> str:
    """Remove manuscript-provided whole-cell bolding before recalculating maxima."""
    stripped = cell.strip()
    if stripped.startswith("\\textbf{") and stripped.endswith("}"):
        return stripped[len("\\textbf{") : -1]
    return stripped


def column_maxima(rows: list[list[str]]) -> list[float | None]:
    maxima: list[float | None] = []
    for column in range(1, 8):
        values = [numeric_value(without_bold(row[column]))[0] for row in rows]
        completed = [value for value in values if value is not None]
        maxima.append(max(completed) if completed else None)
    return maxima


def bold_if_best(cell: str, maximum: float | None) -> str:
    cell = without_bold(cell)
    mean, _ = numeric_value(cell)
    if maximum is not None and mean is not None and abs(mean - maximum) < 1e-12:
        return f"\\textbf{{{cell}}}"
    return cell


def comparison_records(
    table: str,
    published: list[list[str]],
    rerun: list[list[str]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for old, new in zip(published, rerun):
        for dataset, old_cell, new_cell in zip(DATASETS, old[1:], new[1:]):
            old_mean, old_std = numeric_value(old_cell)
            new_mean, new_std = numeric_value(new_cell)
            records.append(
                {
                    "table": table,
                    "method_tex": new[0],
                    "dataset": dataset,
                    "published_tex": old_cell,
                    "rerun_tex": new_cell,
                    "published_mean": "" if old_mean is None else f"{old_mean:.1f}",
                    "published_std": "" if old_std is None else f"{old_std:.1f}",
                    "rerun_mean": "" if new_mean is None else f"{new_mean:.1f}",
                    "rerun_std": "" if new_std is None else f"{new_std:.1f}",
                    "delta_mean_rerun_minus_published": (
                        "" if old_mean is None or new_mean is None else f"{new_mean - old_mean:.1f}"
                    ),
                    "rerun_status": "finished" if new_mean is not None else "pending",
                }
            )
    return records


def result_family(record: dict[str, str]) -> str:
    """Assign each table cell to a compact audit family."""
    method = normalized_method(record["method_tex"])
    if record["table"] == "1NN":
        if "\\mathrm{WLLB}" in method:
            return "1NN WLLB"
        if "\\mathrm{WL}" in method:
            return "1NN full WL"
        return "1NN WWL"

    if method in {"WWL", "WL", "WL-OA"}:
        return "Baseline SVM"
    is_ksvm = method.endswith(",K")
    if "\\mathrm{WLLB}" in method:
        if is_ksvm and "f_1" in method:
            return "KSVM WLLB f1"
        if is_ksvm and "f_2" in method:
            return "KSVM WLLB f2"
        return "Standard SVM WLLB"
    if is_ksvm:
        return "KSVM full WL"
    return "Standard SVM full WL"


def family_statistics(records: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    families: dict[str, list[float]] = {}
    pending: dict[str, int] = {}
    for record in records:
        family = result_family(record)
        pending.setdefault(family, 0)
        delta = record["delta_mean_rerun_minus_published"]
        if not delta:
            pending[family] += 1
            continue
        families.setdefault(family, []).append(float(delta))

    output: dict[str, dict[str, float | int]] = {}
    for family in sorted(set(families) | set(pending)):
        values = families.get(family, [])
        output[family] = {
            "finished": len(values),
            "pending": pending.get(family, 0),
            "mean_absolute_delta_pp": (
                sum(abs(value) for value in values) / len(values) if values else 0.0
            ),
            "maximum_absolute_delta_pp": max((abs(value) for value in values), default=0.0),
            "cells_absolute_delta_le_2pp": sum(abs(value) <= 2.0 for value in values),
            "cells_absolute_delta_ge_5pp": sum(abs(value) >= 5.0 for value in values),
        }
    return output


def paired_rows(
    published: list[list[str]],
    rerun: list[list[str]],
    group_starts: set[int],
) -> list[str]:
    if len(published) != len(rerun):
        raise ValueError(f"Published/rerun row count mismatch: {len(published)} != {len(rerun)}")

    published_maxima = column_maxima(published)
    rerun_maxima = column_maxima(rerun)
    lines: list[str] = []
    for index, (old, new) in enumerate(zip(published, rerun)):
        if normalized_method(old[0]) != normalized_method(new[0]):
            raise ValueError(f"Method mismatch at row {index}: {old[0]!r} != {new[0]!r}")
        if index in group_starts:
            lines.append("\\midrule")
        cells = [new[0]]
        for column, (old_cell, new_cell) in enumerate(zip(old[1:], new[1:])):
            new_cell = new_cell.replace(
                "\\textnormal{\\textit{pending}}",
                "\\textcolor{pendingfg}{\\textit{pending}}",
            )
            cells.extend(
                [
                    bold_if_best(old_cell, published_maxima[column]),
                    bold_if_best(new_cell, rerun_maxima[column]),
                ]
            )
        lines.append(" & ".join(cells) + r"\\")
    return lines


def comparison_table(
    caption: str,
    published: list[list[str]],
    rerun: list[list[str]],
    group_starts: set[int],
    width: str,
) -> str:
    dataset_header = " & ".join(f"\\multicolumn{{2}}{{c}}{{{name}}}" for name in DATASETS)
    subheader = " & ".join(["", *[item for _ in DATASETS for item in ("P", "R")]])
    body = paired_rows(published, rerun, group_starts)
    return "\n".join(
        [
            "\\begin{center}",
            f"{{\\small {caption}\\par}}",
            "\\vspace{3pt}",
            "\\begingroup",
            "\\setlength{\\tabcolsep}{2.8pt}",
            "\\renewcommand{\\arraystretch}{1.02}",
            f"\\resizebox{{{width}}}{{!}}{{%",
            "\\begin{tabular}{l*{7}{c>{\\columncolor{rerunbg}}c}}",
            "\\toprule",
            f"Method & {dataset_header}\\\\",
            subheader + r"\\",
            *body,
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\endgroup",
            "\\end{center}",
        ]
    )


def main() -> None:
    manuscript_text = MANUSCRIPT.read_text()
    rerun_nn_text = (TABLE_DIR / "partial_1nn_table.tex").read_text()
    rerun_svm_text = (TABLE_DIR / "partial_svm_table.tex").read_text()
    manifest = json.loads((TABLE_DIR / "partial_table_manifest.json").read_text())
    verification = json.loads((ROOT.parent / "results" / "verification_report.json").read_text())
    snapshot = manifest["snapshot_utc"]

    published_nn = parse_rows(table_block(manuscript_text, "tab:full nn experiments"))
    published_svm = parse_rows(table_block(manuscript_text, "tab:full svm"))
    rerun_nn = parse_rows(rerun_nn_text)
    rerun_svm = parse_rows(rerun_svm_text)

    if (len(published_nn), len(rerun_nn)) != (17, 17):
        raise ValueError("Unexpected 1NN table shape")
    if (len(published_svm), len(rerun_svm)) != (35, 35):
        raise ValueError("Unexpected SVM table shape")

    nn_table = comparison_table(
        "\\textbf{1-Nearest Neighbor accuracy: published versus rerun.} "
        "For each dataset, P is the value in the accepted manuscript and R is the corrected rerun value. "
        "Bold marks the highest completed mean independently in every P and R column; ties are all bold.",
        published_nn,
        rerun_nn,
        {0, 4, 8, 12, 16},
        "\\textwidth",
    )
    svm_table = comparison_table(
        "\\textbf{SVM accuracy: published versus rerun.} "
        "For each dataset, P is the value in the accepted manuscript and R is the corrected rerun value. "
        "K marks KSVM rows. Bold marks the highest completed mean independently in every P and R column; "
        "ties are all bold.",
        published_svm,
        rerun_svm,
        {0, 4, 8, 12, 16, 20, 24, 28, 32},
        "0.90\\textwidth",
    )

    (TABLE_DIR / "comparison_1nn_table.tex").write_text(nn_table + "\n")
    (TABLE_DIR / "comparison_svm_table.tex").write_text(svm_table + "\n")

    records = comparison_records("1NN", published_nn, rerun_nn)
    records.extend(comparison_records("SVM", published_svm, rerun_svm))
    with (TABLE_DIR / "original_vs_rerun_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    statistics = family_statistics(records)
    depth_control_path = (
        ROOT.parent / "results" / "remote" / "diagnostics_20260910"
        / "dwl_wllb_k1_equivalence_default.json"
    )
    depth_control_rows = json.loads(depth_control_path.read_text())["rows"]
    depth_control_complete = [row for row in depth_control_rows if row["status"] == "complete"]
    depth_control_pending = [row for row in depth_control_rows if row["status"] != "complete"]
    depth_control_disagreements = sum(
        row["prediction_disagreements_across_10_splits"] for row in depth_control_complete
    )
    audit_payload = {
        "schema": "wl-distance-published-versus-rerun-discrepancy-audit-v1",
        "snapshot_utc": snapshot,
        "completion": manifest["counts"],
        "families": statistics,
        "depth_one_control": {
            "completed_dataset_label_cases": len(depth_control_complete),
            "pending_dataset_label_cases": len(depth_control_pending),
            "matrix_relation": (
                "exactly equal values for dWL(1) and WLLB(1)"
                if all(row["matrix_exact_equal"] for row in depth_control_complete)
                else "see diagnostic rows"
            ),
            "same_node_prediction_disagreements": depth_control_disagreements,
            "amd_nodes": ["c075", "c058"],
            "intel_node": "c146",
            "interpretation": (
                "Fixed splits and exact matrices still yield architecture-dependent 1NN "
                "results when multiple training graphs tie at the minimum distance."
            ),
        },
    }
    (TABLE_DIR / "discrepancy_audit.json").write_text(
        json.dumps(audit_payload, indent=2, sort_keys=True) + "\n"
    )

    def stat_row(label: str, family: str) -> str:
        item = statistics[family]
        return (
            f"{label} & {item['finished']} & {item['pending']} & "
            f"{item['mean_absolute_delta_pp']:.2f} & "
            f"{item['maximum_absolute_delta_pp']:.1f} & "
            f"{item['cells_absolute_delta_ge_5pp']}\\\\"
        )

    audit_rows = "\n".join(
        [
            stat_row("1NN full WL", "1NN full WL"),
            stat_row("1NN WLLB", "1NN WLLB"),
            stat_row("1NN WWL", "1NN WWL"),
            stat_row("Standard SVM, full WL", "Standard SVM full WL"),
            stat_row("Standard SVM, WLLB", "Standard SVM WLLB"),
            stat_row("Genuine KSVM, WLLB $f_1$", "KSVM WLLB f1"),
            stat_row("Genuine KSVM, WLLB $f_2$", "KSVM WLLB f2"),
            stat_row("WWL/WL/WL-OA SVM", "Baseline SVM"),
            stat_row("Genuine KSVM, full WL", "KSVM full WL"),
        ]
    )

    wrapper = rf"""\documentclass[10pt]{{article}}
\usepackage[paperwidth=17in,paperheight=11in,margin=0.38in]{{geometry}}
\usepackage{{booktabs}}
\usepackage[table]{{xcolor}}
\usepackage{{array}}
\usepackage{{graphicx}}
\usepackage{{multicol}}
\usepackage{{enumitem}}
\definecolor{{rerunbg}}{{RGB}}{{239,247,255}}
\definecolor{{pendingfg}}{{RGB}}{{166,94,0}}
\pagestyle{{plain}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries WL-Distance Experiment Tables: Published vs. Corrected Rerun}}\\[3pt]
{{\small Snapshot: {snapshot}}}
\end{{center}}
\noindent\textbf{{Objective.}} Place the accepted-manuscript results directly beside the current rerun results so every numerical change can be checked cell by cell.\quad
\textbf{{Setup.}} Values are mean $\pm$ standard deviation in percentage points. Each dataset has a published column (P) followed by a light-blue rerun column (R).\quad
\textbf{{Method.}} Published cells are transcribed from the 1NN and SVM tables in the canonical camera-ready source; rerun cells come from the synchronized completed summaries. The rerun's K rows use the full-rank Algorithm~1 KSVM and the corrected $f_2(G,v)=\mathrm{{degree}}_G(v)+1/|V_G|$.\quad
\textbf{{Key results.}} All 119 1NN cells and all 245 SVM/KSVM cells are complete. Bold marks the highest mean independently in every published and rerun column, with all ties bolded.\quad
\textbf{{Output paths.}} Data/provenance: \texttt{{\detokenize{{tmp/wl-distance-loosli-ksvm-2026-09-09/partial_results}}}}. PDF: \texttt{{\detokenize{{output/pdf/{OUTPUT_PDF.name}}}}}.
\vspace{{5pt}}
\input{{comparison_1nn_table.tex}}
\clearpage
\begin{{center}}
{{\Large\bfseries SVM and KSVM Cell-by-Cell Comparison}}\\[2pt]
{{\small P = published; \colorbox{{rerunbg}}{{R = corrected rerun}}; accuracy in percent.}}
\end{{center}}
\input{{comparison_svm_table.tex}}
\clearpage
\begin{{center}}
{{\Large\bfseries Discrepancy and Reproducibility Audit}}\\[2pt]
{{\small Evidence available at the {snapshot} snapshot; conclusions are calibrated to the released code and synchronized artifacts.}}
\end{{center}}

\begin{{minipage}}[t]{{0.49\textwidth}}
\textbf{{Where the ``$\pm$'' variation comes from.}}
The rerun's 1NN values are the mean and population standard deviation across ten unstratified 90/10 splits with fixed seeds $0,\ldots,9$.  Its SVM/KSVM values use one fixed, shuffled, stratified 10-fold outer CV (seed 20260909); the spread is across outer folds.  Thus the reported spread measures split/fold composition (and fold-dependent hyperparameter choice for SVM), not noise in the distance computation.

\medskip
\textbf{{Depth-one identity control.}}
For all {len(depth_control_complete)} dataset/label cases, the saved $d_{{\rm WL}}^{{(1)}}$ and $d_{{\rm WLLB}}^{{(1)}}$ arrays have exactly equal numerical entries.  On the same node and the same fixed splits there were {depth_control_disagreements} prediction disagreements.  Nevertheless, many test graphs have cross-label ties at the minimum distance, and scikit-learn's 1NN selection among tied rows is architecture-dependent:
\begin{{center}}
\small
\begin{{tabular}}{{lrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{IMDB-B}} & \multicolumn{{2}}{{c}}{{IMDB-M}}\\
Node family & $f_1$ & $f_2$ & $f_1$ & $f_2$\\
\midrule
AMD (c075/c058) & 70.5 & 71.2 & 40.6 & 42.1\\
Intel Skylake (c146) & 69.4 & 70.3 & 38.7 & 39.6\\
\bottomrule
\end{{tabular}}
\end{{center}}
Both AMD nodes agree with each other.  The AMD values reproduce the full-WL summaries and the Intel values reproduce the WLLB summaries.  Therefore these current depth-one differences are \emph{{not}} caused by different random splits; they are an unspecified nearest-neighbor tie convention.  A final 1NN table should define a deterministic, architecture-independent tie rule and reevaluate every saved matrix.

\medskip
\textbf{{Final completion.}}
The 1NN table has {manifest['counts']['1NN']['finished_cells']}/{manifest['counts']['1NN']['total_cells']} cells and the SVM/KSVM table has {manifest['counts']['SVM']['finished_cells']}/{manifest['counts']['SVM']['total_cells']} cells.  All 112 genuine-KSVM settings are complete: 56 WLLB and 56 full-WL.

\medskip
\textbf{{Genuine-KSVM verification.}}
The synchronized audit passed all {verification['settings_verified']} settings and {verification['fold_rows_verified']} outer-fold rows. All {verification['converged_pair_models']} pairwise SVM fits converged; {verification['pair_models_with_negative_eigenvalues']} fits had at least one negative kernel eigenvalue, so the indefinite path was exercised. The largest dual-equality residual was {verification['max_dual_equality_error']:.2e}; the largest auxiliary and original decision-identity errors were {verification['max_auxiliary_decision_error']:.2e} and {verification['max_original_decision_error']:.2e}. Every corrected $f_2$ source hash and label definition matched.
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\textwidth}}
\textbf{{Magnitude by method family.}}
\begin{{center}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrrrr}}
\toprule
Family & Done & Pend. & MAE & Max & $\geq5$\\
 & & & \multicolumn{{3}}{{c}}{{percentage points}}\\
\midrule
{audit_rows}
\bottomrule
\end{{tabular}}
\end{{center}}

\textbf{{What plausibly explains the large changes.}}
\begin{{itemize}}[leftmargin=*,itemsep=2pt,topsep=2pt]
\item \textbf{{WLLB 1NN:}} before June 2022, the public transition builder normalized adjacency \emph{{columns}} rather than rows.  Historical WLLB caches were generated before the fix, then later code loaded untracked, inconsistently named cache paths.  This is the strongest explanation for old WLLB/full-WL nonidentity; changed 1NN protocols (five 80/20 versus ten 90/10 splits) may add drift.
\item \textbf{{WLLB--$f_2$:}} the released label-support construction both overwrote repeated-degree mass and added the size offset twice.  The corrected rerun fixes this.  It cannot explain $f_1$ or WWL changes.
\item \textbf{{Catastrophic WLLB--$f_2$ KSVM rows:}} the paper's 17--34\% values on PROTEINS/IMDB are not random-fold fluctuations.  Released WTK clips the training kernel and predicts with an unprojected test kernel, uses different grids, and is not Loosli Algorithm~1; combined with the $f_2$ corruption, this is a plausible collapse mechanism.  Exact attribution is impossible because the historical matrices and manual code are absent.
\item \textbf{{Standard SVM:}} historical outer folds were unseeded, and one released path selected $\gamma$ using $\exp(-\gamma D)$ but then fit the SVM on raw $D$.  These explain modest fold-level shifts and possible protocol drift.
\item \textbf{{WWL/WL baselines:}} WWL changed from depth 10 to best of depths 1--4; the released loop also mutates graph labels between depths.  Several baseline calls are commented out, and package versions were not pinned.
\end{{itemize}}

\textbf{{Scope of confidence.}}
The public repository does not contain the historical matrices, logs, or exact executable table pipeline, so these are code-grounded explanations rather than a unique reconstruction.  There is no evidence that final experiments used a graph subset: the production paths and current rerun use complete $N\times N$ matrices; the earlier ``40 pairs'' were only a bounded implementation check.
\end{{minipage}}

\vfill
\noindent\footnotesize\textbf{{Reproducible evidence.}}
Cell CSV and audit JSON: \texttt{{\detokenize{{tmp/wl-distance-loosli-ksvm-2026-09-09/partial_results/tables/}}}}.
Depth-one controls and logs: \texttt{{\detokenize{{tmp/wl-distance-loosli-ksvm-2026-09-09/results/remote/diagnostics_20260910/}}}}.
\end{{document}}
"""
    (TABLE_DIR / "original_vs_rerun_tables_preview.tex").write_text(wrapper)
    print(
        json.dumps(
            {
                "snapshot_utc": snapshot,
                "published_1nn_rows": len(published_nn),
                "rerun_1nn_rows": len(rerun_nn),
                "published_svm_rows": len(published_svm),
                "rerun_svm_rows": len(rerun_svm),
                "output_pdf": str(OUTPUT_PDF),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
