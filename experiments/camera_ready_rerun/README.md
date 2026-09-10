# Camera-ready rerun (2026-09-09)

This directory preserves the source files that were actually staged on
Hellbender for the camera-ready rerun.  The files under `frozen/` are copied
verbatim from the corresponding run directories; they are intentionally not
refactored, deduplicated, or rewritten.  `SHA256SUMS` records their remote
source hashes so that the archived code can be audited independently of later
changes to the repository.

The earlier branch commit `65ee9f1` contains the reusable repository fixes and
unit tests.  The frozen bundle here contains the more verbose evaluator,
provenance checks, result serialization, and Slurm launchers that produced the
new table entries.

## Which code produced which entries

| Table entries | Frozen directory | Main program |
| --- | --- | --- |
| WLLB distances, 1NN, and standard-SVM diagnostics with `f1` | `frozen/full_wllb_f1_20260909/` | `diagnose_wllb_ksvm.py` |
| WLLB distances, 1NN, and standard-SVM diagnostics with corrected `f2` | `frozen/full_wllb_f2_20260909/` | `diagnose_wllb_ksvm.py` |
| Full WL distances, 1NN, and standard-SVM diagnostics | `frozen/full_dwl_20260909/` | `diagnose_dwl.py` |
| WWL, WL, and WL-OA baselines | `frozen/baselines_20260909/` | `run_baselines.py` |
| New rows marked `K` (full-rank Loosli KSVM) | `frozen/loosli_ksvm_20260909/` | `run_ksvm_matrix.py` and `loosli_ksvm.py` |

The duplicated scripts in the frozen directories reflect the exact remote run
layout.  For example, the WLLB `f1` and `f2` runs used separate frozen versions
of `diagnose_wllb_ksvm.py`.  Keeping both makes the provenance unambiguous.

## Settings used

The manuscript-stated settings were:

- seven datasets: MUTAG, PROTEINS, PTC-FM, PTC-MR, IMDB-BINARY, IMDB-MULTI,
  and COX2;
- depths `k = 1, 2, 3, 4` and `q = 0.6`;
- `f1(G,v) = degree_G(v)` and
  `f2(G,v) = degree_G(v) + 1 / |V_G|`;
- `K = exp(-gamma D)` with both `C` and `gamma` searched over
  `{10^-3, 10^-2, ..., 10^3}`.

The manuscript does not fully specify the resampling randomness.  The rerun
therefore records these additional choices rather than presenting them as
paper-stated facts: 10 stratified outer folds shuffled with seed `20260909`, 10
unshuffled inner folds for the paper grid, and first-maximum tie breaking in
ascending gamma-major/C-minor order.  The exact choices for every run family
are in its JSON manifest.

## Important differences from the historical released path

### Corrected `f2`

The historical `sz_degree_mapping` tested membership using the integer degree
but stored values under `degree + 1/|V|`.  Repeated-degree vertices could
therefore overwrite one another.  The historical cost construction then
treated those already-adjusted mapping keys as raw degrees and added the graph
size offset a second time.  Commit `65ee9f1` fixes both issues and adds
`tests/test_f2_mapping.py`.

The rerun additionally refuses an `f2` distance source unless its summary says
exactly `f2(G,v)=degree_G(v)+1/|V_G|`, and stores the source file's SHA-256 in
every KSVM result summary.

### Full-rank KSVM rather than WTK clipping

The historical WTK helper is a PSD-repair SVM: it eigendecomposes the training
kernel and sets eigenvalues below its negative tolerance to zero.  It does not
implement the coefficient map-back in Algorithm 1 of Loosli, Canu, and Ong.

For each binary one-vs-one problem, the rerun's `loosli_ksvm.py` instead:

1. forms the label-weighted Gram matrix `G = Y K Y`;
2. diagonalizes `G = U D U^T` and solves an ordinary SVM using the auxiliary
   positive-semidefinite spectrum `|D|`;
3. maps the auxiliary dual coefficients back with `U sign(D) U^T`; and
4. predicts with the original, possibly indefinite, test kernel.

No negative eigenvalue is clipped and no test kernel is projected.  Runtime
checks require the mapped-back training decisions to agree with the auxiliary
SVM decisions.  `smoke_test.py` also checks reduction to ordinary SVM on a PSD
kernel and exercises a deliberately indefinite kernel and the multiclass
one-vs-one path.

## Reproducing or reviewing the frozen run

The Slurm scripts contain the exact Mizzou lab paths and environment used for
the run.  On another system, adjust `RUN_ROOT`, `SOURCE_ROOT`, `DATA_ROOT`, and
`PYTHON_BIN`; doing so creates a new run rather than changing this archived
one.  The KSVM manifest records the completed smoke/gate jobs, WLLB array
`17213276`, full-WL KSVM array `17213277`, and its dependency on the full-WL
distance array `17211228`.

To verify that the frozen files still match the remote snapshots, run from the
repository root:

```bash
sha256sum -c experiments/camera_ready_rerun/SHA256SUMS
```

On macOS, `shasum -a 256 -c` can be used instead.
