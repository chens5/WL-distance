#!/usr/bin/env python3
"""Remote numerical checks for the translated Loosli KSVM implementation."""

from __future__ import annotations

import json

import numpy as np
from sklearn.svm import SVC

from loosli_ksvm import fit_loosli_ksvm, model_diagnostics


def main() -> None:
    generator = np.random.default_rng(20260909)

    features = generator.normal(size=(30, 5))
    psd_kernel = features @ features.T + 0.25 * np.eye(len(features))
    binary_labels = np.asarray([-1] * 15 + [1] * 15)
    standard = SVC(kernel="precomputed", C=1.7, max_iter=-1).fit(psd_kernel, binary_labels)
    krein = fit_loosli_ksvm(psd_kernel, binary_labels, 1.7)
    standard_decision = standard.decision_function(psd_kernel)
    krein_decision = krein.pair_models[0].decision_function(psd_kernel)
    psd_error = float(np.max(np.abs(standard_decision - krein_decision)))
    if psd_error > 1e-6 or not np.array_equal(standard.predict(psd_kernel), krein.predict(psd_kernel)):
        raise RuntimeError(f"PSD reduction check failed: {psd_error}")

    raw = generator.normal(size=(36, 36))
    indefinite_kernel = 0.5 * (raw + raw.T)
    indefinite_kernel += np.diag(np.linspace(-8.0, 8.0, len(raw)))
    indefinite_labels = np.asarray([-1] * 18 + [1] * 18)
    indefinite = fit_loosli_ksvm(indefinite_kernel, indefinite_labels, 0.8)
    diagnostics = model_diagnostics(indefinite)
    if diagnostics["pair_models"][0]["negative_eigenvalue_count"] == 0:
        raise RuntimeError("indefinite fixture unexpectedly has no negative eigenvalue")

    multiclass_labels = np.repeat(np.asarray([0, 1, 2]), 12)
    multiclass = fit_loosli_ksvm(indefinite_kernel, multiclass_labels, 1.0)
    multiclass_prediction, metadata = multiclass.predict_with_metadata(indefinite_kernel)
    if multiclass_prediction.shape != multiclass_labels.shape or len(multiclass.pair_models) != 3:
        raise RuntimeError("multiclass one-vs-one check failed")

    print(
        json.dumps(
            {
                "event": "smoke_passed",
                "psd_reduction_max_decision_error": psd_error,
                "indefinite_diagnostics": diagnostics,
                "multiclass_pair_count": len(multiclass.pair_models),
                "multiclass_metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
