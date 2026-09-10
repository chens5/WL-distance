import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.svm import SVC


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from ksvm_utils import fit_loosli_ksvm, krein_svm_grid_search  # noqa: E402


class LoosliKsvmTests(unittest.TestCase):
    def test_reduces_to_standard_svm_for_psd_kernel(self):
        generator = np.random.default_rng(20260909)
        features = generator.normal(size=(30, 5))
        kernel = features @ features.T + 0.25 * np.eye(len(features))
        labels = np.asarray([-1] * 15 + [1] * 15)

        standard = SVC(kernel="precomputed", C=1.7).fit(kernel, labels)
        krein = fit_loosli_ksvm(kernel, labels, 1.7)

        self.assertTrue(np.array_equal(standard.predict(kernel), krein.predict(kernel)))
        self.assertLess(
            np.max(
                np.abs(
                    standard.decision_function(kernel)
                    - krein.pair_models[0].decision_function(kernel)
                )
            ),
            1e-6,
        )

    def test_handles_indefinite_multiclass_kernel(self):
        generator = np.random.default_rng(7)
        raw = generator.normal(size=(18, 18))
        kernel = 0.5 * (raw + raw.T)
        kernel += np.diag(np.linspace(-5.0, 5.0, len(raw)))
        labels = np.repeat(np.asarray([0, 1, 2]), 6)

        model = fit_loosli_ksvm(kernel, labels, 1.0)
        prediction = model.predict(kernel)

        self.assertEqual(len(model.pair_models), 3)
        self.assertEqual(prediction.shape, labels.shape)

    def test_grid_search_accepts_distance_matrices(self):
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        points = np.asarray([0.0, 0.1, 0.2, 2.0, 2.1, 2.2])
        distance = np.abs(points[:, None] - points[None, :])

        model, accuracy = krein_svm_grid_search(
            distance,
            distance,
            labels,
            labels,
            param_grid={"C": [1.0]},
            gammas=np.asarray([1.0]),
            cv=3,
        )

        self.assertEqual(model.best_params_, {"C": 1.0, "gamma": 1.0})
        self.assertEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
