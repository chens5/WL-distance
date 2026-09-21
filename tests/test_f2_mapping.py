import sys
import types
import unittest
from pathlib import Path

import networkx as nx
import numpy as np


def _dirac_wasserstein_1d(first_support, second_support, first_mass, second_mass):
    """Minimal POT stand-in for this test's one-point measures."""

    first = np.flatnonzero(first_mass)
    second = np.flatnonzero(second_mass)
    if len(first) != 1 or len(second) != 1:
        raise AssertionError("the test fixture must contain Dirac measures")
    return abs(first_support[first[0]] - second_support[second[0]])


# The mapping and cost-support logic tested here does not use CVXPY or Torch.
# Supplying narrow stand-ins keeps this regression test independent of those
# optional heavy dependencies.
fake_ot = types.ModuleType("ot")
fake_ot.wasserstein_1d = _dirac_wasserstein_1d
fake_cvxpy = types.ModuleType("cvxpy")
fake_torch = types.ModuleType("torch")
sys.modules.setdefault("ot", fake_ot)
sys.modules.setdefault("cvxpy", fake_cvxpy)
sys.modules.setdefault("torch", fake_torch)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "utils"))

from distances import calculate_cost_matrix  # noqa: E402
from utils import sz_degree_mapping  # noqa: E402

# Do not leak optional-dependency stand-ins into subsequently imported tests.
for module_name, stand_in in (
    ("ot", fake_ot), ("cvxpy", fake_cvxpy), ("torch", fake_torch)
):
    if sys.modules.get(module_name) is stand_in:
        del sys.modules[module_name]


class SizeAdjustedDegreeMappingTest(unittest.TestCase):
    def setUp(self):
        self.first = nx.path_graph(4)
        self.second = nx.path_graph(3)

    def test_mapping_retains_every_node_under_its_f2_label(self):
        mapping = sz_degree_mapping(self.first, self.second)

        self.assertEqual(mapping[1.25][0], [0, 3])
        self.assertEqual(mapping[2.25][0], [1, 2])
        self.assertEqual(mapping[1.0 + 1.0 / 3.0][1], [0, 2])
        self.assertEqual(mapping[2.0 + 1.0 / 3.0][1], [1])

        first_nodes = sorted(node for groups in mapping.values() for node in groups[0])
        second_nodes = sorted(node for groups in mapping.values() for node in groups[1])
        self.assertEqual(first_nodes, list(self.first.nodes()))
        self.assertEqual(second_nodes, list(self.second.nodes()))

    def test_cost_uses_f2_labels_without_adding_size_offset_twice(self):
        mapping = sz_degree_mapping(self.first, self.second)
        cost = calculate_cost_matrix(
            np.eye(self.first.number_of_nodes()),
            np.eye(self.second.number_of_nodes()),
            mapping,
            mapping="sz_degree_mapping",
        )

        first_labels = np.array(
            [self.first.degree[node] + 1.0 / self.first.number_of_nodes()
             for node in self.first.nodes()]
        )
        second_labels = np.array(
            [self.second.degree[node] + 1.0 / self.second.number_of_nodes()
             for node in self.second.nodes()]
        )
        expected = np.abs(first_labels[:, None] - second_labels[None, :])
        np.testing.assert_allclose(cost, expected, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
