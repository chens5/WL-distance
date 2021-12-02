from utils import *
import scipy
import numpy as np
import networkx as nx
import ot

def calculate_histogram(M, Z, l_inv, ind):
    n = M.shape[0]
    hists = np.zeros((n, len(Z)))
    for i in range(n):
        m = M[i]
        for j in range(len(Z)):
            hists[i][j] = np.sum(m[l_inv[Z[j]][ind]])
    return hists

def calculate_cost_matrix(M_1, M_2, l_inv):
    n = M_1.shape[0]
    m = M_2.shape[0]
    cost_matrix = np.zeros((n, m))
    # Calculating histograms for each vertex
    Z = np.array(list(l_inv.keys()))
    Z1 = Z / n
    Z2 = Z / m
    hist1 = calculate_histogram(M_1, Z, l_inv, 0)
    hist2 = calculate_histogram(M_2, Z, l_inv, 1)
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = ot.wasserstein_1d(Z1, Z2, hist1[i], hist2[j])
    return cost_matrix


def wl_lower_bound(G, H, k, q=0.6, mapping=degree_mapping):
    #l_inv = {degree:[[g1, ..., gk], [h1, ...., hk]]}
    l_inv = degree_mapping(G, H)

    M_G = weighted_transition_matrix(G, q)
    M_H = weighted_transition_matrix(H, q)

    # calculate M_G^k and M_H^k
    expm_G = np.array(np.linalg.matrix_power(M_G, k))
    expm_H = np.array(np.linalg.matrix_power(M_H, k))

    # calculate stationary measures
    G_measures = get_extremal_stationary_measures(G, M_G)
    H_measures = get_extremal_stationary_measures(H, M_H)
    couplings = get_couplings(G_measures, H_measures)
    cost_matrix = calculate_cost_matrix(expm_G, expm_H, l_inv)

    dist = np.inf
    coupling = None
    for cp in couplings:
        m1 = cp[0]
        m2 = cp[1]
        W = ot.emd2(m1, m2, cost_matrix)
        # W = ot.sinkhorn2(m1, m2, cost_matrix)
        if W < dist:
            dist = W
            coupling = (m1, m2)
    return W, coupling

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edges_from([(0, 1)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2, 3])
    H.add_edges_from([(0, 1), (2, 3)])
    dist, cp = wl_lower_bound(G, H, 1)
    print(dist)
    print(cp)
