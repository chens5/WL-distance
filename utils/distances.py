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
    Z = list(l_inv.keys())
    hist1 = calculate_histogram(M_1, Z, l_inv, 0)
    hist2 = calculate_histogram(M_2, Z, l_inv, 1)
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = ot.wasserstein_1d(Z, Z, hist1[i], hist2[j])
    return cost_matrix

# Default l is degree function
# TODO: additional functionality for l.
def wl_lower_bound(G, H, k, mat_fn=default_transition_matrix):
    # deg_dict maps each possible degree to nodes of that degree
    # deg_dict[u][0] is dictionary for G, deg_dict[u][1] for H
    # deg_dict[u] = [[G1, G2, ...., Gk], [H1, ..., Hk]]
    deg_dict = {}
    for node in G.nodes():
        deg = G.degree[node]
        if deg not in deg_dict:
            deg_dict[deg] = [[node], []]
        else:
            deg_dict[deg][0].append(node)

    for node in H.nodes():
        deg = H.degree[node]
        if deg not in deg_dict:
            deg_dict[deg] = [[], [node]]
        else:
            deg_dict[deg][1].append(node)

    M_G = mat_fn(G)
    M_H = mat_fn(H)

    # calculate M_G^k and M_H^k
    expm_G = np.array(np.linalg.matrix_power(M_G, k))
    expm_H = np.array(np.linalg.matrix_power(M_H, k))

    # calculate stationary measures
    # G_measures = get_extremal_stationary_measures(G, M_G)
    # H_measures = get_extremal_stationary_measures(H, M_H)
    G_measures = np.array(get_stationary_measures(M_G))
    print(G_measures)
    H_measures = np.array(get_stationary_measures(M_H))
    print(H_measures)
    couplings = get_couplings(G_measures, H_measures)
    cost_matrix = calculate_cost_matrix(expm_G, expm_H, deg_dict)
    print("Cost Matrix:")
    print(cost_matrix)
    dist = np.inf
    coupling = None
    for cp in couplings:
        m1 = cp[0]
        m2 = cp[1]
        W = ot.emd2(m1, m2, cost_matrix)
        if W < dist:
            dist = W
            coupling = (m1, m2)
    return W, coupling

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1),(2, 3)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2])
    H.add_edges_from([(0, 1)])
    dist, cp = wl_lower_bound(G, H, 1)
    print(dist)
    print(cp)
