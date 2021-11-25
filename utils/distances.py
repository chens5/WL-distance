from utils import default_transition_matrix
from get_stationary_measures
import scipy
import numpy as np
import networkx as nx
import ot

# IMPLEMENT
def one_dimensional_wasserstein(x_a, x_b, a, b):
    return 0

# IMPLEMENT
def wasserstein(cost_matrix, m1, m2):
    return 0

# IMPLEMENT
def calculate_cost_matrix( M_1, M_2, l):
    # l = degree to node mapping mapping
    n = M_1.shape[0]
    m = M_2.shape[1]
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            measure1 = M_1[i]
            measure2 = M_2[i]
            l1 = []
            l2 = []
            for val in l:
                l1[val] = np.sum(measure1[l[val]])
                l2[val] = np.sum(measure2[l[val]])
            cost_matrix[i][j] = one_dimensional_wasserstein(deg1, deg2, l1, l2)
    return cost_matrix

def wl_lower_bound(G, H, k):
    deg_dict = {}
    for node in G:
        deg = node.degree
        if deg not in deg_dict:
            deg_dict[deg] = [[node], []]
        else:
            deg_dict[deg][0].append(node)
    for node in H:
        deg = node.degree
        if deg not in deg_dict:
            deg_dict[deg] = [[], [node]]
        else:
            deg_dict[deg][1].append(node)

    M_G = default_transition_matrix(G)
    M_H = default_transition_matrix(H)

    # calculate M_G^k and M_H^k
    expm_G = np.matrix_power(M_G, k)
    expm_H = np.matrix_power(M_H, k)

    # calculate stationary measures
    G_measures = get_stationary_measures(M_G)
    H_measures = get_stationary_measures(M_H)
    couplings = get_couplings(G_measures, H_measures)
    cost_matrix = calculate_cost_matrix(expm_G, expm_H)

    for cp in couplings:
        m1 = cp[0]
        m2 = cp[1]
        wasserstein( cost_matrix, m1, m2)
