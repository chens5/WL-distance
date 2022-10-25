from utils import *
import scipy
import numpy as np
import networkx as nx
import ot
import time
import multiprocessing as mp
from tqdm import tqdm
import torch

def calculate_histogram(M, Z, l_inv, ind):
    n = M.shape[0]
    hists = np.zeros((n, len(Z)))
    len_Z = len(Z)
    for i in range(n):
        m = M[i]
        for j in range(len_Z):
            # Z[j] = value of Z at index j
            # l_inv[Z[j]][ind] = nodes mapped to Z[j], nodes are expressed at indices
            # m[l_inv[Z[j]][ind]] = get measure m_{x[i]}^X(x) of all nodes, x
            hists[i][j] = np.sum(m[l_inv[Z[j]][ind]])
    return hists

def calculate_Z(deg, n, mult):
    Z = np.zeros(len(deg))
    zero_mask = Z == 0
    non_zero_mask = np.invert(zero_mask)
    Z[non_zero_mask] = (1 / Z[non_zero_mask]) + mult*n
    Z[zero_mask] = mult*n
    return Z

def deg_Z(deg, n):
    return deg

def calculate_Z4(deg, n):
    return deg + 1/n


def calculate_cost_matrix(M_1, M_2, l_inv, mapping="degree_mapping"):
    n = M_1.shape[0]
    m = M_2.shape[0]
    cost_matrix = np.zeros((n, m))
    # Calculating histograms for each vertex
    deg = np.array(list(l_inv.keys()))
    if mapping == "degree_mapping":
        Z1 = deg_Z(deg, n)
        Z2 = deg_Z(deg, n)
    else:
        Z1 = calculate_Z4(deg, n)
        Z2 = calculate_Z4(deg, m)

    hist1 = calculate_histogram(M_1, deg, l_inv, 0)
    hist2 = calculate_histogram(M_2, deg, l_inv, 1)
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = ot.wasserstein_1d(Z1, Z2, hist1[i], hist2[j])
    return cost_matrix


# version of WL distance that works with general labelled measure Markov Chains
# Note that for both this version and the networkx version, you can either use
# EMD2 or Sinkhorn2 to compute WL distance (as implemented by POT library)
# Other implementations for OT must be added by the user.
def wlk_mc(Mx, My, lx, ly, muX, muY,k, return_costs=False, method='emd2'):
    n = Mx.shape[0]
    m = My.shape[0]
    prev_matrix = torch.zeros((n, m))
    cost_matrix = torch.zeros((n, m))
    for i in range(n):
        for j in range(m):
            prev_matrix[i][j] = torch.norm(lx[i] - ly[j])
    costs = []
    for step in range(k):
        for i in range(n):
            for j in range(m):
                m1 = Mx[i]
                m2 = My[j]
                if method == 'emd2':
                    cost_matrix[i][j] = ot.emd2(m1, m2, prev_matrix)
                elif method == 'sinkhorn2':
                    cost_matrix[i][j] = ot.sinkhorn2(m1, m2, prev_matrix, reg=0.1)
                else:
                    raise Exception("method not implemented; use sinkhorn2 or emd2")
        if return_costs:
            costs.append(cost_matrix)
        prev_matrix = cost_matrix.detach().clone()
    if method == 'emd2':
        return ot.emd2(muX, muY, cost_matrix)
    elif method == 'sinkhorn2':
        return ot.sinkhorn2(muX, muY, cost_matrix)
    else:
        raise Exception("method not implemented; use sinkhorn2 or emd2")

# Version of WL distance that works with networkX graphs
def wl_k(G, H, k, q=0.6, mapping="degree", method="emd", return_cost=False):
    M_G = weighted_transition_matrix(G, q)
    M_H = weighted_transition_matrix(H, q)
    n = M_G.shape[0]
    m = M_H.shape[0]
    prev_matrix = np.zeros((n, m))
    cost_matrix = np.zeros((n, m))
    for n1 in G.nodes():
        for n2 in H.nodes():
            if mapping == "degree":
                prev_matrix[n1][n2] = np.abs(G.degree[n1] - H.degree[n2])
            elif mapping == "label":
                prev_matrix[n1][n2] = np.abs(G.nodes[n1]["attr"] - H.nodes[n2]["attr"])
            else:
                prev_matrix[n1][n2] = np.abs(G.degree[n1] + (1/n) - H.degree[n2] - (1/m))
    costs = []
    for step in range(k):
        for i in range(n):
            for j in range(m):
                m1 = M_G[i]
                m2 = M_H[j]
                #if method == "emd":
                #    cost_matrix[i][j] = ot.emd2(m1, m2, prev_matrix)
                #elif method == "sinkhorn":
                    #m1 = m1 + 1e-3
                    #m2 = m2 + 1e-3
                    #m11, m21, prev_matrix1 = ot.utils.clean_zeros(m1, m2, prev_matrix)
                #    m1 = m1 + 1e-3
                #    m2 = m2 + 1e-3
                #    cost_matrix[i][j] = ot.sinkhorn2(m1, m2, prev_matrix, 50)
                    
                #cost_matrix[i][j] = ot.sinkhorn2(m1, m2, prev_matrix, 100)
                cost_matrix[i][j] = ot.emd2( m1, m2 ,prev_matrix)
        prev_matrix = np.copy(cost_matrix)
        costs.append(prev_matrix)

    muX = normalized_degree_measure(G)
    muY = normalized_degree_measure(H)
    #dWLk = ot.emd2( muX, muY, cost_matrix)
    if method=="emd":
        dWLk = ot.emd2(muX, muY, cost_matrix)
    elif method == "sinkhorn":
        #muX = muX + 1e-3
        #muY = muY + 1e-3
        dWLk = ot.sinkhorn2(muX, muY, cost_matrix, 1, method='sinkhorn')
    #dWLk = ot.sinkhorn2(muX, muY, cost_matrix, 100)
    if return_cost:
        return dWLk, costs
    return dWLk

# mapping options: sz_degree_mapping, degree_mapping
def wl_lower_bound(G, H, k, q=0.6, mapping="degree_mapping", ref_measures="norm_degree", method="emd"):
    #l_inv = {degree:[[g1, ..., gk], [h1, ...., hk]]}
    # l_inv = degree_mapping(G, H)

    if mapping=="sz_degree_mapping":
        l_inv = sz_degree_mapping(G, H)
    else:
        l_inv = degree_mapping(G, H)
    M_G = weighted_transition_matrix(G, q)
    M_H = weighted_transition_matrix(H, q)
    #print(M_G)
    # calculate M_G^k and M_H^k
    expm_G = np.linalg.matrix_power(M_G, k)
    #print(expm_G)
    expm_H = np.linalg.matrix_power(M_H, k)

    # calculate stationary measures
    G_measure = normalized_degree_measure(G)
    H_measure = normalized_degree_measure(H)

    cost_matrix = calculate_cost_matrix(expm_G, expm_H, l_inv, mapping=mapping)
    if np.all(cost_matrix<=1e-3):
        return 0.0
    if method == "emd":
        return ot.emd2(G_measure, H_measure, cost_matrix)
    elif method == "sinkhorn":
        G_measure = G_measure + 1e-3
        H_measure = H_measure + 1e-3
        return ot.sinkhorn2(G_measure, H_measure, cost_matrix, 100)
    #return ot.sinkhorn2(G_measure, H_measure, cost_matrix, 100)
    #return ot.emd2(G_measure, H_measure, cost_matrix)


if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2, 3, 4])
    H.add_edges_from([(0, 1), (0, 2), (0, 3) ,(0, 4)])
    # print(wl_k(G, H, 1))
    # print(wl_k(G, H, 10))
    wlk, costs = wl_k(G, H, 30, q=0.8,  return_cost=True)
    for i in range(len(costs) - 1):
        print(np.linalg.norm(costs[i] - costs[i + 1], ord='fro'))
