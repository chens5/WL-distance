from utils import *
import scipy
import numpy as np
import networkx as nx
import ot
import time
import multiprocessing as mp
from tqdm import tqdm

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


def calculate_cost_matrix(M_1, M_2, l_inv):
    n = M_1.shape[0]
    m = M_2.shape[0]
    cost_matrix = np.zeros((n, m))
    # Calculating histograms for each vertex
    deg = np.array(list(l_inv.keys()))
    #Z1 = calculate_Z(deg, n, 2)
    #Z2 = calculate_Z(deg, m, 2)
    Z1 = deg_Z(deg, n)
    Z2 = deg_Z(deg, n)
    #Z1 = calculate_Z4(deg, n)
    #Z2 = calculate_Z4(deg, m)

    hist1 = calculate_histogram(M_1, deg, l_inv, 0)
    hist2 = calculate_histogram(M_2, deg, l_inv, 1)
    for i in range(n):
        for j in range(m):
            cost_matrix[i][j] = ot.wasserstein_1d(Z1, Z2, hist1[i], hist2[j])
    return cost_matrix


def wl_lower_bound(G, H, k, q=0.6, mapping=degree_mapping, ref_measures="norm_degree"):
    #l_inv = {degree:[[g1, ..., gk], [h1, ...., hk]]}
    l_inv = degree_mapping(G, H)
    #l_inv = node_label_mapping(G, H)
    M_G = weighted_transition_matrix(G, q)
    M_H = weighted_transition_matrix(H, q)

    # calculate M_G^k and M_H^k
    expm_G = np.linalg.matrix_power(M_G, k)
    expm_H = np.linalg.matrix_power(M_H, k)

    # calculate stationary measures
    #G_measures = get_extremal_stationary_measures(G, M_G)
    #H_measures = get_extremal_stationary_measures(H, M_H)
    G_measure = normalized_degree_measure(G)
    H_measure = normalized_degree_measure(H)

    cost_matrix = calculate_cost_matrix(expm_G, expm_H, l_inv)
    if np.all(cost_matrix<=1e-3):
        return 0.0
    
    #coupling_matrix = np.zeros((len(G_measures), len(H_measures)))
    #for i in range(len(G_measures)):
    #    for j in range(len(H_measures)):
    #        m1 = G_measures[i]
    #        m2 = H_measures[j]
    #        W = ot.emd2(m1, m2, cost_matrix)
            #W = ot.sinkhorn2(m1, m2, cost_matrix, 1)
    #        coupling_matrix[i][j] = W
    #seen_g = np.zeros(len(G_measures))
    #seen_h = np.zeros(len(H_measures))
    #lst_dists = []
    #dist = np.inf
    
    #while np.sum(seen_g) != len(G_measures) and np.sum(seen_h) != len(H_measures):
    #    ind = np.unravel_index(np.argmin(coupling_matrix), coupling_matrix.shape)
    #    dist = coupling_matrix[ind]
    #    coupling_matrix[ind] = np.inf
    #    if seen_g[ind[0]] != 1:
    #        seen_g[ind[0]] = 1
    #    if seen_h[ind[1]] != 1:
    #        seen_h[ind[1]] = 1

    #g_minimax = -np.inf
    #for i in range(len(G_measures)):
    #    g_min = np.min(coupling_matrix[i])
    #    if g_min > g_minimax:
    #        g_minimax = g_min
    #h_minimax = -np.inf
    #for i in range(len(H_measures)):
    #    h_min = np.min(coupling_matrix[:, i])
    #    if h_min > h_minimax:
    #        h_minimax = h_min
    
    return ot.emd2(G_measure, H_measure, cost_matrix)

def mp_compute_dist_train(graph_data, k, n_cpus = 10):
    pool = mp.Pool(processes=n_cpus)
    jobs = []
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n), 2):
        pairs.append(pairs_of_indexes)
    for i in range(len(pairs)):
        G1 = graph_data[pairs[i][0]]
        G2 = graph_data[pairs[i][1]]
        job = pool.apply_async(wl_lower_bound, args=(G1, G2, k))
        jobs.append(job)
    pool.close()

    for job in tqdm(jobs):
        job.wait()
    results = [job.get() for job in jobs]

    dist_matrix = np.zeros((n, n))
    for i in range(len(pairs)):
        pair = pairs[i]
        dist_matrix[pair[0]][pair[1]] = results[i]
        dist_matrix[pair[0]][pair[1]] = results[i]
    return dist_matrix

def mp_compute_dist_test(G_train, G_test, k, n_cpus=10):
    n = len(G_test)
    m = len(G_train)

    pool = mp.Pool(processes=n_cpus)
    jobs = []

    for i in range(n):
        for j in range(m):
            G1 = G_test[i]
            G2 = G_train[j]
            job = pool.apply_async(wl_lower_bound, args=(G1, G2, k))
            jobs.append(job)
    pool.close()
    for job in tqdm(jobs):
        job.wait()
    results = [job.get() for job in jobs]
    return np.reshape(results, (n, m))

def wl_lb_distance_matrices(G_train,G_test, k, n_cpus=10):
    D_train = mp_compute_dist_train(G_train, k, n_cpus=n_cpus)
    D_test = mp_compute_dist_test(G_train, G_test, k, n_cpus=n_cpus)
    return D_train, D_test

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3), (4, 5), (5, 6), (6, 7), (7, 4)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    H.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)])
    start = time.time()
    dist = wl_lower_bound(G, H, 2)
    end = time.time()
    print("Computed in time:", end - start)
    print(dist)
