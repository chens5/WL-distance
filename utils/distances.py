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


def wl_k(G, H, k, q=0.6, mapping="degree", method="emd"):
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
    return dWLk

#def node_label_mapping(G, H):
#    return 0

def wl_lower_bound(G, H, k, q=0.6, mapping=degree_mapping, ref_measures="norm_degree", method="emd"):
    #l_inv = {degree:[[g1, ..., gk], [h1, ...., hk]]}
    #l_inv = degree_mapping(G, H)
    l_inv = node_label_mapping(G, H)
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

    cost_matrix = calculate_cost_matrix(expm_G, expm_H, l_inv)
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
    #diffs_LB = []
    #diff_dWLk = []
    #for i in range(20):
    #    sz1 = np.randint(5, 20)
    #    sz2 = np.randint(5, 20)
    #    G = nx.erdos_renyi_graph(sz1, 0.6)
    #    H = nx.erods_renyi_graph(sz2, 0.6)
    #    dLB1 = wl_lower_bound(G, H, 1)
    #    dLB2 = wl_lower_bound(G, H, 2)
    #    dWL1 = wL_k(G, H, 1)
    #    dWL2 = wl_k(G, H, 2)
    #    diff_LB.append(abs(dLB1 - dLB2))
    #    diff_dWLk.append(abs(dWL1 - dWL2))
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2, 3])
    H.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    print(wl_k(G, H, 1))
    print(wl_k(G, H, 10))
