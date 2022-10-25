import networkx as nx
import numpy as np
import itertools
import cvxpy as cp
import ot


def weighted_transition_matrix(G, q):
    A = np.asarray(nx.adjacency_matrix(G, weight=None).todense())
    n = A.shape[0]
    D = np.sum(A, axis = 1)
    mask = D == 0
    D[mask] = 1
    D = D.reshape((A.shape[0], 1))
    A = (1 - q)*A/D
    A = A + q * np.identity(n)
    single_node_inds = np.nonzero(mask)[0]
    A[single_node_inds, single_node_inds] = 1
    return A

def sz_degree_mapping(G1, G2):
    mapping = {}
    n = G1.number_of_nodes()
    m = G2.number_of_nodes()
    for node in G1.nodes():
        deg = G1.degree[node]
        if deg not in mapping:
            mapping[deg + 1/n] = [[node], []]
        else:
            mapping[deg + 1/n][0].append(node)

    for node in G2.nodes():
        deg = G2.degree[node]
        if deg not in mapping:
            mapping[deg + 1/m] = [[], [node]]
        else:
            mapping[deg + 1/m][1].append(node)
    return mapping

def degree_mapping(G1, G2):
    mapping = {}
    for node in G1.nodes():
        deg = G1.degree[node]
        if deg not in mapping:
            mapping[deg] = [[node], []]
        else:
            mapping[deg][0].append(node)

    for node in G2.nodes():
        deg = G2.degree[node]
        if deg not in mapping:
            mapping[deg] = [[], [node]]
        else:
            mapping[deg][1].append(node)
    return mapping

def node_label_mapping(G1, G2):
    mapping = {}
    for node in G1.nodes():
        deg = G1.degree[node]
        lbl = (2**deg)*(3**G1.nodes[node]["attr"])
        if lbl not in mapping:
            mapping[lbl] = [[node], []]
        else:
            mapping[lbl][0].append(node)
    for node in G2.nodes():
        deg = G2.degree[node]
        lbl = (2**deg) * (3**G2.nodes[node]["attr"])
        if lbl not in mapping:
            mapping[lbl] = [[], [node]]
        else:
            mapping[lbl][1].append(node)
    return mapping

def get_components(G):
    L = []
    for c in nx.connected_components(G):
        L.append(c)
    return L

def normalized_degree_measure(G):
    n = G.number_of_nodes()
    measure = np.zeros(n)
    total = 0
    for node in G.nodes():
        measure[node] = G.degree[node]
        total += G.degree[node]
    return measure/total

def get_extremal_stationary_measures(G, transition_mat):
    # Get transition matrix for G
    components = get_components(G)
    n = transition_mat.shape[0]
    stationary_measures = []
    for c in components:
        l = list(c)
        c_transition = transition_mat[l][:, l]
        measures = get_stationary_measures(c_transition)
        for m in measures:
            sm = np.zeros(n)
            sm[l] = m
            stationary_measures.append(sm)

    return stationary_measures

def get_stationary_measures(M):
    # find basis of subspace of stationary measures
    evals, evecs = np.linalg.eig(M.T)
    evals = np.real_if_close(evals)
    evecs_1 = evecs[:,np.isclose(evals, 1)]
    stationary_measures = np.real_if_close(evecs_1.T)

    sum_row = np.sum(stationary_measures, axis=1)
    stationary_measures = stationary_measures/sum_row
    return stationary_measures

def get_couplings(n, m):
    couplings = [(x, y) for x in range(n) for y in range(m)]
    return couplings


if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1),(1, 2), (2, 3)])
    M = weighted_transition_matrix(G, 0)
    mu = normalized_degree_measure(G)
    print(mu)
    print(mu.T @ M)
    #C = np.array([[1, 0.5], [0.5, 1]])
    #m1 = np.array([0.5, 0.5])
    #m2 = np.array([0.3, 0.7])
    #print(ot_solver(m1, m2, C))
    #print(ot.emd2(m1, m2, C))
