import networkx as nx
import numpy as np
import itertools

# Input: networkx graph, G
# Output: transition matrix for G, m(x, x') = 1/(deg(x) + 1) for x' \in N(x).
def default_transition_matrix(G):
    A = nx.adjacency_matrix(G, weight=None).todense()
    n = A.shape[0]
    A = A + np.identity(n)
    D = np.sum(A, axis=1)
    return A/D

def get_components(G):
    L = []
    for c in nx.connected_components(G):
        L.append(c)
    return L

# TO IMPLEMENT
def get_extremal_stationary_measures(G, fn=default_transition_matrix):
    # Get transition matrix for G
    components = get_components(G)
    transition_mat = fn(G)
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
    evecs_1 = evecs[:,np.isclose(evals, 1)]
    stationary_measures = evecs_1.T

    sum_row = np.sum(stationary_measures, axis=1)
    stationary_measures = stationary_measures/sum_row
    return stationary_measures

def get_couplings(m1, m2):
    couplings = [(x, y) for x in m1 for y in m2]
    return couplings


if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (1, 2)])
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2, 3])
    H.add_edges_from([(0, 1), (2, 3)])
    M_H = default_transition_matrix(H)
    M_G = default_transition_matrix(G)
    sm_G = get_extremal_stationary_measures(G)
    sm_H = get_extremal_stationary_measures(H)
    print(sm_G)
    print(sm_H)
    coupling = get_couplings(sm_G, sm_H)
    for c in coupling:
        print("-------")
        print(c)
