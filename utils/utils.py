import networkx as nx
import numpy as np
import itertools

def weighted_transition_matrix(G, q):
    A = np.asarray(nx.adjacency_matrix(G, weight=None).todense())
    n = A.shape[0]
    D = np.sum(A, axis = 1)
    mask = D == 0
    D[mask] = 1
    A = (A * (1 - q))/D
    A = A + q * np.identity(n)
    single_node_inds = np.nonzero(mask)[0]
    A[single_node_inds, single_node_inds] = 1
    return A


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

def get_components(G):
    L = []
    for c in nx.connected_components(G):
        L.append(c)
    return L

# TO IMPLEMENT
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

def get_couplings(m1, m2):
    couplings = [(x, y) for x in m1 for y in m2]
    return couplings


if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from([(0, 1), (2, 3)])
    print(weighted_transition_matrix(G, 0.6))
