from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from svm import grakel_to_nx
from grakel import GraphKernel, Graph
import itertools
from grakel.datasets import fetch_dataset
import networkx as nx
import time
import sys
sys.path.insert(1, './utils/')
from distances import wl_lower_bound
import wwl
import igraph as ig


def compute_dist_train(graph_data, k):
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n),  2):
        pairs.append(pairs_of_indexes)
    dist_matrix = np.zeros((n, n))
    for pair in pairs:
        G1 = graph_data[pair[0]]
        G2 = graph_data[pair[1]]
        dist, cp = wl_lower_bound(G1, G2, k)
        dist_matrix[pair[0]][pair[1]] = dist
        dist_matrix[pair[1]][pair[0]] = dist
    return dist_matrix

def compute_dist_test(G_test, G_train, k):
    n = len(G_test)
    m = len(G_train)
    dist_matrix = np.zeros((n, m))
    for i in range(n):
        G1 = G_test[i]
        for j in range(m):
            G2 = G_train[j]
            dist, cp= wl_lower_bound(G1, G2, k)
            dist_matrix[i][j] = dist
    return dist_matrix

# TO IMPLEMENT
def knn_mlb_experiments(G, y, k_neigh, k_step):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=23)
    clf = KNeighborsClassifier(n_neighbors=k_neigh, metric='precomputed')
    print("Compting pairwise distances in train set.....")
    start = time.time()
    D_train = compute_dist_train(G_train, k_step)
    end = time.time()
    print("Time to compute:", end - start)

    print("Computing pairwise distances in test set......")
    start = time.time()
    D_test = compute_dist_test(G_test, G_train, k_step)
    end = time.time()
    print("Time to compute:", end - start)
    clf.fit(D_train, y_train)

    y_pred = clf.predict(D_test)

    return accuracy_score(y_test, y_pred)

def knn_wwl(G, y, k_neigh, k_step):
    # get indices of train set
    # get indices of test set
    num_graphs = len(G)
    train_indices, test_indices, y_train, y_test = train_test_split(np.arange(0, num_graphs), y, test_size = 0.2, random_state=23)

    mat = wwl.pairwise_wasserstein_distances(G)
    D_train = mat[train_indices][:, train_indices]
    D_test = mat[test_indices][:, train_indices]
    clf = KNeighbors(n_neighbors = k_neigh, metric = 'recomputed')

    clf.fit(D_train, y_train)

    y_pred = clf.predict(D_test)

    return accuracy_score(y_test, y_pred)

def experiments(G, y):
    k_neigh = 5
    steps = [1, 2]
    print("MUTAG dataset results on 5-nearest neighbor")
    for k_step in steps:
        accuracy = knn_mlb_experiments(G, y, k_neigh, k_step)
        print("k =", k_step, "accuracy:", accuracy)

if __name__ == "__main__":
    MUTAG = fetch_dataset("MUTAG", as_graphs = True)
    G = MUTAG.data
    nx_G = grakel_to_nx(G)
    y = MUTAG.target
    experiments(nx_G, y)
