from grakel import GraphKernel, Graph
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
from grakel.datasets import fetch_dataset
import networkx as nx
import numpy as np
import time
#from wl_distance.utils import wl_lower_bound
import sys
sys.path.insert(1, './utils/')
from distances import wl_lower_bound

def markov_chain_lb_kernel(G1, G2, k, lam):
    dist, cp = wl_lower_bound(G1, G2, k)
    return np.exp(-lam * dist)

def compute_kernel_matrix(graph_data, k, lam):
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n),  2):
        pairs.append(pairs_of_indexes)
    kernel_matrix = np.zeros((n, n))
    for pair in pairs:
        G1 = graph_data[pair[0]]
        G2 = graph_data[pair[1]]
        kernel = markov_chain_lb_kernel(G1, G2, k, lam)
        kernel_matrix[pair[0]][pair[1]] = kernel
        kernel_matrix[pair[1]][pair[0]] = kernel
    return kernel_matrix

def compute_kernel_test(G_test, G_train, k, lam):
    n = len(G_test)
    m = len(G_train)
    kernel_matrix = np.zeros((n, m))
    for i in range(n):
        G1 = G_test[i]
        for j in range(m):
            G2 = G_train[j]
            kernel= markov_chain_lb_kernel(G1, G2, k, lam)
            kernel_matrix[i][j] = kernel
    return kernel_matrix

def grakel_to_nx(G):
    nx_G = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        nx_G.append(nx.from_numpy_matrix(adj_mat))
    return nx_G

# Assume G in networkx format
def run_markov_chain_svm(G, y, k, lam):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=23)

    K_train = compute_kernel_matrix(G_train, k, lam)
    K_test = compute_kernel_test(G_test, G_train, k, lam)
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return accuracy_score(y_test, y_pred)


# These graphs G need to be switched to grakel format
def run_grakel_svm(kernel, G, y):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=23)
    
    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)
    
    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return accuracy_score(y_test, y_pred)

def experiments(k, lam):
    #kernels = ["random_walk", "shortest_path", "weisfeiler_lehman_optimal_assignment", "weisfeiler_lehman"]
    MUTAG = fetch_dataset("MUTAG", as_graphs=True)
    G = MUTAG.data
    nx_G = grakel_to_nx(G)
    y = MUTAG.target
    sp_kernel = GraphKernel(kernel="shortest_path")
    print("Running SVC with shortest path")
    start = time.time()
    print("Accuracy:", run_grakel_svm(sp_kernel, G, y))
    end = time.time()
    print("Done in:", end - start)
    
    #print("Running SVC with random walk")
    #rw_kernel = GraphKernel(kernel="random_walk")
    #start = time.time()
    #print("Accuracy:", run_grakel_svm(rw_kernel, G, y))
    #end = time.time()
    #print("Done in:", end - start)

    print("Running SVC with lower bound kernel")
    start = time.time()
    print(run_markov_chain_svm(nx_G, y, k , lam))
    end = time.time()
    print("Done in:", end - start)

    #sp_kernel.fit_transform(G[:5])
    #tg = Graph(G[0])
    #print(G[0].get_adjacency_matrix())

if __name__ == "__main__":
    experiments(1, 1)
