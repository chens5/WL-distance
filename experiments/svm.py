from grakel import GraphKernel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
from grakel.datasets import fetch_dataset

def markov_chain_lb_kernel(G1, G2, k, lam):
    dist = wl_lower_bound(G1, G2, k)
    return np.exp(-lam * dist)

def compute_kernel_matrix(graph_data, k, lam):
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n_train)), 2):
        pairs.append(pairs_of_indexes)
    kernel_matrix = np.zeros((n, n))
    for pair in pairs:
        G1 = graph_data[pair[0]]
        G2 = graph_data[pair[1]]
        kernel = markov_chain_lb_kernel(G1, G2, k, lam)
        kernel_matrix[pair[0]][pair[1]] = kernel
        kernel_matrix[pair[1]][pair[0]] = kernel
    return kernel_matrix

def grakel_to_nx(G):
    nx_G = []
    for graph in G:
        nx_G.append(nx.from_numpy_matrix(graph.adjacency_matrix))
    return 0

# Assume G in networkx format
def run_markov_chain_svm(G, y, k, lam):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=23)
    n_train = len(G_train)
    n_test = len(G_test)

    K_train = compute_kernel_matrix(G_train, k, lam)
    K_test = compute_kernel_matrix(G_test, k, lam)

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

def experiments():
    #kernels = ["random_walk", "shortest_path", "weisfeiler_lehman_optimal_assignment", "weisfeiler_lehman"]
    MUTAG = fetch_dataset("MUTAG", verbose=False)
    G = MUTAG.data
    y = MUTAG.target
