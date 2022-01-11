from grakel import GraphKernel, Graph, WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import itertools
from grakel.datasets import fetch_dataset
import networkx as nx
import numpy as np
import time
#from wl_distance.utils import wl_lower_bound
import sys
from wtk.utilities import krein_svm_grid_search, KreinSVC
sys.path.insert(1, './utils/')
from distances import wl_lower_bound, wl_lb_distance_matrices
from tqdm import tqdm, trange
import multiprocessing as mp
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import wwl

def grakel_to_nx(G, include_attr=False):
    nx_G = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        nx_graph = nx.from_numpy_matrix(adj_mat)
        nodes = sorted(list(graph.node_labels.keys()))
        if include_attr == True:
            for i in range(adj_mat.shape[0]):
                nx_graph.nodes[i]["attr"] = graph.node_labels[nodes[i]]
        nx_G.append(nx_graph)
    return nx_G


def grakel_to_igraph(G, add_attr=False):
    lst = []
    attr_list = []
    max_nodes = 0
    szs = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        igraph = ig.Graph.Adjacency(adj_mat)
        n = adj_mat.shape[0]
        if add_attr:
            nodes = sorted(list(graph.node_labels.keys()))
            attrs = []
            for i in range(nodes):
                attrs.append(graph.node_labels[nodes[i]])
            igraph.vs["labels"] = attrs
        if n > max_nodes:
            max_nodes = n
        lst.append(igraph)
    if add_attr:
        for graph in G:
            nodes = sorted(list(graph.node_labels.keys()))
            attrs = [0]*max_nodes
            for i in range(graph.n):
                attrs[i] = graph.node_labels[nodes[i]]
            attr_list.append(attrs)
    return lst, attr_list

def run_ksvm(G, y, k):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=23)

    D_train = mp_compute_dist_train(G_train, k)
    D_test = mp_compute_dist_test(G_train, G_test, k)
    print("starting grid search")
    svm_param = krein_svm_grid_search(D_train, D_test, y_train, y_test)
    print("finised grid search")
    clf = KreinSVC(C = svm_param)
    print("started fitting")
    clf.fit(D_train, y_train)
    y_pred = clf.predict(D_test)
    return accuracy_score(y_test, y_pred)


# These graphs G need to be switched to grakel format
def grakel_experiments(G, y):
    wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)
    wl_oa_kernel = WeisfeilerLehmanOptimalAssignment(n_iter=5)
    skf = StratifiedKFold(n_splits=10, shuffle=True)

    #for train_index, test_index in skf.split(np.array(len(G)), y):


    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return accuracy_score(y_test, y_pred)

def wwl_svm_experiment(G, y):
    igraphs, attr_list = grakel_to_igraph(G, add_attr=True)
    distances = wwl.pairwise_wasserstein_distance(igraphs, num_iterations=10)
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


    skf = StratifiedKFold(n_splits=10, shuffle=True)
    accuracies = []
    for i in range(10):
        for train_index, test_index in skf.split(distances, y):
            D_train = distances[train_index][:, train_index]
            D_test = distances[test_index][:, train_index]
            y_train = y[train_index]
            y_test = y[test_index]
            params = choose_parameters(gammas, Cs, D_train, y_train, cv=5)
            #print(params)
            K_train = np.exp(-params[0]*D_train)
            K_test = np.exp(-params[0]*D_test)

            clf = SVC(kernel='precomputed', C = params[1], max_iter=5000)
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)
            accuracies.append(accuracy_score(y_test, y_pred))
    print("Done with WWL experiments")
    print("Average accuracy:", np.mean(accuracies), "Standard deviation:", np.std(accuracies))


def svm_experiment(num_G, y, dataset_name, f):
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #train_test_split(np.arange(num_G), y, test_size = 0.1)
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    k_step = [1, 2, 3, 4]
    avg_accuracies = []
    std = []
    for k in k_step:
        distance_fname = "/data/sam/" + dataset_name + "/f4/distances_" + str(k) + ".npy"
        distances = np.load(distance_fname)
        accuracies = []
        for i in range(10):
            for train_index, test_index in skf.split(distances, y):
                D_train = distances[train_index][:, train_index]
                D_test = distances[test_index][:, train_index]
                y_train = y[train_index]
                y_test = y[test_index]
                params = choose_parameters(gammas, Cs, D_train, y_train, cv=5)
                #print(params)
                K_train = np.exp(-params[0]*D_train)
                K_test = np.exp(-params[0]*D_test)

                clf = SVC(kernel='precomputed', C = params[1], max_iter=5000)
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
                accuracies.append(accuracy_score(y_test, y_pred))
        avg_accuracies.append(np.mean(accuracies))
        std.append(np.std(accuracies))
        print("DONE WITH k = ", k)

    for i in range(4):
        k = i + 1
        f.write("k = " + str(k) + " Average accuracy = " + str(avg_accuracies[i]) + " Std. Dev = " + str(std[i])+ "\n")
        print("k =", k, "Average accuracy = ", avg_accuracies[i], "Std. Dev. = ", std[i])


@ignore_warnings(category=ConvergenceWarning)
def choose_parameters( gammas, Cs, D, y, cv=5):
    cv = StratifiedKFold(n_splits=cv)
    results = []
    param_pairs = []
    for g in gammas:
        for c in Cs:
            param_pairs.append((g, c))

    for train_index, test_index in cv.split(D, y):
        split_results = []
        for i in range(len(gammas)):
            for j in range(len(Cs)):
                g = gammas[i]
                c = Cs[j]
                D_train = D[train_index][:, train_index]
                D_test = D[test_index][:, train_index]
                K_train = np.exp(-g * D_train)
                K_test = np.exp(-g * D_test)
                y_train = y[train_index]
                y_test = y[test_index]
                clf = SVC(kernel='precomputed', C = c, max_iter=1000)
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
                split_results.append(accuracy_score(y_test, y_pred))
        results.append(split_results)

    results = np.array(results)
    fin_results = results.mean(axis=0)
    best_idx = np.argmax(fin_results)
    return param_pairs[best_idx]

def new_experiments():
    filename = "results.txt"
    f = open(filename, "w")
    datasets = ["PTC_FM", "PTC_MR", "MUTAG", "COX2_MD", "PROTEINS"]
    ds_name = ["ptc_fm", "ptc_mr", "mutag",  "cox2_md", "proteins"]
    for i in range(len(ds_name)):
        DS = fetch_dataset(datasets[i], as_graphs = True)
        G = DS.data
        nx_G = grakel_to_nx(G)
        y = DS.target
        f.write("---- Results for:" + datasets[i] + "-------\n")
        #print("---- Results for: ", datasets[i], "------")
        svm_experiment(len(G), y, ds_name[i],f )
        wwl_svm_experimeng(DS, y, )
        print("Finished with", datasets[i])


if __name__ == "__main__":
    new_experiments()
    #MUTAG = fetch_dataset("MUTAG", as_graphs=True)
    #G = MUTAG.data[:30]
    #y = MUTAG.target[:30]
    #nx_G = grakel_to_nx(G)
