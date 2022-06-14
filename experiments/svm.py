from grakel import GraphKernel, Graph, WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment
from grakel.utils import cross_validate_Kfold_SVM
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
#from wtk.utilities import krein_svm_grid_search, KreinSVC
from ksvm_utils import *
sys.path.insert(1, './utils/')
from distances import wl_lower_bound, wl_lb_distance_matrices
from tqdm import tqdm, trange
import multiprocessing as mp
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import wwl
import igraph as ig
from tqdm import trange, tqdm

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
            for i in range(len(nodes)):
                attrs.append(graph.node_labels[nodes[i]])
            igraph.vs["label"] = attrs
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

def run_ksvm(D_train, D_test, y_train, y_test):
    #G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=23)

    #D_train = mp_compute_dist_train(G_train, k)
    #D_test = mp_compute_dist_test(G_train, G_test, k)
    print("starting grid search")
    svm_clf, accuracy = krein_svm_grid_search(D_train, D_test, y_train, y_test)
    #print(svm_param)
    
    #print("finised grid search")
    #clf = KreinSVC(C = svm_param[0], gamma=svm_param[1])
    #print("started fitting")
    #clf.fit(D_train, y_train)
    #y_pred = clf.predict(D_test)
    return accuracy


# These graphs G need to be switched to grakel format
def grakel_experiments(G, y):
    Ks_wl = []
    Ks_wloa = []
    for i in range(1, 5):
        wl_kernel = WeisfeilerLehman(n_iter=i, normalize=True, base_graph_kernel=VertexHistogram)
        wloa_kernel = WeisfeilerLehmanOptimalAssignment(n_iter=i, normalize=True)
        K_wl = wl_kernel.fit_transform(G)
        K_wloa = wloa_kernel.fit_transform(G)
        Ks_wl.append(K_wl)
        Ks_wloa.append(K_wloa)
    
    accs_wl = cross_validate_Kfold_SVM([Ks_wl], y, n_iter=10)
    accs_wloa = cross_validate_Kfold_SVM([Ks_wloa], y, n_iter=10)
    print("WL Average accuracy:", np.mean(accs_wl), "Standard Dev:", np.std(accs_wl))
    print("WLOA Average accuracy:", np.mean(accs_wloa), "Standard Dev:", np.std(accs_wloa))


def wwl_svm_experiment(G, y, f):
    igraphs, attr_list = grakel_to_igraph(G)
    distances = wwl.pairwise_wasserstein_distance(igraphs, num_iterations=5)
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


    skf = StratifiedKFold(n_splits=10, shuffle=True)
    accuracies = []
    for train_index, test_index in tqdm(skf.split(distances, y)):
        D_train = distances[train_index][:, train_index]
        D_test = distances[test_index][:, train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        params = choose_parameters(gammas, Cs, D_train, y_train, cv=10)
        #clf=KreinSVC(C = params)
        #print(params)
        #K_train = np.exp(-params[0]*D_train)
        #K_test = np.exp(-params[0]*D_test)
        #eigv, eigvecs = np.linalg.eigh(K_train)
        #eigv[eigv< 0] = 0
        #K_train = eigvecs @ np.diag(eigv) @ np.linalg.inv(eigvecs)
        clf = SVC(kernel='precomputed', C = params[1], max_iter=5000)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    print("Done with WWL experiments")
    print("Average accuracy:", np.mean(accuracies), "Standard deviation:", np.std(accuracies))
    f.write("WWL Average accuracy = " + str(np.mean(accuracies)) + " std dev = " + str(np.std(accuracies)) + "\n")


def svm_experiment(num_G, y, dataset_name, f):
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #train_test_split(np.arange(num_G), y, test_size = 0.1)
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    k_step = [1, 2, 3, 4]
    avg_accuracies = []
    std = []
    for k in k_step:
        distance_fname = "/data/sam/" + dataset_name + "/f3/distances_" + str(k) + ".npy"
        distances = np.load(distance_fname)
        accuracies = []
        for train_index, test_index in skf.split(distances, y):
            D_train = distances[train_index][:, train_index]
            D_test = distances[test_index][:, train_index]
            y_train = y[train_index]
            y_test = y[test_index]
            #params = choose_parameters(gammas, Cs, D_train, y_train, cv=10)
            #print(params)
            accuracy = run_ksvm(D_train, D_test, y_train, y_test)
            #h()
            #print("choosing params")
            #params = krein_svm_grid_search(D_train, D_test, y_train, y_test)
            #K_train = np.exp(-params[0]*D_train)
            #K_test = np.exp(-params[0]*D_test)
            #eigenvalues, eigenvectors = np.linalg.eigh(K_train)
            #eigenvalues[eigenvalues < 0] = 0
            #K_train = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
            #eigv, eigvec = np.linalg.eigh(K_test)
            #eigv[eigv < 0] = 0
            #K_test = eigvec @ np.diag(eigv) @ np.linalg.inv(eigvec)
            #pct_nonneg = np.sum(eigenvalues > 0)/len(eigenvalues)
            #print("k = ", k, "percentage positive eigenvalues", pct_nonneg)
            #clf = KreinSVC()
            #clf.fit(D_train, y_train)
            #y_pred = clf.predict(D_test)
            #accuracies.append(accuracy_score(y_test, y_pred))
            accuracies.append(accuracy)
        avg_accuracies.append(np.mean(accuracies))
        std.append(np.std(accuracies))
        print("DONE WITH k = ", k)
        print("AVERAGE ACCURACY", np.mean(accuracies))
        

    for i in range(4):
        k = i + 1
        f.write("k = " + str(k) + " Average accuracy = " + str(avg_accuracies[i]) + " Std. Dev = " + str(std[i])+ "\n")
        print("k =", k, "Average accuracy = ", avg_accuracies[i], "Std. Dev. = ", std[i])


@ignore_warnings(category=ConvergenceWarning)
def choose_parameters( gammas, Cs, D, y, cv=5, ksvm=False):
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

# strip labels from grakel graph
def strip_labels(G):
    for graph in G:
        for node in graph.node_labels:
            #print(graph.node_labels[node])
            if node in graph.edge_dictionary:
                graph.node_labels[node] = len(graph.neighbors(node))
            #print(len(graph.neighbors(node)))
            #print(graph.node_labels[node])

def new_experiments():
    filename = "ksvm_wllb_f3.txt"
    f = open(filename, "w")
    datasets = ["PTC_FM", "PTC_MR", "MUTAG", "IMDB-BINARY", "IMDB-MULTI", "PROTEINS", "COX2"]
    ds_name = ["ptc_fm", "ptc_mr", "mutag",  "imdb_b", "imdb_m", "proteins", "cox2"]
    #datasets=["PROTEINS",  "COX2"]
    #ds_name=["proteins", "cox2"]
    for i in range(len(ds_name)):
        DS = fetch_dataset(datasets[i], as_graphs = True, produce_labels_nodes=True)
        G = DS.data
        #print(G[0].node_labels)
        #strip_labels(G)
        #print(G[0].node_labels)
        nx_G = grakel_to_nx(G)
        y = DS.target
        f.write("---- Results for:" + datasets[i] + "-------\n")
        print("---- Results for: ", datasets[i], "------")
        svm_experiment(len(G), y, ds_name[i],f )
        #wwl_svm_experiment(G, y, f )
        #grakel_experiments(G, y)
        print("Finished with", datasets[i])


if __name__ == "__main__":
    new_experiments()
