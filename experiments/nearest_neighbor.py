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
import ot
import sys
sys.path.insert(1, './utils/')
from distances import wl_lower_bound
import wwl
import igraph as ig
from tqdm import trange, tqdm
import cProfile
import re
import multiprocessing as mp

def grakel_to_igraph(G, add_attr=False):
    lst = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        igraph = ig.Graph.Adjacency(adj_mat)
        if add_attr:
            n = adj_mat.shape[0]
            attrs = np.zeros(n)
            nodes = sorted(list(graph.node_labels.keys()))
            for i in range(n):
                attrs[i] = graph.node_labels[nodes[i]]
            igraph.vs["attr"] = attrs
        lst.append(igraph)
    return lst

def mp_compute_dist_train(graph_data, k, n_cpus=10):
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
    print("added jobs to pool")
    for job in tqdm(jobs):
        job.wait()
    results = [job.get() for job in jobs]
    ## format results
    dist_matrix = np.zeros((n, n))
    for i in range(len(pairs)):
        pair = pairs[i]
        dist_matrix[pair[0]][pair[1]] = results[i]
        dist_matrix[pair[1]][pair[0]] = results[i]

    return dist_matrix


def caluclate_pairwise_distances(G, y, k_step, dataset_name):
    print("Compting pairwise distances in train set.....")
    start = time.time()
    distances = mp_compute_dist_train(G, k_step, n_cpus=24)
    end = time.time()
    save_train_name = "/data/sam/" + dataset_name + "/f3/distances_" + str(k_step)
    np.save(save_train_name, distances)
    return 0

def nearest_neighbor_exp(num_G, y, dataset_name):
    k_step = [1, 2, 3, 4]
    iterations = 10
    graph_index = np.arange(num_G)
    mat_wwl = wwl.pairwise_wasserstein_distance(G, num_iterations=10)
    wwl_accuracies = []
    for i in range(iterations):
        train_index, test_index, y_train, y_test = train_test_split(np.arange(0, num_graphs), y, test_size=0.1, random_state=i)
        D_train = mat[train_index][:, train_index]
        D_test = mat[test_index][:, train_index]
        clf.KNeighborsClassifier(n_neighbors = k_neigh, metric='precomputed')
        clf.fit(D_train, y_train)
        y_pred = clf.predict(D_test)
        wwl_accuracies.append(accuracy_score(y_test, y_pred))

    print("WWL average accuracy:", np.mean(wwl_accuracies), "std dev:", np.std(wwl_accuracies))

    for k in k_step:
        distance_fname = "/data/sam/" + dataset_name + "/f2/distances_" + str(k)
        dist_matrix = np.load(distance_fname)
        k_accuracies = []
        for i in range(iterations):
            train_index, test_index, y_train, y_test = train_test_split(np.arange(num_G), y, test_size=0.1, random_state=i)
            D_train = dist_matrix[train_index][: , train_index]
            D_test = dist_matrix[test_index][:, train_index]
            clf = KNeighborsClassifier(n_neighbors = 1, metric='precomputed')
            clf.fit(D_train, y_train)
            y_pred = clf.predict(D_test)
            k_accuracies.append(accuracy_score(y_test, y_pred))
        print("k = ", k, "Avg. Accuracy = ", np.mean(k_accuracies), "Standard Dev. = ", np.std(k_accuracies))

if __name__ == "__main__":
    datasets = ["PTC_FM", "PTC_MR", "MUTAG", "IMDB-BINARY", "IMDB-MULTI", "COX2_MD", "PROTEINS"]
    ds_name = ["ptc_fm", "ptc_mr", "mutag", "imdb_b", "imdb_m", "cox2_md", "proteins"]
    for i in range(len(ds_name)):
        MUTAG = fetch_dataset(datasets[i], as_graphs = True)
        G = MUTAG.data
        nx_G = grakel_to_nx(G)
        y = MUTAG.target
        print("---- Results for: ", datasets[i], " ----")
        nearest_neighbor_exp(len(G), y, ds_name[i])
