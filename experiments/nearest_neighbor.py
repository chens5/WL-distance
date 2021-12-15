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

def grakel_to_igraph(G):
    lst = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        lst.append(ig.Graph.Adjacency(adj_mat))
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

def mp_compute_dist_test(G_test, G_train, k, n_cpus=10):
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


def compute_dist_train(graph_data, k):
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n),  2):
        pairs.append(pairs_of_indexes)
    dist_matrix = np.zeros((n, n))
    for pair in tqdm(pairs):
        G1 = graph_data[pair[0]]
        G2 = graph_data[pair[1]]
        dist = wl_lower_bound(G1, G2, k, q=0.6)
        dist_matrix[pair[0]][pair[1]] = dist
        dist_matrix[pair[1]][pair[0]] = dist
    return dist_matrix

def compute_dist_test(G_test, G_train, k):
    n = len(G_test)
    m = len(G_train)
    dist_matrix = np.zeros((n, m))
    for i in trange(n):
        G1 = G_test[i]
        for j in range(m):
            G2 = G_train[j]
            dist = wl_lower_bound(G1, G2, k, q=0.6)
            dist_matrix[i][j] = dist
    return dist_matrix

# TO IMPLEMENT
def knn_mlb_experiments(G, y, k_neigh, k_step, random_state=23):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=random_state)
    clf = KNeighborsClassifier(n_neighbors=k_neigh, metric='precomputed')
    print("Compting pairwise distances in train set.....")
    start = time.time()
    #D_train = compute_dist_train(G_train, k_step)
    D_train = mp_compute_dist_train(G_train, k_step, n_cpus=20)
    end = time.time()
    print("Time to compute:", end - start)
    print("Computing pairwise distances in test set......")
    start = time.time()
   # D_test = compute_dist_test(G_test, G_train, k_step)
    D_test = mp_compute_dist_test(G_test, G_train, k_step, n_cpus=20)
    end = time.time()
    print("Time to compute:", end - start)
    clf.fit(D_train, y_train)

    y_pred = clf.predict(D_test)

    return accuracy_score(y_test, y_pred)

def knn_wwl(G, y, k_neigh, random_state=23):
    # get indices of train set
    # get indices of test set
    num_graphs = len(G)
    train_indices, test_indices, y_train, y_test = train_test_split(np.arange(0, num_graphs), y, test_size = 0.2, random_state=random_state)

    mat = wwl.pairwise_wasserstein_distance(G)
    D_train = mat[train_indices][:, train_indices]
    D_test = mat[test_indices][:, train_indices]
    clf = KNeighborsClassifier(n_neighbors = k_neigh, metric = 'precomputed')

    clf.fit(D_train, y_train)

    y_pred = clf.predict(D_test)

    return accuracy_score(y_test, y_pred)

def experiments(G, y):
    k_neigh = 1
    steps = [1, 2, 3, 4]
    random_states = [23, 42, 64, 73, 91]
    print("MUTAG dataset results on 1-nearest neighbor")
    for k_step in steps:
        accuracies = []
        for rs in random_states:
            accuracy = knn_mlb_experiments(G, y, k_neigh, k_step, random_state=rs)
            #accuracy = knn_wwl(G, y, 1, random_state=rs)
            print("k = ", k_step, "accuracy:", accuracy)
            accuracies.append(accuracy)
        print("k = ", k_step, "average accuracy:", np.mean(accuracies), "std:", np.std(accuracies))
    #random_states = [23, 42, 64, 73]
    #for rs in random_states:
    #    print("Accuracy for WWL:", knn_wwl(G, y, k_neigh, random_state = rs)

if __name__ == "__main__":
    MUTAG = fetch_dataset("PROTEINS", as_graphs = True)
    G = MUTAG.data
    #nx_G = grakel_to_igraph(G)
    nx_G = grakel_to_nx(G)
    y = MUTAG.target
   # print(compute_dist_train(nx_G, 1))
   # mp_compute_dist_train(nx_G, 1)
   # experiments(ig_G, y)
   # print("Accuracy for WWL", knn_wwl(ig_G, y, 1, 1))
    experiments(nx_G, y)
   # cProfile.run('re.compile("wl_lower_bound")')
