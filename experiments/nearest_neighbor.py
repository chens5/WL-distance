from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from svm import grakel_to_nx, grakel_to_igraph
from grakel import GraphKernel, Graph
import itertools
from grakel.datasets import fetch_dataset
import networkx as nx
import time
import ot
import sys
sys.path.insert(1, './utils/')
from distances import wl_lower_bound, wl_k
import wwl
import igraph as ig
from tqdm import trange, tqdm
import cProfile
import re
import multiprocessing as mp
import matplotlib.pyplot as plt

def mp_compute_dist_train(graph_data, k, n_cpus=10, dist="real"):
    pool = mp.Pool(processes=n_cpus)
    jobs = []
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n), 2):
        pairs.append(pairs_of_indexes)

    for i in range(len(pairs)):
        G1 = graph_data[pairs[i][0]]
        G2 = graph_data[pairs[i][1]]
        if dist == "real":
            job = pool.apply_async(wl_k, args=(G1, G2, k))
        else:
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

def time_experiment(k):
    for k in range(1, 2):
        wllb = []
        wllb_sh = []
        wlk = []
        wlk_sh = []
        wwl_t = []
        for i in range(5, 100):
            G = nx.erdos_renyi_graph(i, 0.6)
            H = nx.erdos_renyi_graph(i, 0.6)
            start = time.time()
            wl_lower_bound(G, H, k)
            end = time.time()
            wllb.append(end - start)
            start = time.time()
            wl_lower_bound(G, H, k, method="sinkhorn")
            end = time.time()
            wllb_sh.append(end - start)

            ig_G = ig.Graph.from_networkx(G)
            ig_H = ig.Graph.from_networkx(H)
            start = time.time()
            wwl.pairwise_wasserstein_distance([ig_G, ig_H], num_iterations=k)
            end = time.time()
            wwl_t.append(end - start)

            start = time.time()
            wl_k(G, H, k)
            end = time.time()
            wlk.append(end - start)
            #print("WL distance, exact", end - start)

            start = time.time()
            wl_k(G, H, k, method="sinkhorn")
            end = time.time()
            wlk_sh.append(end - start)
            #print("WL distance, sinkhorn", end - start)
            
        plt.plot(np.arange(5, 100), wllb, label="WL lower bound, exact, k =" + str(k))
        plt.plot(np.arange(5, 100), wllb_sh, label="WL lower bound, sinkhorn, k=" + str(k))
        plt.plot(np.arange(5, 100), wlk, label="k-WL distance,exact, k =" + str(k))
        plt.plot(np.arange(5, 100), wlk_sh, label="k-WL distance, sinkhorn, k=" + str(k))
        plt.plot(np.arange(5, 100), wwl_t,label= "WWL distance, iterations =" + str(k))
    plt.xlabel("Size of graph")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.savefig("time_comparison.png")
    

def calculate_pairwise_distances(G, y, k_step, dataset_name):
    print("Compting pairwise distances in train set.....")
    start = time.time()
    distances = mp_compute_dist_train(G, k_step, n_cpus=24)
    end = time.time()
    save_train_name = "/data/sam/" + dataset_name + "/dwlk/deg/distances_" + str(k_step)
    np.save(save_train_name, distances)
    return 0

def nearest_neighbor_exp(num_G,igraphs, y, dataset_name, wwl_features=None):
    k_step = [1, 2, 3, 4]
    iterations = 10
    graph_index = np.arange(num_G)
    acc = []
    std = []
    for i in range(1, 5):
       mat_wwl = wwl.pairwise_wasserstein_distance(igraphs, num_iterations=i)
       wwl_accuracies = []
       for i in range(iterations):
           train_index, test_index, y_train, y_test = train_test_split(np.arange(0, num_G), y, test_size=0.1, random_state=i)
           D_train = mat_wwl[train_index][:, train_index]
           D_test = mat_wwl[test_index][:, train_index]
           clf = KNeighborsClassifier(n_neighbors = 1, metric='precomputed')
           clf.fit(D_train, y_train)
           y_pred = clf.predict(D_test)
           wwl_accuracies.append(accuracy_score(y_test, y_pred))
       acc.append(np.mean(wwl_accuracies))
       std.append(np.std(wwl_accuracies))
    ind =np.argmax(acc)

    print("WWL average accuracy:", acc[ind], "std dev:", std[ind])

    for k in k_step:
        distance_fname = "/data/sam/" + dataset_name + "/dwlk/f3/distances_" + str(k) + ".npy"
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
        
        
def max_k_nearest_neighbor_exp(num_G, y, dataset_name):
    k_step = [1, 2, 3, 4]
    iterations = 10
    d_mats = []
    for k in k_step:
        distance_fname = "/data/sam/" + dataset_name + "/f2/distances_" + str(k) + ".npy"
        d_mats.append(np.load(distance_fname))
    combos = [(1, 2), (1, 2, 3), (1, 2, 3, 4)]
    diff = []
    for i in range(d_mats[0].shape[0]):
        for j in range(d_mats[0].shape[1]):
            diff.append(np.abs(d_mats[0][i][j] - d_mats[3][i][j]))
    print("average difference between k = 0 and k = 4:", np.mean(diff), "std dev:", np.std(diff))


if __name__ == "__main__":
    datasets = ["MUTAG", "PTC_MR", "PTC_FM","IMDB-BINARY", "IMDB-MULTI", "COX2", "PROTEINS"]
    ds_name = [ "mutag", "ptc_mr", "ptc_fm","imdb_b", "imdb_m", "cox2", "proteins"]

    for i in range(len(ds_name)):
        MUTAG = fetch_dataset(datasets[i], as_graphs=True)
        G = MUTAG.data
        nx_G = grakel_to_nx(G, include_attr=False)
        igraphs, graph_attrs = grakel_to_igraph(G, add_attr=False)
        y = MUTAG.target
        print("---- Results for: ", datasets[i], " ----")
        #max_k_nearest_neighbor_exp(len(G), y, ds_name[i])
        nearest_neighbor_exp(len(G),igraphs, y, ds_name[i])
        #for j in range(1, 5):
        #    calculate_pairwise_distances(nx_G, y, j, ds_name[i])
