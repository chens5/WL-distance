from grakel import GraphKernel, Graph, WeisfeilerLehman, VertexHistogram
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
from wtk.utilities import krein_svm_grid_search, KreinSVC
sys.path.insert(1, './utils/')
from distances import wl_lower_bound, wl_lb_distance_matrices
from tqdm import tqdm, trange
import multiprocessing as mp

def markov_chain_lb_kernel(G1, G2, k, lam):
    dist = wl_lower_bound(G1, G2, k)
    return np.exp(-lam * dist)

def compute_kernel_matrix(graph_data, k, lam):
    pairs = []
    n = len(graph_data)
    for pairs_of_indexes in itertools.combinations(range(0, n),  2):
        pairs.append(pairs_of_indexes)
    kernel_matrix = np.zeros((n, n))
    dist_mat = np.zeros((n, n))
    for pair in tqdm(pairs):
        G1 = graph_data[pair[0]]
        G2 = graph_data[pair[1]]
        dist_mat[pair[0]][pair[1]] = wl_lower_bound(G1, G2, k)
        #kernel = markov_chain_lb_kernel(G1, G2, k, lam)
        #kernel_matrix[pair[0]][pair[1]] = kernel
        #kernel_matrix[pair[1]][pair[0]] = kernel
    return dist_mat


def compute_kernel_test(G_test, G_train, k, lam):
    n = len(G_test)
    m = len(G_train)
    kernel_matrix = np.zeros((n, m))
    for i in trange(n):
        G1 = G_test[i]
        for j in range(m):
            G2 = G_train[j]
            kernel = wl_lower_bound(G1, G2, k)
            #kernel= markov_chain_lb_kernel(G1, G2, k, lam)
            kernel_matrix[i][j] = kernel
    return kernel_matrix

def mp_compute_dist_train(graph_data, k, n_cpus = 10):
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
    dist_matrix = np.zeros((n, n))
    for i in range(len(pairs)):
        pair = pairs[i]
        dist_matrix[pair[0]][pair[1]] = results[i]
        dist_matrix[pair[1]][pair[0]] = results[i]
    return dist_matrix

def mp_compute_dist_test(G_test, G_train, k, n_cpus = 10):
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

def grakel_to_nx(G):
    nx_G = []
    for graph in G:
        adj_mat = graph.get_adjacency_matrix()
        nx_G.append(nx.from_numpy_matrix(adj_mat))
    return nx_G

def run_mlb_svm(G, y):
    k_steps = [1, 2, 3, 4]
    gammas = [0.01, 0.1, 1, 10]
    random_states = [23, 42, 64, 73, 91]
    for k in k_steps:
        sum_acc = []
        for rs in random_states:
            G_train, G_test, y_train, y_test = train_test_split(G, y, test_size = 0.2, random_state=23)
            D_train_name = '/data/sam/mutag/' + 'D_train' + str(rs) + '_' + str(k)
            D_test_name = '/data/sam/mutag/' + 'D_test' + str(rs) + '_' + str(k)
            #D_train = compute_kernel_matrix(G_train, k, 0.1)
            #D_test = compute_kernel_test(G_test, G_train, k, 0.1)
            g_acc = -np.inf
            gam = None
            for g in tqdm(gammas):
                K_train = np.exp(-g * D_train)
                K_test = np.exp(-g * D_test)
                clf = SVC(kernel='precomputed')
                clf.fit(K_train, y_train)
                y_pred = clf.predict(K_test)
                accuracy = accuracy_score(y_test, y_pred)
                print("accuracy", accuracy)
                if g_acc < accuracy:
                    g_acc = accuracy
                    gam = g
            print("k = ", k, "gamma = ", gam, "accuracy = ", g_acc)
            sum_acc.append(g_acc)
        print("k = ", k, "average accuracy = ", np.mean(sum_acc), "std = ", np.std(sum_acc))

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
def run_grakel_svm(kernel, G, y, random_state=23):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=random_state)

    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return accuracy_score(y_test, y_pred)

def experiments(k, lam):
    #kernels = ["random_walk", "shortest_path", "weisfeiler_lehman_optimal_assignment", "weisfeiler_lehman"]
    MUTAG = fetch_dataset("PROTEINS",as_graphs=True )
    G = MUTAG.data
    nx_G = grakel_to_nx(G)
    y = MUTAG.target
    #sp_kernel = GraphKernel(kernel="shortest_path")
    #print("Running SVC with shortest path")
    #start = time.time()
    #random_states = [23, 42, 64, 73, 91]
    #acc = []
    #for rs in random_states:
    #    acc.append(run_grakel_svm(sp_kernel, G, y, random_state=rs))

    #print("Accuracy:", np.mean(acc), "std:", np.std(acc))
    #end = time.time()
    #print("Done in:", end - start)

    #print("Running SVC with Weisfeiler-Lehman Kernel")
    #wl_subtree_kernel = WeisfeilerLehman(n_iter=20, base_graph_kernel=VertexHistogram, normalize=True)
    #start = time.time()
    #print("Accuracy:", run_grakel_svm(wl_subtree_kernel, G, y))
    #end = time.time()
    #print("Done in:", end - start)
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    Cs = [0.01, 0.1, 1, 10, 100]
    random_states = [23, 42, 64, 73, 91]
    k_step = [1, 2, 3, 4]
    for k in k_step:
        sum_acc = []
        per_param_avg = []
        per_param_std = []
        for g in gammas:
            for c in Cs:
                avg = []
                for rs in random_states:
                    G_train, G_test, y_train, y_test = train_test_split(nx_G, y, test_size=0.2, random_state=rs)
                    D_train_name = "/data/sam/proteins/point7/f2/D_train" + str(rs) + "_" + str(k) + ".npy"
                    D_test_name = "/data/sam/proteins/point7/f2/D_test" + str(rs) + "_" + str(k) + ".npy"
                    D_train = np.load(D_train_name)
                    D_test = np.load(D_test_name)
                    K_train = np.exp(-g * D_train)
                    K_test = np.exp(-g * D_test)
                    clf = SVC(kernel = 'precomputed', C=c, max_iter=1000)
                    clf.fit(K_train, y_train)
                    y_pred =clf.predict(K_test)
                    #print(len(y_pred))
                    #print(len(y_test))
                    avg.append(accuracy_score(y_test, y_pred))
                per_param_avg.append(np.mean(avg))
                per_param_std.append(np.std(avg))
        ind = np.argmax(per_param_avg)
        print("k = ", k, "Average = ", per_param_avg[ind], "std = ", per_param_std[ind])


    # print("Running SVC with lower bound kernel")
    # start = time.time()
    # print(run_markov_chain_svm(nx_G, y, k , lam))
    # end = time.time()
    # print("Done in:", end - start)

    #sp_kernel.fit_transform(G[:5])
    #tg = Graph(G[0])
    #print(G[0].get_adjacency_matrix())


if __name__ == "__main__":
    experiments(1, 1)
    #MUTAG = fetch_dataset("MUTAG", as_graphs=True)
    #G = MUTAG.data[:30]
    #y = MUTAG.target[:30]
    #nx_G = grakel_to_nx(G)
    
