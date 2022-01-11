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

def svm_experiment(num_G, y, dataset_name, f):
    gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #train_test_split(np.arange(num_G), y, test_size = 0.1)
    skf = StratifiedKFold(n_splits = 10)
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
        print("Finished with", datasets[i])



def experiments(k, lam):
    #kernels = ["random_walk", "shortest_path", "weisfeiler_lehman_optimal_assignment", "weisfeiler_lehman"]
    MUTAG = fetch_dataset("PROTEINS",as_graphs=True )
    G = MUTAG.data
    nx_G = grakel_to_nx(G)
    y = MUTAG.target

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
    new_experiments()
    #MUTAG = fetch_dataset("MUTAG", as_graphs=True)
    #G = MUTAG.data[:30]
    #y = MUTAG.target[:30]
    #nx_G = grakel_to_nx(G)
