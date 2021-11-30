from grakel import GraphKernel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_svm(kernel, G, y):
    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.1, random_state=23)
    K_train = kernel.fit_transform(G_train)
    K_test = kernel.transform(G_test)

    clf = SVC(kernel='precomputed')
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    return accuracy_score(y_test, y_pred)

def experiments():
    kernels = ["random_walk", "shortest_path", "weisfeiler_lehman_optimal_assignment", "weisfeiler_lehman"]
