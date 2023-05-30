from sklearn.datasets import load_svmlight_file, load_digits
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
from lightning.classification import AdaGradClassifier, SVRGClassifier, SGDClassifier, SAGAClassifier, SAGClassifier
from functools import partial

from sklearn.model_selection import train_test_split

def load_data(dataset, train = False):
    if dataset in ['a9a', 'w8a']:
        X_train, y_train = load_svmlight_file('../datasets/{dataset}/{dataset}'.format(dataset = dataset))
        X_test, y_test = load_svmlight_file('../datasets/{dataset}/{dataset}.t'.format(dataset = dataset))
        
        if dataset == 'a9a' : X_test = np.append(np.array(X_test.todense()), np.zeros((X_test.shape[0], 1)), axis=1)
        
        return X_train, y_train, X_test, y_test

    if dataset == 'gisette':
        X_train, y_train = load_svmlight_file('../datasets/{dataset}/{dataset}.bz2'.format(dataset = dataset))
        X_test, y_test = load_svmlight_file('../datasets/{dataset}/{dataset}.t.bz2'.format(dataset = dataset))

        return X_train, y_train, X_test, y_test
     
    if dataset == 'mnist':
        X, y = load_digits(return_X_y=True)
        
        # train test split sklearn
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        return X_train, y_train, X_test, y_test

    if dataset == 'news20':
        X, y = load_svmlight_file('../datasets/{dataset}/{dataset}.bz2'.format(dataset = dataset))

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        return X_train, y_train, X_test, y_test

    if dataset == 'australian':
        X, y = load_svmlight_file('../datasets/{dataset}/{dataset}'.format(dataset = dataset))

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        return X_train, y_train, X_test, y_test

    if dataset == 'diabetes':
        X, y = load_svmlight_file('../datasets/{dataset}/{dataset}'.format(dataset = dataset))

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

        return X_train, y_train, X_test, y_test

def train_test_model(optimizer, loss, X_train, y_train, X_test, y_test, eta):
    clf = SGDClassifier(loss = loss, penalty= 'l2',  eta0 = eta, max_iter = 100, random_state = 0, alpha = 1e-2)
    if optimizer == 'sgd':
        clf = SGDClassifier(loss = loss, penalty= 'l2',  eta0 = eta, max_iter = 100, random_state = 0, alpha = 1e-2)
    if optimizer == 'adagrad':
        clf = AdaGradClassifier(loss = loss, l1_ratio = 0,  eta = eta, n_iter = 100, random_state = 0, alpha = 1e-2)
    if optimizer == 'svrg':
        clf = SVRGClassifier(loss = loss, eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    if optimizer == 'saga':
        clf = SAGAClassifier(loss = loss, beta = 0, eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    if optimizer == 'sag':
        clf = SAGClassifier(loss = loss, beta = 0, eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    
    clf.fit(X_train, y_train)
    return 1 - clf.score(X_test, y_test)


def run_hyperopt(dataset, max_evals = 100, loss = 'log'):
    hyper_params = {}
    for optimizer in ['sgd', 'svrg', 'saga', 'adagrad', 'sag']:

        X_train, y_train, X_test, y_test = load_data(dataset)
        # X_test, y_test = load_data(dataset, train = False)

        space = hp.uniform('eta', 1e-4, 1)

        best = fmin(fn = partial(train_test_model, optimizer, loss, X_train, y_train, X_test, y_test), space = space,
                    max_evals = max_evals) 

        # print
        print(dataset, optimizer, best)
        hyper_params[optimizer] = best['eta']
    return hyper_params

if __name__ == '__main__':

    with open("output.txt", "w") as f:
        
        for dataset in ['a9a', 'w8a', 'gisette', 'diabetes', 'australian', 'news20']:
            for optimizer in [ 'svrg', 'saga', 'adagrad', 'sag']:

                X_train, y_train, X_test, y_test = load_data(dataset)
                # X_test, y_test = load_data(dataset, train = False)

                space = hp.uniform('eta', 1e-4, 1)

                best = fmin(fn = partial(train_test_model, optimizer, 'log', X_train, y_train, X_test, y_test), space = space,
                            max_evals = 50) 

                # print
                print(optimizer, dataset, best)
                f.write(f"{dataset}, {optimizer}, {best}\n")
        


