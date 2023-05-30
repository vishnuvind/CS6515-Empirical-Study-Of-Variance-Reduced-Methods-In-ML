from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp
from lightning.classification import SVRGClassifier
from functools import partial

def load_data(dataset, train = False):
    if dataset in ['a9a', 'w8a']:
        if train:
            return load_svmlight_file('../datasets/{dataset}/{dataset}'.format(dataset = dataset))
        else:
            return load_svmlight_file('../datasets/{dataset}/{dataset}.t'.format(dataset = dataset))



def train_test_model(dataset, X_train, y_train, X_test, y_test, eta):
    clf = SVRGClassifier(loss = 'log', eta = eta, max_iter = 100, random_state = 0, alpha = 1e-2, tol = 1e-24)
    clf.fit(X_train, y_train)
    if dataset == 'a9a' : X_test = np.append(np.array(X_test.todense()), np.zeros((X_test.shape[0], 1)), axis=1)
    return 1 - clf.score(X_test, y_test)


if __name__ == '__main__':

    for dataset in ['a9a', 'w8a']:
        X_train, y_train = load_data(dataset, train = True)
        X_test, y_test = load_data(dataset, train = False)

        space = hp.uniform('eta', 1e-7, 1e-1)

        best = fmin(fn = partial(train_test_model, dataset, X_train, y_train, X_test, y_test), space = space,
                    max_evals = 100) 

        print(best)

