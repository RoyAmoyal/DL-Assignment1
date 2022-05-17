import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models
import pandas as pd
from network_tests import grad_test_W, grad_test_b, jacobian_test_W, jacobian_test_b, grad_test_W_whole_network, grad_test_b_whole_network

def read_mat(name):
    from scipy.io import loadmat
    return loadmat(name)

if __name__=="__main__":


    # mat = read_mat('./Data/SwissRollData.mat')
    # print(mat)
    # X = pd.DataFrame(mat['Yt']).to_numpy()
    # print(X)
    # print(X.shape)
    # C = pd.DataFrame(mat['Ct']).to_numpy().T
    batch_size  = 50
    num_epochs = 100
    num_classes = 2
    hidden_units = 100
    hidden_units2 = 10
    dimensions = 2

    # PeaksData  da, SwissRollData, GMMData
    X_train, y_train, X_test, y_test = utils.get_data('PeaksData')
    X_train, y_train = shuffle(X_train, y_train)

    # gradient and jacobian tests
    grad_test_W(X_train, y_train)
    grad_test_b(X_train, y_train)
    jacobian_test_W(X_train, y_train)
    jacobian_test_b(X_train, y_train)
    grad_test_W_whole_network(X_train, y_train)
    grad_test_b_whole_network(X_train, y_train)

    model = models.MyNeuralNetwork()
    model.add(layers.Linear(dimensions, hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units, 5))
    optimizer = optimizers.SGD(model.parameters, lr=0.1)
    losses, train_accuracy, test_accuracy = model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer)

    # plotting
    utils.plot_scores(train_accuracy, test_accuracy)



