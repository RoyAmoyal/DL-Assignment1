import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models
import pandas as pd
from network_tests import grad_test, jacobian_test, grad_test_whole_network

def read_mat(name):
    from scipy.io import loadmat
    return loadmat(name)

if __name__=="__main__":
    # HyperParams
    batch_size = 50
    num_epochs = 100
    num_classes = 2
    hidden_units = 100
    hidden_units2 = 10
    dimensions = 2
    network = 'ResNet'  # or FeedForward

    # PeaksData  da, SwissRollData, GMMData
    X_train, y_train, X_test, y_test = utils.get_data('SwissRollData')
    X_train, y_train = shuffle(X_train, y_train)

    # test of softmax (2.1.3))
    # softmax_model = models.MyNeuralNetwork()
    # softmax_model.add(layers.Softmax(dimensions, 2))
    # optimizer = optimizers.SGD(softmax_model.parameters, lr=1)
    # losses, train_accuracy, test_accuracy = softmax_model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs,
    #                                                   optimizer)
    # utils.plot_scores(train_accuracy, test_accuracy)

    # gradient and jacobian tests
    grad_test(X_train, y_train)
    jacobian_test(X_train, y_train)
    grad_test_whole_network(X_train, y_train, network='Linear')

    model = models.MyNeuralNetwork()
    if network == 'ResNet':
        model.add(layers.ResBlock(dimensions, hidden_units))
        model.add(layers.ReLU())
        model.add(layers.Linear(hidden_units, hidden_units2))
    else:
        model.add(layers.Linear(dimensions, hidden_units2))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units2, 2))
    optimizer = optimizers.SGD(model.parameters, lr=1)
    losses, train_accuracy, test_accuracy = model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer)
    # plotting
    utils.plot_scores(train_accuracy, test_accuracy)



