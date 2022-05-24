import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models
import pandas as pd
from network_tests import grad_test_w, grad_test_b, jacobian_test_w1_resnet, jacobian_test_w2_resnet, jacobian_test_b, jacobian_test_w, grad_test_whole_network_w, grad_test_whole_network_b, jacobian_test_b1_resnet, grad_test_whole_resnet_network_w, grad_test_whole_resnet_network_b,  jacobian_test_b2_resnet
from LinearRegression import loss_func_SGD
from matplotlib import pyplot as plt
import time

def read_mat(name):
    from scipy.io import loadmat
    return loadmat(name)

if __name__=="__main__":
    # SGD test on least squares
    theta = np.random.rand(2)  # m, b: mx+b
    data = np.random.rand(2, 10)
    x = data[0, :]
    y = data[1,:]
    for i in range(100000):
        theta = loss_func_SGD(data,theta,learning_rate=0.1)
    m,b = theta
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [min(m * x + b), max(m * x + b)], color='green')  # regression line
    plt.show()

    # HyperParams
    batch_size = 20
    num_epochs = 300
    num_classes = 2
    hidden_units = 100
    hidden_units2 = 10
    hidden_units3 = 10
    dimensions = 2
    network = 'ResNet'  # or FeedForward

    # PeaksData  da, SwissRollData, GMMData
    X_train, y_train, X_test, y_test = utils.get_data('SwissRollData')
    X_train, y_train = shuffle(X_train, y_train)

    # 2.2.5
    # X_train= X_train[:200,:]
    # y_train= y_train[:200]

    # test of softmax (2.1.3))
    softmax_model = models.MyNeuralNetwork()
    softmax_model.add(layers.Softmax(dimensions, 2))
    optimizer = optimizers.SGD(softmax_model.parameters, lr=0.01)
    losses, train_accuracy, test_accuracy = softmax_model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs,
                                                      optimizer)
    utils.plot_scores(train_accuracy, test_accuracy)


    # gradient and jacobian tests
    grad_test_w(X_train, y_train)  # for the softmax
    grad_test_b(X_train, y_train)  # for the softmax

    jacobian_test_w(X_train, y_train)  # jacobian tests
    jacobian_test_b(X_train, y_train)

    jacobian_test_w1_resnet(X_train, y_train)  # jacobian test sfo resnet
    jacobian_test_w2_resnet(X_train, y_train)
    jacobian_test_b1_resnet(X_train, y_train)
    jacobian_test_b2_resnet(X_train, y_train)

    grad_test_whole_network_w(X_train, y_train, layer_number=1)  # testing the whole networks
    grad_test_whole_network_b(X_train, y_train, layer_number=1)
    grad_test_whole_resnet_network_w(X_train, y_train, layer_number=1)
    grad_test_whole_resnet_network_b(X_train, y_train, layer_number=1)


    # running the model
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



