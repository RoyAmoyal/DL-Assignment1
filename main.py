import utils
import numpy as np
from sklearn.utils import shuffle
import activations
import layers
import optimizers
import models
import pandas as pd
from network_tests import grad_test_W, grad_test_b, jacobian_test_W, jacobian_test_b, grad_test_W_whole_network, grad_test_b_whole_network
from LinearRegression import loss_func_SGD
from matplotlib import pyplot as plt
import time
def read_mat(name):
    from scipy.io import loadmat
    return loadmat(name)

if __name__=="__main__":
    theta = np.random.rand(2)  # m, b: mx+b
    data = np.random.rand(2, 10)
    x = data[0, :]
    y = data[1,:]
    for i in range(10000):
        theta = loss_func_SGD(data,theta,learning_rate=0.1)
    m,b = theta

    print(x)
    print(y)
    plt.scatter(x, y)
    plt.plot([min(x), max(x)], [min(m * x + b), max(m * x + b)], color='green')  # regression line
    plt.show()

    # HyperParams
    batch_size = 100
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
    #X_train= X_train[:200,:]
    #y_train= y_train[:200]

    # gradient and jacobian tests
    grad_test_W(X_train, y_train)
    grad_test_b(X_train, y_train)
    jacobian_test_W(X_train, y_train)
    jacobian_test_b(X_train, y_train)
    grad_test_W_whole_network(X_train, y_train)
    grad_test_b_whole_network(X_train, y_train)

    model = models.MyNeuralNetwork()
    if network == 'ResNet':
        model.add(layers.ResBlock(dimensions, hidden_units))
        # model.add(activations.ReLU())
        # model.add(layers.Linear(hidden_units, hidden_units2))
    else:
        model.add(layers.Linear(dimensions, hidden_units))
    model.add(activations.ReLU())
    model.add(layers.Softmax(hidden_units, 2))
    optimizer = optimizers.SGD(model.parameters, lr=1)
    losses, train_accuracy, test_accuracy = model.fit(X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer)

    # plotting
    utils.plot_scores(train_accuracy, test_accuracy)



