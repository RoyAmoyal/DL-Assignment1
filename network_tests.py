import numpy as np
import matplotlib.pyplot as plt
import models
import layers
import copy
import utils
import activations
from numpy import linalg as LA


def grad_test_w(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Softmax(softmax_in, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((2, 5))
    d = d / np.sum(d)
    grad_diff = []
    sec_grad = []
    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epsk in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d*epsk  # changing x
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[0].weights.grad
        print(d.flatten().T.shape)
        print(grad_x.flatten().shape)
        sec_grad.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    plt.plot(x, sec_grad, label='first order')
    plt.legend()
    plt.title('Softmax gradient test for the weights')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()


def grad_test_b(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Softmax(softmax_in, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((1, 5))
    d = d / np.sum(d)
    grad_diff = []
    sec_grad = []
    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epsk in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d*epsk  # changing x
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))
    # fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[0].bias.grad
        print("first", d.flatten().T.shape)
        print("second", grad_x.flatten().shape)
        sec_grad.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    plt.plot(x, sec_grad, label='first order')
    plt.legend()
    plt.title('Softmax gradient test for the biases')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()


def jacobian_test_w(X_train, y_train, network="Linear"):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 10
    model = models.MyNeuralNetwork()
    if network == 'Linear':
        model.add(layers.Linear(softmax_in, hidden_units))
    else:  # resnet
        model.add(layers.ResBlock(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((2, 10))
    d = d / np.sum(d)

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    grad_diff = []

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d*epss
        probabilities_grad2 = model2.forward(x_data)

        f_x_eps_d = model2.graph[1].activation_output
        f_x = model_grad.graph[1].activation_output

        grad_diff.append(LA.norm(f_x_eps_d - f_x))
    print(grad_diff)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')

    new_grad = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model_grad.backward(x_label)
        f_x_eps_d = model2.graph[1].activation_output
        f_x = model_grad.graph[1].activation_output
        grad = model_grad.graph[0].weights.grad
        JacMV = epss * np.matmul(d.T, grad)
        diff = LA.norm(f_x_eps_d - f_x - JacMV)
        new_grad.append(diff*epss)

    plt.plot(x, new_grad, label='first order')
    plt.legend()
    plt.title('Jacobian test for the weights')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()


def jacobian_test_b(X_train, y_train, network="Linear"):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 10
    model = models.MyNeuralNetwork()
    if network == 'Linear':
        model.add(layers.Linear(softmax_in, hidden_units))
    else:  # resnet
        model.add(layers.ResBlock(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((1, 10))
    d = d / np.sum(d)

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    grad_diff = []

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d*epss
        probabilities_grad2 = model2.forward(x_data)

        f_x_eps_d = model2.graph[1].activation_output
        f_x = model_grad.graph[1].activation_output

        grad_diff.append(LA.norm(f_x_eps_d - f_x))
    print(grad_diff)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')

    new_grad = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model_grad.backward(x_label)
        f_x_eps_d = model2.graph[1].activation_output
        f_x = model_grad.graph[1].activation_output
        grad = model_grad.graph[0].bias.grad
        JacMV = epss * np.matmul(d.T, grad)
        diff = LA.norm(f_x_eps_d - f_x - JacMV)
        new_grad.append(diff*epss)

    plt.plot(x, new_grad, label='first order')
    plt.legend()
    plt.title('Jacobian test for the biases')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()


def grad_test_whole_network_w(X_train, y_train, network='Linear'):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 5
    model = models.MyNeuralNetwork()
    if network == 'Linear': # resnet
        model.add(layers.Linear(softmax_in, hidden_units))
    else:
        model.add(layers.ResBlock(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((5, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[2].weights.data += d * epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    print(grad_diff)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')
    new_grad = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[2].weights.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[2].weights.grad
        new_grad.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    plt.plot(x, new_grad, label='first order')
    plt.legend()
    plt.title('Gradient test for the whole network (weights)')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()


def grad_test_whole_network_b(X_train, y_train, network='Linear'):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 5
    model = models.MyNeuralNetwork()
    if network == 'Linear': # resnet
        model.add(layers.Linear(softmax_in, hidden_units))
    else:
        model.add(layers.ResBlock(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(8)])

    d = np.random.random((1, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[2].bias.data += d * epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    print(grad_diff)
    plt.figure(figsize=(12, 8))
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.plot(x, grad_diff, label='zero order')
    new_grad = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[2].bias.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[2].bias.grad
        new_grad.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    plt.plot(x, new_grad, label='first order')
    plt.legend()
    plt.title('Gradient test for the whole network (biases)')
    plt.ylabel('error')
    plt.xlabel('k')
    plt.semilogy([1, 2, 3, 4, 5, 6, 7, 8], grad_diff)
    plt.show()




