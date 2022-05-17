import numpy as np
import matplotlib.pyplot as plt
import models
import layers
import copy
import utils
import activations
from numpy import linalg as LA

def grad_test_W(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Softmax(softmax_in, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    d = np.random.random((2, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d*epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Gradient test by W', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].weights.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[0].weights.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

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
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    d = np.random.random((1, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d*epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Gradient test by b', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[0].bias.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[0].bias.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()

def jacobian_test_W(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 10
    model = models.MyNeuralNetwork()
    model.add(layers.Linear(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Jacobian test by W', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$||f(x+\epsilon d) - f(x)||$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
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
        grad_diff.append(diff*epss)

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$||f(x+\epsilon d) - f(x) -  JavMV(x, \epsilon d)||$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()

def jacobian_test_b(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 10
    model = models.MyNeuralNetwork()
    model.add(layers.Linear(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

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

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Jacobian test by b', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$||f(x+\epsilon d) - f(x)||$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
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
        grad_diff.append(diff*epss)

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$||f(x+\epsilon d) - f(x) -  JavMV(x, \epsilon d)||$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()

def grad_test_W_whole_network(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Linear(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    d = np.random.random((5, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[2].weights.data += d*epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Full network gradient test by W', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[2].weights.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[2].weights.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()

def grad_test_b_whole_network(X_train, y_train):
    softmax_in = 2
    softmax_out = 5
    hidden_units = 5
    model = models.MyNeuralNetwork()
    model.add(layers.Linear(softmax_in, hidden_units))
    model.add(activations.Tanh())
    model.add(layers.Softmax(hidden_units, softmax_out))
    model.init()
    for p in model.parameters:
        p.grad = 0.

    eps0 = 1
    eps = np.array([(0.5 ** i) * eps0 for i in range(10)])

    d = np.random.random((1, 5))
    d = d / np.sum(d)
    grad_diff = []

    x_data = np.array([X_train[0]])
    x_label = np.array([y_train[0]])

    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = model_grad.forward(x_data)
        model2 = copy.deepcopy(model)
        model2.graph[2].bias.data += d*epss
        probabilities_grad2 = model2.forward(x_data)
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label)))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle('Full network gradient test by b', fontsize=16)

    axs[0, 0].plot(eps, grad_diff)
    axs[0, 0].set_xlabel('$\epsilon$')
    axs[0, 0].set_title('$|f(x+\epsilon d) - f(x)|$')

    axs[0, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[0, 1].set_xlabel('$i$')
    axs[0, 1].set_title('rate of decrease')
    axs[0, 1].set_ylim([0, 1])

    grad_diff = []
    for epss in eps:
        model_grad = copy.deepcopy(model)
        probabilities_grad = copy.deepcopy(model_grad.forward(x_data))
        model2 = copy.deepcopy(model)
        model2.graph[2].bias.data += d * epss
        probabilities_grad2 = copy.deepcopy(model2.forward(x_data))
        model2.backward(x_label)
        grad_x = model2.graph[2].bias.grad
        grad_diff.append(np.abs(utils.cross_entropy_loss(probabilities_grad2, x_label) -
                                utils.cross_entropy_loss(probabilities_grad, x_label) -
                                epss * np.dot(d.flatten().T, grad_x.flatten())))

    axs[1, 0].plot(eps, grad_diff)
    axs[1, 0].set_xlabel('$\epsilon$')
    axs[1, 0].set_title('$|f(x+\epsilon d) - f(x) - \epsilon d^{T} grad(x)|$')

    axs[1, 1].plot(range(len(grad_diff) - 1), [grad_diff[i + 1] / grad_diff[i] for i in range(len(grad_diff) - 1)])
    axs[1, 1].set_xlabel('$i$')
    axs[1, 1].set_title('rate of decrease')
    axs[1, 1].set_ylim([0, 1])

    plt.show()
