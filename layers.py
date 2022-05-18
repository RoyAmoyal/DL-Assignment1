import numpy as np
from activations import ReLU


class Tensor:
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)  # here keep the weights
        self.grad = np.ndarray(shape, np.float32)  # here keep the grads of the weights


class Abstract_Layer(object):
    input: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self):
        return []


class Linear(Abstract_Layer):
    def __init__(self, in_nodes, out_nodes):
        self.type = 'linear'
        self.weights = Tensor((in_nodes, out_nodes))
        self.bias = Tensor((1, out_nodes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights.data) + self.bias.data

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        self.weights.grad += np.dot(self.input.T, d_y)
        self.bias.grad += np.sum(d_y, axis=0, keepdims=True)
        return np.dot(d_y, self.weights.data.T)

    def parameters(self):
        return [self.weights, self.bias]


class Softmax(Abstract_Layer):
    probabilities: np.ndarray
    true_label: np.ndarray

    def __init__(self, in_nodes, out_nodes):
        self.weights = Tensor((in_nodes, out_nodes))
        self.bias = Tensor((1, out_nodes))
        self.type = 'softmax'

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        linear_output = np.dot(x, self.weights.data) + self.bias.data
        logits = np.exp(linear_output - np.max(linear_output, axis=1, keepdims=True))
        self.probabilities = logits / np.sum(logits, axis=1, keepdims=True)
        return self.probabilities

    def backward(self, true_label: np.ndarray) -> np.ndarray:
        error = self.probabilities
        # print("error: ",error)
        error[range(len(true_label)), true_label] -= 1.0  # exponents - C
        # print("new error: ",error)

        error /= len(true_label)

        self.weights.grad += np.dot(self.input.T, error)
        self.bias.grad += np.sum(error, axis=0, keepdims=True)
        return np.dot(error, self.weights.data.T)

    def parameters(self):
        return [self.weights, self.bias]


class ResBlock(Abstract_Layer):  # with only 1 relu

    def __init__(self, in_nodes, out_nodes):
        self.type = 'resblock'
        self.weights1 = Tensor((in_nodes, out_nodes))
        self.weights2 = Tensor((in_nodes, out_nodes))
        self.bias1 = Tensor((1, out_nodes))
        self.bias2 = Tensor((1, out_nodes))
        self.relu = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights2.data) + self.relu.forward(np.dot(x, self.weights1.data) + self.bias1.data) + self.bias2.data

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        temp_x = self.input
        self.weights1.grad += np.dot((self.input * np.where(temp_x > 0, temp_x, 0)).T, d_y)
        self.weights2.grad += np.dot(self.input.T, d_y)
        self.bias1.grad += np.sum(np.dot((np.where(temp_x > 0, temp_x, 0)).T, d_y), axis=0, keepdims=True)
        self.bias2.grad += np.sum(d_y, axis=0, keepdims=True)
        return np.dot(d_y, self.weights1.data.T) * np.where(temp_x > 0, temp_x, 0) + np.dot(d_y, self.weights2.data.T)

    def parameters(self):
        return [self.weights1, self.weights2, self.bias1, self.bias2]
