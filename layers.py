import numpy as np
from activations import ReLU


class Tensor:
    def __init__(self, shape):
        self.data = np.ndarray(shape, np.float32)
        self.grad = np.ndarray(shape, np.float32)


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


class ResNet(Abstract_Layer):

    def __init__(self, in_nodes1, out_nodes1, in_nodes2, out_nodes2):
        self.type = 'linear'
        self.weights1 = Tensor((in_nodes1, out_nodes1))
        self.weights2 = Tensor((in_nodes2, out_nodes2))
        self.bias1 = Tensor((1, out_nodes1))
        self.bias2 = Tensor((1, out_nodes2))
        self.relu1 = ReLU()
        self.relu2 = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return self.relu2.forward(
            x @ self.weights2 + self.relu1.forward(np.dot(x, self.weights1.data) + self.bias1.data) + self.bias2)

    def parameters(self):
        return [self.weights1, self.weights2, self.bias1, self.bias2]
