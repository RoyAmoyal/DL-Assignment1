import numpy as np


class Abstract_Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Abstract_Activation):
    activation_output: np.ndarray

    def __init__(self):
        self.type = 'activation'
        self.activation_output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x[x < 0] = 0.
        self.activation_output = x
        return self.activation_output

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        return d_y * (self.activation_output > 0)


class Tanh(Abstract_Activation):

    def __init__(self):
        self.type = 'activation'

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activation_output = np.tanh(x)
        return self.activation_output

    def backward(self, d_y: np.ndarray) -> np.ndarray:
        self.activation_grad = 1.0 - np.tanh(d_y) ** 2
        return self.activation_grad
