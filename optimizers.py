
class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

    def zeroGrad(self):
        for p in self.parameters:
            p.grad = 0.


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data = p.data - self.lr * p.grad

    def update_lr(self):
        self.lr /= 10
        print("Updating learning rate to ",self.lr)

