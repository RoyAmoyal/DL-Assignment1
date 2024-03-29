from activations import Abstract_Activation
import numpy as np
import utils


class Abstract_Model(object):
    def forward(self, X):
        raise NotImplementedError

    def backward(self, true_label):
        raise NotImplementedError


class MyNeuralNetwork(Abstract_Model):
    def __init__(self):
        self.graph = []
        self.parameters = []

    def add(self, layer):
        self.graph.append(layer)
        if not isinstance(layer, Abstract_Activation):  # in case of adding an activation
            self.parameters += layer.parameters()

    def init(self):
        for f in self.graph:
            if f.type == 'linear' or f.type == 'softmax':
                weights, bias = f.parameters()
                weights.data = .01 * np.random.random((weights.data.shape[0], weights.data.shape[1]))
                bias.data = np.zeros((1, weights.data.shape[1]))
            if f.type == 'resblock':
                weights1, weights2, bias1, bias2 = f.parameters()
                weights1.data = .01 * np.random.random((weights1.data.shape[0], weights1.data.shape[1]))
                weights2.data = .01 * np.random.random((weights2.data.shape[0], weights2.data.shape[1]))
                bias1.data = np.zeros((1, weights1.data.shape[1]))
                bias2.data = np.zeros((1, weights2.data.shape[1]))

    def fit(self, X_train, y_train, X_test, y_test, batch_size, num_epochs, optimizer):
        loss_history = []
        train_accuracy = []
        test_accuracy = []
        self.init()
        data_gen = utils.DataGenerator(X_train, y_train, batch_size)
        itr = 0
        epochs_counter = 0
        for epoch in range(num_epochs):
            epoch_iter = 0
            epoch_accuracy = []
            for X, Y in data_gen:
                optimizer.zeroGrad()
                probabilities = self.forward(X)
                loss = utils.cross_entropy_loss(probabilities, Y)
                self.backward(Y)
                loss_history += [loss]
                itr += 1
                epoch_iter += 1
                optimizer.step()
                epoch_acc = self.evaluate(X, Y)
                epoch_accuracy.append(epoch_acc)
            train_acc = np.array(epoch_accuracy).sum()/epoch_iter
            train_accuracy.append(train_acc)
            test_acc = self.evaluate(X_test, y_test)
            test_accuracy.append(test_acc)
            print("epoch = {}, train accuracy = {} test accuracy = {}".format(epoch, train_acc, test_acc))
            epochs_counter += 1
            if epochs_counter == 80:
                optimizer.update_lr()
                epochs_counter = 0
        return loss_history, train_accuracy, test_accuracy

    def forward(self, X):
        for f in self.graph: X = f.forward(X)
        return X

    def backward(self, true_label):
        grad = true_label
        for f in self.graph[::-1]:
            grad = f.backward(grad)

    def evaluate(self, X_test, y_test):
        predicted_labels = np.argmax(self.predict(X_test), axis=1)
        accuracy = np.sum(predicted_labels == y_test) / len(y_test)
        return accuracy

    def predict(self, data):
        X = data
        for f in self.graph: X = f.forward(X)
        return X
