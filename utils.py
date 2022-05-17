import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def plot_scores(accuracy_scores_train, accuracy_scores_test):
    plt.plot(range(len(accuracy_scores_train)), accuracy_scores_train)
    plt.plot(range(len(accuracy_scores_test)), accuracy_scores_test)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(ymin=0, ymax=1)
    plt.legend(["train", "test"])
    plt.title('accuracy over epochs')
    plt.show()


def cross_entropy_loss(X, target):
    probability = X
    loss = -np.log(probability[range(len(target)), target])
    return loss.mean()


def get_data(dataset_name):
    SwissRoll = scipy.io.loadmat('data/' + dataset_name + '.mat')
    y_train = np.array(SwissRoll['Ct'])
    X_train = np.array(SwissRoll['Yt'])
    y_test = np.array(SwissRoll['Cv'])
    X_test = np.array(SwissRoll['Yv'])

    X_train = np.transpose(X_train)
    y_train = np.transpose(y_train)
    X_test = np.transpose(X_test)
    y_test = np.transpose(y_test)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    return X_train, y_train, X_test, y_test


class DataGenerator:
    def __init__(self, data, target, batch_size, shuffle=True):
        self.shuffle = shuffle
        if shuffle:
            shuffled_indices = np.random.permutation(len(data))
        else:
            shuffled_indices = range(len(data))

        self.data = data[shuffled_indices]
        self.target = target[shuffled_indices]
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(data.shape[0] / batch_size))
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.num_batches:
            batch_data = self.data[self.counter * self.batch_size:(self.counter + 1) * self.batch_size]
            batch_target = self.target[self.counter * self.batch_size:(self.counter + 1) * self.batch_size]
            self.counter += 1
            return batch_data, batch_target
        else:
            if self.shuffle:
                shuffled_indices = np.random.permutation(len(self.target))
            else:
                shuffled_indices = range(len(self.target))

            self.data = self.data[shuffled_indices]
            self.target = self.target[shuffled_indices]

            self.counter = 0
            raise StopIteration
