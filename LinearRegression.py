import numpy as np


def gradient_linear(x, y, theta):
    """
    gradient_linear is a function with a close formula for the gradient of the residual errors (D_m,D_b) when
                m is the slope and b is the bias/offset from the x-axis.
                X=x1|x2
                   x1 = (a0,b0)
    :param x:
    :param y:
    :param w:
    :return:
    """
    samples = x.shape[0]  # the number of samples
    m, b = theta
    y_pred = m * x + b
    d_m = (-2 / samples) * np.sum(x @ (y - y_pred))  # Derivative wrt m
    d_b = (-2 / samples) * np.sum(y - y_pred)  # Derivative wrt c

    return np.array([d_m, d_b])


def loss_func_SGD(x, theta, mini_batch=4, learning_rate=0.001):
    """
    loss_func_SGD is a function that finding the minimum of the loss function, given the gradient of the loss function
    and the current weight. The fuction is finding the minimum using Stochastic Gradient Decent method
    that updates the W that minimize the loss function for the current mini batch images.
    :param loss_grad: The close formula of the gradient of the loss function.
    :param x: The data X=[x1|x2|x3...|xn], dimensions: n x m
    :param w: The weights, the parameters that we try to find. If is a linear regression then w=[m,b] when y=mx+b
    :param c: The matrix C=[c1|c2|c3|...|cn] when ci is a {0,1}^n vector, when every entry j is 1 if the ground truth of
                image xj is class i and 0 otherwise.
    :param mini_batch: The amount of images we want to use to update the weights for the Stochastic Gradient Decent
                        method.
    :param learning_rate: The
    :return:
    """

    new_theta = theta.copy()
    iteration = 1
    # if mini_batch == 1:
    #     batch_begin = 0
    #     batch_end = 1
    batch_begin = iteration * mini_batch - mini_batch
    batch_end = (iteration * mini_batch)
    while batch_end <= x.shape[1]:  # update the weights for one epoch
        # new_m = m - learning_rate*f_m
        # new_b = b - learning_rate*f_b
        # new_w = [m - learning_rate*f_m, b - learning_rate*f_b]
        new_theta = new_theta - learning_rate * gradient_linear(x[0, batch_begin:batch_end],
                                                                x[1, batch_begin:batch_end],
                                                                new_theta)
        iteration += 1
        batch_begin = iteration * mini_batch - mini_batch + 1
        batch_end = (iteration * mini_batch) + 1
    # update the weights for the last data if it wasn't used in the iterations because of the batch size.
    new_theta = new_theta - learning_rate * gradient_linear(x[0, x.shape[1] - batch_end: x.shape[1]],
                                                    x[1, x.shape[1] - batch_end: x.shape[1]], new_theta)
    return new_theta

