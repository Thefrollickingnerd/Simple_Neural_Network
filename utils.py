import numpy as np
from numpy.random import randint
from numpy.random import random_sample
from numpy.random import rand
from numpy.random import uniform


def create_theta_dict(n_h_layers, n_nodes_hl):
    """
    Create theta matrices inside dictionary. 
    Built on the assumption the final layer has 7 nodes. 
    set matrix elements to random value between -1 and 1.
    For n layers there are n-1 theta matrices.
    """
    n_features = 3
    n_outnodes = 7

    thetas = dict()
    thetas[0] = uniform(-1, 1, size=(n_nodes_hl, n_features + 1)).astype(
        dtype=np.float128
    )
    for i in range(1, n_h_layers):
        thetas[i] = uniform(-1, 1, size=(n_nodes_hl, n_nodes_hl + 1)).astype(
            dtype=np.float128
        )
    thetas[n_h_layers] = uniform(-1, 1, size=(n_outnodes, n_nodes_hl + 1)).astype(
        dtype=np.float128
    )
    return thetas


def forward_propagation(features, thetas):
    m = np.shape(features)[0]
    L1 = np.concatenate(
        (np.ones((1, m), dtype=np.float128), ((features - 127.5) / 127.5).T), axis=0
    )
    network = dict()
    network[0] = L1

    # === Loop through layers of network === #
    for i in range(0, len(thetas) - 1):
        Z = thetas[i] @ network[i]
        A = activation_func(Z)
        network[i + 1] = np.concatenate((np.ones((1, m), dtype=np.float128), A), axis=0)

    # === Final layer === #
    Z = thetas[len(thetas) - 1] @ network[len(thetas) - 1]
    A = activation_func(Z)
    network[len(thetas)] = A

    return network


def sigmoid_grad(z):
    """Gradient of sigmoid function

    Args:
        z (np array or float): [description]

    Returns:
        [np array]: gradient of sigmoid function of z where z == theta*layer.
    """
    ones = np.ones(z.shape)
    s1 = activation_func(z)
    s2 = ones - (activation_func(z))
    sig_grad = s1 * s2
    return sig_grad


def activation_func(x):
    """
    Apply sigmoid function on input. Can apply to matrices.

    Args:
        x (int/float/list/matrix): list or matrix of numbers.

    Returns:
        [x]: Same 2, size as input, sigmoid applied.
    """
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid


def transform_target(y):
    """
    Take y vector and transform to matrix with a 1 at 
    index corresponding to output ie 
    y[i] = 2 -> Y[i,:] = [0,0,1,0,0,0,0]
    """
    Y = np.zeros((7, len(y)))
    for i, row in enumerate(y):
        Y[row, i] = 1
    return Y


def cost_func(network, thetas, x, y):
    """
    Find cost of predictions 

    Returns:
        [float]: distance between predicted and known 
    """
    m = np.shape(x)[0]
    pred = network[max(network.keys())]
    Y = transform_target(y)

    cost = -1 / m * np.sum(Y * np.log(pred) + (1 - Y) * np.log(1 - pred))

    return cost


def back_propagate(network, m, target, thetas):
    """
    Back propagation of neural network 
    """
    thet_grad = dict()
    pred = network[max(network.keys())]
    Y = transform_target(target)

    delta = {max(network.keys()): pred - Y}
    thet_grad[max(network.keys()) - 1] = (
        1 / m * (delta[max(network.keys())] @ network[max(network.keys()) - 1].T)
    )

    for i in range(max(network.keys()) - 1, 0, -1):
        delta[i] = (delta[i + 1].T @ thetas[i][:, 1:]).T * sigmoid_grad(
            thetas[i - 1] @ network[i - 1]
        )
        thet_grad[i - 1] = 1 / m * (delta[i] @ network[i - 1].T)

    return thet_grad


def gradient_descent(learning_rate, no_epochs, thetas, m, x, y):

    J_history = []
    alpha = learning_rate
    for i in range(no_epochs):
        network = forward_propagation(x, thetas)
        cost = cost_func(network, thetas, x, y)
        J_history.append(cost)

        thet_grad = back_propagate(network, m, y, thetas)
        for i in thetas.keys():
            thetas[i] = thetas[i] - (alpha * thet_grad[i])

    return thetas, J_history


def unroll_thetas(thetas):
    unrolled = np.array([])
    dimensions = []

    for i in thetas.keys():
        dimensions.append(thetas[i].shape)
        unrolled = np.append(unrolled, thetas[i].flatten())
    return unrolled, dimensions


def reroll_thetas(unrolled, dimensions):
    rolled = {}
    start = 0
    for i, mat in enumerate(dimensions):
        elements = [unrolled[start : start + mat[0] * mat[1]]]
        rolled[i] = np.reshape(elements, mat)
        start = start + mat[0] * mat[1]
    return rolled
