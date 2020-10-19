import numpy as np
import random
import math
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize
from scipy.optimize import fmin
from utils import *


class NeuralNet:
    """A simple neural network. 
    """

    def __init__(self, n_h_layers=2, n_nodes_hl=4):
        """
        Args:
            model_name (string): name of model
            n_layers (int): number of hidden layers
            n_features (int): number of features from input
            n_nodes_hl (int): Number of nodes in hidden layers
        """
        # self.n_features = 3
        self.n_h_layers = n_h_layers
        self.n_nodes_hl = n_nodes_hl
        # self.n_classes = 7

    def get_network(self):
        """Return network (nodes)

        Returns:
            [dictionary]: Each np array is a layer of nodes. First array is input features. 
        """
        return self.network

    def train_model(self, learning_rate=0.8, no_epochs=20_000):
        training_set = np.load("weights/training_colours_shuffled.npy")

        x = training_set[:, :-1]
        y = training_set[:, -1].astype(int)
        m = np.shape(x)[0]

        self.thetas = create_theta_dict(self.n_h_layers, self.n_nodes_hl)

        self.thetas, cost_history = gradient_descent(
            learning_rate, no_epochs, self.thetas, m, x, y
        )
        self.current_cost = cost_history[-1]

    def predict(self, features):
        """Make prediction given a set of features.

        Args:
            features (np.array): RGB values

        Returns:
            [np.array]: array with boolean classification
        """
        self.network = forward_propagation(features, self.thetas)
        prediction = np.argmax((self.network[max(self.network.keys())]))
        return prediction
