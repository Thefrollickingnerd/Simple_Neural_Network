import numpy as np
import random
import math
from utils import *


class NeuralNet:
    """A simple neural network. 
    """

    def __init__(self, layers, n_features=3, n_classes=7):
        """
        Args:
            layers (list) : each element is the number of nodes for that hidden layer.
            n_features (int) : number of features
            n_classes (int) : number of classes (nodes in output layer)
        """
        self.n_features = n_features
        self.layers = layers
        self.n_classes = n_classes

    def get_network(self):
        """Return network (nodes)

        Returns:
            [dictionary]: Each np array is a layer of nodes. First array is input features. 
        """
        return self.network

    def train_model(
        self,
        dataset="weights/training_colours_shuffled.npy",
        learning_rate=0.8,
        no_epochs=20_000,
        colour_model=True,
    ):
        """
        Trains model from dataset provided by user. Dataset must be a numpy array with final column the target variable.

        Args:
            learning_rate (float, optional): Learning rate, used in gradient descent. Defaults to 0.8.
            no_epochs ([type], optional): Number of epochs is the number of forward/backward propagations of 
            the network over the entire dataset. Defaults to 20_000.
        """
        training_set = np.load(dataset)

        x = training_set[:, :-1]
        y = training_set[:, -1].astype(int)
        m = np.shape(x)[0]

        if colour_model:
            x = (x - 127.5) / 127.5

        self.thetas = create_theta_dict(self.layers, self.n_features, self.n_classes)

        self.thetas, cost_history = gradient_descent(
            learning_rate, no_epochs, self.thetas, m, x, y
        )
        self.current_cost = cost_history[-1]

    def predict(self, features, colour_model=True):
        """Make prediction given a set of features.

        Args:
            features (np.array): RGB values

        Returns:
            [np.array]: array with boolean classification
        """
        if colour_model:
            features = (features - 127.5) / 127.5
        self.network = forward_propagation(features, self.thetas)
        prediction = np.argmax((self.network[max(self.network.keys())]))
        return prediction
