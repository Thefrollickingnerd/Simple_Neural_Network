import numpy as np
from neural_net import *
from utils import *


""" Features (RGB values)"""
feature1 = np.array([[148, 0, 211]], dtype=np.float128)
feature2 = np.array([[255, 0, 0]], dtype=np.float128)
""" Initialise class """
neural = NeuralNet(2, 4)

neural.train_model(0.8, 20_000)

colour_dict = {
    0: "Pink",
    1: "Purple",
    2: "Blue",
    3: "Green",
    4: "Yellow",
    5: "Orange",
    6: "Red",
}

print(
    f"Model prediction for [148], [0], [211] : {colour_dict[neural.predict(feature1)]}"
)
# print(neural.get_network())

print(f"Model prediction for [255], [0], [0] : {colour_dict[neural.predict(feature2)]}")
# print(neural.get_network())
