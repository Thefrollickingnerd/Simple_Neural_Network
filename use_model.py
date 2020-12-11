import numpy as np
from neural_net import *
from utils import *


""" Features (RGB values)"""
try_colour = np.array([[0, 255, 128]], dtype=np.float128)  # R G B

""" Initialise class """
neural = NeuralNet([2, 2])
neural.train_model("weights/training_colours_shuffled.npy", 0.8, 20_000)

colour_dict = {
    0: "Pink",
    1: "Purple",
    2: "Blue",
    3: "Green",
    4: "Yellow",
    5: "Orange",
    6: "Red",
}

print(f"The final cost is : {neural.current_cost}")
print(f"Model prediction for {try_colour} : {colour_dict[neural.predict(try_colour)]}")

