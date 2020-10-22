# Simple_Neural_Network
A neural network built from the ground up. 
Comes pre-packaged with a dataset of colours and functionality to predict what colour of the rainbow is closest to a colour passed by the user.

* Training set of colours were collected from this website [Rapid tables](https://www.rapidtables.com/web/color/).

## Usage 
This neural net is very simple to use.
  ### Initialise model 
    * Initialise the class and pass it a list, the number of features and the number of classes. 
    * Each element in the list is the number of nodes for that hidden layer - [2,3] == 2 nodes in hidden layer 1, 3 in hidden layer 2.
    * Number of features and classes are set to 3 and 7 respectively by default for the colour prediction model. 
  ### Train model 
    * Pass the filepath to the dataset stored in a numpy file. Defaults to dataset provided. 
    * Pass the learning rate and the number of epochs.
    * If colour_model == True, then method will treat the dataset like the default data and normalise it. 
  ### Predict 
    * Pass a numpy array. If colour_model == true then predict will assume default model and normalise accordingly. 
  ### Get network
    * This returns the network after the most recent prediction as a dictionary. 
  
  To use your own data, pull repo down and save a dataset into the weights folder and remember to set `colour model = False`.
