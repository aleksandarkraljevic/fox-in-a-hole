# Classical fox in a hole reinforcement learning model, by Aleksandar Kraljevic

The code consists of seven .py files, and one .slurm file:
- classical_model.py:
    This is the file that contains most of the classical RL model. It handles the training of the DDQN.
- dnn.py:
    This file contains the code which initializes a deep neural network.
- evaluate_model.py
    This code is for evaluating experiments that have already been performed. This includes the plotting of such experiments, as well as evaluating the performance of an already trained model over many samples.
- experiment.py
    This makes use of the classical_model.py and dnn.py in order to run larger experiments at once, such as hyperparameter tuning.
- experiment_alice.py
    This file is nearly the same as experiment.py, however it has slight alterations to it as it was used to perform experiments via a computational cluster of Leiden University, called ALICE.
- experiment.slurm
    This file is used to run experiment_alice.py on ALICE.
- fox_in_a_hole.py
    This is the environment of the fox in a hole game. Other functions such as classical_model.py and helper.py interact with this code in order to play the game and receive rewards.
- helper.py
    This file contains a multitude of functions that are used by the other python files, in order to not clutter them. Think of functions whose job it is to plot data, compute results, compute exploration parameters, etc.

## Dependencies
The code uses some python packages that need to be installed to run:
- numpy
- matplotlib
- tensorflow
- seaborn
- pandas
- scipy

## Running the code
To run any of the python files, make sure to have all the .py files in the same folder. In addition to this, make sure to create three empty folders called "data", "models", and "plots". As the names suggest, these are the files that the data, models, and plots will be saved in.
