import tensorflow as tf

class ClassicalModel():
    def __init__(self, n_holes, memory_size, hidden_layers, n_nodes, learning_rate):
        '''
        Initializes the DNN parameters.

        Parameters
        ----------
        n_holes (int):
            Number of outputs of the DNN. / Number of holes in the environment.
        memory_size (int):
            The size of the input to the DNN. / The amount of guesses that the agent is allowed to look back.
        hidden_layers (int):
            The amount of hidden layers in the DNN.
        n_nodes (int):
            The amount of nodes per hidden layer.
        learning_rate (float):
            The learning rate that the optimizer of the DNN uses in order to update the weights.
        '''
        self.n_holes = n_holes
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.hidden_layers = hidden_layers

    def initialize_model(self):
        '''
        Builds the deep neural network.
        '''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.n_nodes, activation='relu', input_shape=(self.memory_size,), kernel_initializer=tf.keras.initializers.GlorotUniform()))
        for _ in range(self.hidden_layers-1):
            model.add(tf.keras.layers.Dense(self.n_nodes, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()))
        model.add(tf.keras.layers.Dense(self.n_holes, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform()))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])

        return model