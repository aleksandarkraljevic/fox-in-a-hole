import tensorflow as tf

class ClassicalModel():
    def __init__(self, n_holes, memory_size, n_nodes, learning_rate):
        self.n_holes = n_holes
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes

    def initialize_model(self):
        '''
        Build the model. It is a simple Neural Network consisting of 3 densely connected layers with Relu activation functions.
        The only argument is the learning rate.
        '''
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(self.n_nodes, activation='relu', input_shape=(self.memory_size,), kernel_initializer=tf.keras.initializers.GlorotUniform()),
          tf.keras.layers.Dense(self.n_nodes, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
          tf.keras.layers.Dense(self.n_holes, activation='linear', kernel_initializer=tf.keras.initializers.GlorotUniform())
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
        return model