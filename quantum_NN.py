from helper import *
import pennylane as qml

n_holes = 5
n_qubits = 2*n_holes
depth = 1
learning_rate = 0.01
dev = qml.device('default.qubit', wires=n_qubits)

def circuit(weights, scaling_parameter, actions, depth):
    weight_counter = 0
    # input-scaling layer
    for qubit in range(n_qubits):
        qml.RX(actions[qubit] * scaling_parameter, wires=qubit)
    for _ in range(depth):
        # variational parameters layer
        for qubit in range(n_qubits):
            qml.RX(weights[qubit + weight_counter * (n_qubits)], wires=qubit)
        weight_counter += 1
        # entangling layer
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, qubit+1])

@qml.qnode(dev, interface="tf")
def run_circuit(weights, scaling_parameters, actions, depth):
    circuit(weights, scaling_parameters, actions, depth)
    wire_indices = np.arange(0, n_qubits)
    return qml.probs(wires=wire_indices)

# initialize the weights and scaling parameters
n_scaling_parameters = n_qubits
scaling_parameter = 2*np.pi / n_holes

n_weights = n_qubits * depth
init_weights = np.random.uniform(low=-np.pi, high=np.pi, size=(n_weights))
weights = tf.cast(init_weights, tf.float32)
weights = tf.Variable(weights, trainable=True)

# define the loss function and optimizer that will be used to train the model with
loss_func = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)