from helper import *
import pennylane as qml

n_qubits = 10
dev = qml.device('default.qubit', wires=n_qubits)

def circuit(weights, depth):
    weight_counter = 0
    for _ in range(depth - 1):
        for qubit in range(n_qubits):
            qml.RX(weights[qubit + weight_counter * (n_qubits)], wires=qubit)
        weight_counter += 1
        for qubit in range(n_qubits):
            qml.RY(weights[qubit + weight_counter * (n_qubits)], wires=qubit)
        weight_counter += 1
        for qubit in range(n_qubits):
            qml.RZ(weights[qubit + weight_counter * (n_qubits)], wires=qubit)
        weight_counter += 1
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, qubit+1])
        qml.CNOT(wires=[0, n_qubits-1])

@qml.qnode(dev, interface="tf")
def run_circuit():
    test = 1
    return test

def random_angle():
    phi = np.random.uniform(low=0.0,high=2*np.pi, size=1)
    return phi