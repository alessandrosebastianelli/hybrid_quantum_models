import pennylane as qml
import numpy as np

def print_circuit(circuit):
    weights = np.random.random(size=circuit.weight_shape['weights'])
    inputs  = np.random.random(size=(circuit.n_qubits))
    print(qml.draw(circuit.circuit, expansion_strategy="device")(inputs, weights))

