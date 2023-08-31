import pennylane as qml

def torch_layer(circuit, n_qubits, n_layers):
    weight_shape = {"weights": (n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shape)
    return qlayer