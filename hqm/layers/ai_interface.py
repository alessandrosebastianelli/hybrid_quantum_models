import pennylane as qml

def ai_interface(circuit, weight_shape, n_qubits, aiframework):
    
    if aiframework == "torch":
        qlayer = qml.qnn.TorchLayer(circuit, weight_shape)
    elif aiframework == "keras":
        qlayer = qml.qnn.KerasLayer(circuit, weight_shape, output_dim=n_qubits)
    else:
        raise Exception(f"Framerwork can be only torch or keras, found {aiframework}!")
    return qlayer
