import pennylane as qml

def ai_interface(circuit, weight_shape, n_qubits, framework):
    
    if framework == 'torch':
        qlayer = qml.qnn.TorchLayer(circuit, weight_shape)
    elif framework == 'keras':
        qlayer = qml.qnn.KerasLayer(circuit, weight_shape, output_dim=n_qubits)
    else:
        raise Exception('Framerwork can be only torch or keras!')
    return qlayer
