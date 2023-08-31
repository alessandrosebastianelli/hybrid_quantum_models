import pennylane as qml

class BasicEntangledCircuit:
    
    def __init__(self, n_qubits, n_layers, dev=None):
        
        if dev is None: dev = qml.device("default.qubit", wires=n_qubits)
        
        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.dev          = dev
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.circuit      = self.__circ(self.dev, self.n_qubits)
        self.torch_layer  = qml.qnn.TorchLayer(self.circuit, self.weight_shape)

    @staticmethod
    def __circ(dev, n_qubits):
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
        return qnode
    
class StronglyEntangledCircuit:

    def __init__(self, n_qubits, n_layers, dev=None) -> None:
        
        if dev is None: dev = qml.device("default.qubit", wires=n_qubits)

        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.dev          = dev
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.__circ(self.dev, self.n_qubits)
        self.torch_layer  = qml.qnn.TorchLayer(self.circuit, self.weight_shape)
    
    @staticmethod
    def __circ(dev, n_qubits):
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
    
        return qnode
