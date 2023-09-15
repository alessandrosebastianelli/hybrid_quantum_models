'''
    Work in progress
'''
from types import FunctionType
import pennylane as qml
import numpy as np
import sys

sys.path += ['.', './utils/']

from .circuit import QuantumCircuit

class BellmanCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using the bellman
        circuit. 
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, dev : qml.devices = None, encoding : str = 'angle') -> None:
        '''
            BellmanCircuit constructor.  

            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            Nothing, a BellmanCircuit object will be created.  
        '''
        
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)

        if encoding not in ['angle', 'amplitude']: raise(f"encoding can be angle or amplitude, found {encoding}")
        
        self.encoding     = encoding
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.circ(self.dev, self.n_qubits, self.n_layers, self.encoding)

    @staticmethod
    def circ(dev : qml.devices, n_qubits : int, n_layers : int, encoding : str) -> FunctionType:
        '''
            BellmanCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'  
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            - qnode : qml.qnode  
                the actual PennyLane circuit   
        '''
        @qml.qnode(dev)
        def qnode(inputs : np.ndarray, weights : np.ndarray) -> list:
            '''
                PennyLane based quantum circuit composed of an angle embedding layer and a basic entangler
                layer.  

                Parameters:  
                -----------  
                - inputs : np.ndarray  
                    array containing input values (can came from previous torch/keras layers or quantum layers)  
                - weights : np.ndarray  
                    array containing the weights of the circuit that are tuned during training, the shape of this
                    array depends on circuit's layers and qubits.   
                
                Returns:  
                --------  
                - measurements : list  
                    list of values measured from the quantum circuits  
            '''

            if encoding == 'angle':     qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if encoding == 'amplitude': qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

            for n in range(n_layers):
                qml.Hadamard(wires=0)
                
                # CNOT gates
                for q in range(1, n_qubits):
                    qml.CNOT(wires=[q-1, q])

                # Rot gates in the middle
                for q in range(n_qubits):
                    qml.Rot(weights[n, q, 0], weights[n, q, 1], weights[n, q, 2], wires=q)

                # CNOT gates
                for q in range(1, n_qubits):
                    qml.CNOT(wires=[n_qubits - q-1, n_qubits - q])

            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return measurements
    
        return qnode

class RealAmplitudesCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using the real amplitude circuit. 
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, dev : qml.devices = None, encoding : str = 'angle') -> None:
        '''
            RealAmplitudesCircuit constructor.  

            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            Nothing, a RealAmplitudesCircuit object will be created.  
        '''
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)

        if encoding not in ['angle', 'amplitude']: raise(f"encoding can be angle or amplitude, found {encoding}")
        
        self.encoding     = encoding
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.circ(self.dev, self.n_qubits, self.n_layers, self.encoding)

    @staticmethod
    def circ(dev : qml.devices, n_qubits : int, n_layers : int, encoding : str) -> FunctionType:
        '''
            RealAmplitudesCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'  
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - encoding : str  
                string representing the type of input data encoding in quantum circuit, can be 'angle' or 'amplitude'
            
            Returns:  
            --------  
            - qnode : qml.qnode  
                the actual PennyLane circuit   
        '''
        @qml.qnode(dev)
        def qnode(inputs : np.ndarray, weights : np.ndarray) -> list:
            '''
                PennyLane based quantum circuit composed of an angle embedding layer and a basic entangler
                layer.  

                Parameters:  
                -----------  
                - inputs : np.ndarray  
                    array containing input values (can came from previous torch/keras layers or quantum layers)  
                - weights : np.ndarray  
                    array containing the weights of the circuit that are tuned during training, the shape of this
                    array depends on circuit's layers and qubits.   
                
                Returns:  
                --------  
                - measurements : list  
                    list of values measured from the quantum circuits  
            '''

            if encoding == 'angle':     qml.AngleEmbedding(inputs, wires=range(n_qubits))
            if encoding == 'amplitude': qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

            for q in range(n_qubits): qml.Hadamard(wires=q)

            for n in range(n_layers):
                for q1 in range(0, n_qubits):
                    for q2 in range(0, n_qubits):
                        if q1 < q2:
                            qml.CNOT(wires=[q2, q1])
                
                # Rot gates in the middle
                for q in range(n_qubits):
                    qml.Rot(weights[n, q, 0], weights[n, q, 1], weights[n, q, 2], wires=q)


            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return measurements
    
        return qnode