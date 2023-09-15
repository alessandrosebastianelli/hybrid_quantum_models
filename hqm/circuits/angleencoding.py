from types import FunctionType
import pennylane as qml
import numpy as np
import sys

sys.path += ['.', './utils/']

from .circuit import QuantumCircuit

class BasicEntangledCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using a basic entangler
        circuit. 
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, dev : qml.devices = None) -> None:
        '''
            BasicEntangledCircuit constructor.  

            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            
            Returns:  
            --------  
            Nothing, a BasicEntangledCircuit object will be created.  
        '''

        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)
               
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.circuit      = self.circ(self.dev, self.n_qubits)
        

    @staticmethod
    def circ(dev : qml.devices, n_qubits : int) -> FunctionType:
        '''
            BasicEntangledCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'  
            - n_qubits : int  
                number of qubits for the quantum circuit  

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
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return measurements
    
        return qnode
    
class StronglyEntangledCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using a strongly entangler
        circuit.
    '''

    def __init__(self, n_qubits : int, n_layers : int, dev : qml.devices = None) -> None:
        '''
            StronglyEntangledCircuit constructor.  

            Parameters:  
            -----------  
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit    
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            
            Returns:  
            --------  
            Nothing, a StronglyEntangledCircuit object will be created.  
        '''        

        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)
        
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.circ(self.dev, self.n_qubits)
        
    @staticmethod
    def circ(dev : qml.devices, n_qubits : int) -> FunctionType:
        '''
            StronglyEntangledCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            - n_qubits : int  
                number of qubits for the quantum circuit  

            Returns:  
            --------  
            - qnode : qml.qnode  
                the actual PennyLane circuit  
        '''
        @qml.qnode(dev)
        def qnode(inputs : np.ndarray, weights : np.ndarray) -> list:
            '''
                PennyLane based quantum circuit composed of an angle embedding layer and a strongly entangler
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

            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return measurements
    
        return qnode

class RandomCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using a random
        circuit. 
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, dev : qml.devices = None) -> None:
        '''
            RandomCircuit constructor.  

            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            
            Returns:  
            --------  
            Nothing, a RandomCircuit object will be created.  
        '''

        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)
               
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.circuit      = self.circ(self.dev, self.n_qubits)

    @staticmethod
    def circ(dev : qml.devices, n_qubits : int) -> FunctionType:
        '''
            RandomCircuit static method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'  
            - n_qubits : int  
                number of qubits for the quantum circuit  

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
            
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.RandomLayers(weights, wires=range(n_qubits))
            measurements = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return measurements
    
        return qnode