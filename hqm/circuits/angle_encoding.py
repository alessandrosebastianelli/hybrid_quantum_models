from types import FunctionType
import pennylane as qml
import numpy as np
import warnings
import sys

sys.path += ['.', './layers/']

from .circuit import QuantumCircuit
from hqm.layers.ai_interface import ai_interface

class BasicEntangledCircuit(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using a basic entangler
        circuit. 
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, aiframework : str, dev : qml.devices = None) -> None:
        '''
            BasicEntangledCircuit constructor.  

            Parameters:  
            -----------
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - aiframework : str  
                string representing the AI framework in use, can be 'torch' or 'keras'. This will create  
                a compatible trainable layer for the framework.   
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            
            Returns:  
            --------  
            Nothing, a BasicEntangledCircuit object will be created.  
        '''
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, aiframework=aiframework, dev=dev)
               
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.circuit      = self.circ(self.dev, self.n_qubits)
        self.qlayer       = ai_interface(circuit      = self.circuit, 
                                         weight_shape = self.weight_shape, 
                                         n_qubits     = self.n_qubits, 
                                         aiframework  = self.aiframework)

    @staticmethod
    def circ(dev : qml.device, n_qubits : int) -> FunctionType:
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

    def __init__(self, n_qubits : int, n_layers : int, aiframework : str, dev : qml.devices = None) -> None:
        '''
            StronglyEntangledCircuit constructor.  

            Parameters:  
            -----------  
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - aiframework : str  
                string representing the AI framework in use, can be 'torch' or 'keras'. This will create
                a compatible trainable layer for the framework.  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'  
            
            Returns:  
            --------  
            Nothing, a StronglyEntangledCircuit object will be created.  
        '''        
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, aiframework=aiframework, dev=dev)
        
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.circ(self.dev, self.n_qubits)
        self.qlayer       = ai_interface(circuit      = self.circuit, 
                                         weight_shape = self.weight_shape, 
                                         n_qubits     = self.n_qubits, 
                                         aiframework  = self.aiframework)
        
    @staticmethod
    def circ(dev : qml.device, n_qubits : int) -> FunctionType:
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
