from types import FunctionType
import pennylane as qml
import numpy as np
import warnings
import sys

sys.path += ['.', './layers/']

from hqm.layers.ai_interface import ai_interface

class BasicEntangledCircuit:
    '''
        This class implements a torch/keras quantum layer using a basic entangler
        circuit, as below:

        0: ──RX(x1)──RX(w1.1)─╭●───────╭X──RX(w1.n)─╭●───────╭X─┤  \<Z\>  
        1: ──RX(x2)──RX(w2.1)─╰X─╭●────│───RX(w2.n)─╰X─╭●────│──┤  \<Z\>  
        2: ...........................................................  
        3: ──RX(xm)──RX(wm.1)───────╰X─╰●──RX(wm.n)───────╰X─╰●─┤  \<Z\>  
    
    '''
    
    def __init__(self, n_qubits : int, n_layers : int, aiframework : str, dev : qml.device = None) -> None:
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

        # Checking for exceptions
        if aiframework not in ['torch', 'keras']: raise Exception(f"Accepted values for framerwork are 'torch' or 'keras', found {aiframework}")
        if n_qubits < 1: raise Exception(f"Number of qubits must be greater or equal than 1, found {n_qubits}")
        if n_layers < 1: raise Exception(f"Number of layers must be greater or equal than 1, found {n_layers}")

        # Set dev to 'default.qubit' if dev is None
        if dev is None: 
            dev = qml.device("default.qubit", wires=n_qubits)
            warnings.warn(f"Dev has been set to None, setting it to {dev}")

        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.aiframework  = aiframework
        self.dev          = dev
        self.weight_shape = {"weights": (n_layers, n_qubits)}
        self.circuit      = self.__circ(self.dev, self.n_qubits)
        self.qlayer       = ai_interface(circuit      = self.circuit, 
                                         weight_shape = self.weight_shape, 
                                         n_qubits     = self.n_qubits, 
                                         framework    = self.aiframework)

    @staticmethod
    def __circ(dev : qml.device, n_qubits : int) -> FunctionType:
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
    
class StronglyEntangledCircuit:
    '''
        This class implements a torch/keras quantum layer using a strongly entangler
        circuit, as below:

        0: ──RX(x1)──Rot(w1.1,w1.2,w1.3)─╭●───────╭X──Rot(w1.n-2,w1.n-1,w1.n)─╭●────╭X────┤  <Z>  
        1: ──RX(x2)──Rot(w2.1,w2.2,w2.3)─╰X─╭●────│───Rot(w2.n-2,w2.n-1,w2.n)─│──╭●─│──╭X─┤  <Z>  
        2: ................................................................................  
        3: ──RX(xm)──Rot(wm.1,wm.2,wm.3)───────╰X─╰●──Rot(wm.n-2,wm.n-1,wm.n)────╰X────╰●─┤  <Z>  
    '''

    def __init__(self, n_qubits : int, n_layers : int, aiframework : str, dev : qml.device = None) -> None:
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

        # Checking for exceptions
        if aiframework not in ['torch', 'keras']: raise Exception(f"Accepted values for framerwork are 'torch' or 'keras', found {aiframework}")
        if n_qubits < 1: raise Exception(f"Number of qubits must be greater or equal than 1, found {n_qubits}")
        if n_layers < 1: raise Exception(f"Number of layers must be greater or equal than 1, found {n_layers}")

        # Set dev to 'default.qubit' if dev is None
        if dev is None: 
            dev = qml.device("default.qubit", wires=n_qubits)
            warnings.warn(f"Dev has been set to None, setting it to {dev}")

        self.n_qubits     = n_qubits
        self.n_layers     = n_layers
        self.aiframework  = aiframework
        self.dev          = dev
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
        self.circuit      = self.__circ(self.dev, self.n_qubits)
        self.qlayer       = ai_interface(circuit      = self.circuit, 
                                         weight_shape = self.weight_shape, 
                                         n_qubits     = self.n_qubits, 
                                         framework    = self.aiframework)
        
    @staticmethod
    def __circ(dev : qml.device, n_qubits : int) -> FunctionType:
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
