from typing import TypeVar, Generic
import pennylane as qml
import warnings

quantumcircuit = TypeVar("quantumcircuit")
class QuantumCircuit(Generic[quantumcircuit]):
    '''
        Basic QuantumCircuit object.
    '''
    def __init__(self, n_qubits : int, n_layers : int, aiframework : str, dev : qml.devices = None) -> None:
        '''
            QuantumCircuit parent class.  

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
            Nothing, a QuantumCircuit object will be created.  
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