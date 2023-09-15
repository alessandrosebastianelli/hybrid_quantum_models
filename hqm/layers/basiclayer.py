import sys

sys.path += ['.', './utils/', '/circuits/']

from hqm.circuits.circuit import QuantumCircuit
from hqm.utils.aiinterface import AIInterface


class BasicLayer:
    '''
        Basic Quantum Layer
    '''

    def __init__(self, qcircuit : QuantumCircuit, aiframework : str) -> None:
        '''
        BasicLayer constructor.  

        Parameters:  
        -----------  
        - qcircuit : hqm.circuits.circuit.QuantumCircuit  
            QuantumCircuit object to be embedded into the quantum layer
        - aiframework : str    
            string representing the AI framework in use, can be 'torch' or 'keras'. This will create  
            a compatible trainable layer for the framework.   

        Returns:    
        --------     
        Nothing, a BasicLayer object will be created.  
        '''
        
        if aiframework not in ['torch', 'keras']: raise Exception(f"Accepted values for framerwork are 'torch' or 'keras', found {aiframework}")

        self.aiframework = aiframework
        self.n_qubits    = qcircuit.n_qubits
        self.qlayer      = AIInterface.network_layer(
                                circuit      = qcircuit.circuit, 
                                weight_shape = qcircuit.weight_shape, 
                                n_qubits     = qcircuit.n_qubits, 
                                aiframework  = self.aiframework
                            )