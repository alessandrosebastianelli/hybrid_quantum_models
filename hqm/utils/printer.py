import sys
sys.path += ['.', './circuits/']

import pennylane as qml
import numpy as np

from hqm.circuits.circuit import QuantumCircuit

class Printer:
    '''
        This class contains static methods to print quantum circuit information.
    '''

    @staticmethod
    def draw_circuit(circuit : QuantumCircuit) -> str:
        '''
            Draw circuit structure.  

            Parameters:  
            -----------  
            - circuit : hqm.circuits.circuit.QuantumCircuit  
                hqm circuit to be drawn  
            
            Return:  
            -------  
            - str_circ : str  
                string containing circuit structure  

        '''
        
        weights  = np.random.random(size=circuit.weight_shape['weights'])
        inputs   = np.random.random(size=(circuit.n_qubits))
        str_circ = qml.draw(circuit.circuit, expansion_strategy="device")(inputs, weights)
        print(str_circ)
        return str_circ
