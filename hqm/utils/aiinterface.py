import pennylane as qml
import typing

class AIInterface:
    '''
        This class implements the inteface between quantum circuits implemented in PennyLane and the two most popular DL libraries,
        Pytorch and Tensorflow. 
    '''

    @staticmethod
    def network_layer(circuit : qml.qnode, weight_shape : dict, n_qubits : int, aiframework : str) -> typing.Union[qml.qnn.TorchLayer, qml.qnn.KerasLayer]:
        '''
            Static methods that embedd quantum layer into a torch or a keras layer.  

            Parameters:  
            -----------  
            - circuit : qml.qnode  
                pennylane circuit to be embedded   
            - weight_shape : dict  
                shape of the trainalbe weights, it is derived from hqm.circuits.circuit.QuantumCircuit  
            - n_qubits : int  
                integer representing the number of qubits used for the circuit  
            - aiframeworkks : str  
                string representing which in wich ai framework the circuit will be embedded, can be 'torch' or 'keras'  

            Returns:   
            --------  
            - qlayer : qml.qnn.TorchLayer or qml.qnn.KerasLayer  
        '''

        if   aiframework not in ['torch', 'keras']: raise Exception(f"Accepted values for framerwork are 'torch' or 'keras', found {aiframework}")
        if   aiframework == "torch": qlayer = qml.qnn.TorchLayer(circuit, weight_shape)
        elif aiframework == "keras": qlayer = qml.qnn.KerasLayer(circuit, weight_shape, output_dim=n_qubits)
        else: raise Exception(f"Framerwork can be only torch or keras, found {aiframework}!")
        
        return qlayer
