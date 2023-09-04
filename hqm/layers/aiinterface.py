import pennylane as qml

class AIInterface:
    '''

    '''

    @staticmethod
    def network_layer(circuit : qml.qnode, weight_shape : dict, n_qubits : int, aiframework : str) -> qml.qnn.TorchLayer | qml.qnn.KerasLayer:
        '''

            Parameters:
            -----------
            - circuit : qml.qnode
            - weight_shape : dict
            - n_qubits : int
            - aiframeworkks : str

            Returns:
            --------
            - qlayer : qml.qnn.TorchLayer or qml.qnn.KerasLayer
        '''

        if   aiframework not in ['torch', 'keras']: raise Exception(f"Accepted values for framerwork are 'torch' or 'keras', found {aiframework}")
        if   aiframework == "torch": qlayer = qml.qnn.TorchLayer(circuit, weight_shape)
        elif aiframework == "keras": qlayer = qml.qnn.KerasLayer(circuit, weight_shape, output_dim=n_qubits)
        else: raise Exception(f"Framerwork can be only torch or keras, found {aiframework}!")
        
        return qlayer
