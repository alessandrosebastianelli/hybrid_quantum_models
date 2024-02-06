from types import FunctionType
import pennylane as qml
import numpy as np
import sys

sys.path += ['.', './ciruits/']

from hqm.circuits.circuit import QuantumCircuit

class GaussianLikeNoiseGenerator(QuantumCircuit):
    '''
        This class implements a torch/keras quantum layer using a basic entangler
        circuit. 
    '''
    
    def __init__(self, location : float, scale : float, dev : qml.devices = None) -> None:
        '''
            GaussianLikeNoiseGenerator constructor.  

            Parameters:  
            -----------
            - location : float  
                location parameter translate the pdf, relative to the standard normal distribution
            - scale : float  
                scale parameter that stretches the pdf. The greater the magnitude, the greater the stretching. 
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'. Recommendent qml.device("default.mixed", wires=1, shots=1000).
            
            Returns:  
            --------  
            Nothing, a GaussianLikeNoiseGenerator object will be created.  
        '''

        super().__init__(n_qubits=1, n_layers=1, dev=dev)

        self.shots     = dev.shots
        self.location  = location + 1/2
        self.scale     = scale * np.sqrt(self.shots/(1/2))
        self.circuit   = self.circ(self.dev)
    
    
    def generate_noise(self):
        '''
            GaussianLikeNoiseGenerator that generates on observable  

            Parameters:  
            -----------
            Nothing
            
            Returns:  
            --------  
            n : float
                gaussianlike quantum noise observable
        '''
        counts = self.circuit()
        n = self.scale*(counts[1]/self.shots-self.location)
        return n

    
    def generate_noise_array(self, shape:tuple):
        '''
            GaussianLikeNoiseGenerator that generates a matrix of observables

            Parameters:  
            -----------
            shape : tuple
                shape of the desired maxtrix of observables
            
            Returns:  
            --------  
            n : float
                gaussianlike quantum noise matrix of observables
        '''
        n = []
        for _ in range(np.prod(shape)): n.append(self.generate_noise())
        n = np.array(n)
    
        return n.reshape(shape)
    
    def circ(self, dev : qml.devices) -> FunctionType:
        '''
            GaussianLikeNoiseGenerator method that implements the quantum circuit.  

            Parameters:  
            -----------  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set  
                to 'default.qubit'   

            Returns:  
            --------  
            - qnode : qml.qnode  
                the actual PennyLane circuit   
        '''

        @qml.qnode(dev)
        def qnode() -> dict:
            '''
                PennyLane based quantum circuit composed of a Ry gate

                Parameters:  
                -----------  
                Nothing

                Returns:  
                --------  
                - counts : dict  
                    Sample from the supplied observable, with the number of shots determined from the dev.shots attribute of the corresponding device, 
                    returning the number of counts for each sample. If no observable is provided then basis state samples are returned directly 
                    from the device. Note that the output shape of this measurement process depends on the shots specified on the device. 
            '''

            qml.RY(np.pi/2, 0)
            return qml.counts(qml.PauliZ(0))
    
        return qnode