from types import FunctionType
import pennylane as qml
import numpy as np
import random
import sys

sys.path += ['.', './ciruits/']

from hqm.circuits.circuit import QuantumCircuit

class RandomCircuitNoiseGenerator(QuantumCircuit):
    '''
        This class implements a quantum noise generator using a random circuit for each sample.
    '''
    
    def __init__(self, location : float, scale : float, n_qubits: int, n_layers : int, dev : qml.devices = None) -> None:
        '''
            RandomCircuitNoiseGenerator constructor.  

            Parameters:  
            -----------
            - location : float  
                location parameter translate the pdf, relative to the standard normal distribution
            - scale : float  
                scale parameter that stretches the pdf. The greater the magnitude, the greater the stretching. 
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit  
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'. Recommendent qml.device("default.mixed", wires=1, shots=1000).
            
            Returns:  
            --------  
            Nothing, a RandomCircuitNoiseGenerator object will be created.  
        '''

        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)

        self.shots     = dev.shots
        self.location  = location 
        self.scale     = scale
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.dev       = dev
        
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
        expval = self.circ(self.dev)
        n = self.location + self.scale*expval
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

            qml.RandomLayers(np.random.random((len(dev.wires),self.n_layers)), range(len(dev.wires)))
            return qml.expval(qml.PauliZ(0))

        return qnode()

class RandomCZNoiseGenerator(QuantumCircuit):
    '''
        This class implements a quantum noise generator using a random circuit
    '''
    
    def __init__(self, location : float, scale : float, n_qubits: int, n_layers : int, n_entangling_layers : int = 2, bit_string : bool = True, state_id : int = 0, dev : qml.devices = None) -> None:
        '''
            RandomCircuitNoiseGenerator constructor.  

            Parameters:  
            -----------
            - location : float  
                location parameter translate the pdf, relative to the standard normal distribution
            - scale : float  
                scale parameter that stretches the pdf. The greater the magnitude, the greater the stretching. 
            - n_qubits : int  
                number of qubits for the quantum circuit  
            - n_layers : int  
                number of layers for the quantum circuit 
            - n_entangling_layers : int  
                number of layers with entanglement
            - state_id : int
                name of the state to consider
            - dev : qml.device  
                PennyLane device on wich run quantum operations (dafault : None). When None it will be set
                to 'default.qubit'. Recommendent qml.device("default.mixed", wires=1, shots=1000).
            
            Returns:  
            --------  
            Nothing, a RandomCircuitNoiseGenerator object will be created.  
        '''

        super().__init__(n_qubits=n_qubits, n_layers=n_layers, dev=dev)

        self.shots               = dev.shots
        self.location            = location 
        self.scale               = scale
        self.n_qubits            = n_qubits
        self.n_layers            = n_layers
        self.n_entangling_layers = n_entangling_layers
        self.bit_string          = bit_string
        self.state_id            = state_id
        self.dev                 = dev
        self.circuit             = self.circ(self.dev)
        
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
        if self.bit_string:
            unitary_probs = self.circuit()
            state_id = np.random.choice(2**self.n_qubits, p=unitary_probs)
            expval = unitary_probs[state_id]
        else:
            expval = self.circuit()[self.state_id]
        
        n = self.location + self.scale*expval
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

            for _ in range(self.n_layers):
                for j in range(self.n_qubits):
                    
                    qml.Hadamard(wires=j)
                    rZ = 2*np.pi*random.random()
                    qml.RZ(rZ,wires=j)

                    rY = 2*np.pi*random.random()
                    qml.RY(rY,wires=j)

                    for _ in range(self.n_entangling_layers):
                        r = random.sample(range(0, self.n_qubits), 2)        
                        qml.CZ(wires=r)

                    #qml.CZ(wires=r)
                # Hadamards
                qml.Barrier()
            for i in range(self.n_qubits): qml.Hadamard(wires=i)
        
            return qml.probs(wires=range(0,self.n_qubits))

        return qnode