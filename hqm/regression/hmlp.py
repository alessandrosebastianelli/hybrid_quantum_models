import sys
sys.path += ['.', './circuits/']

from hqm.circuits.circuit import QuantumCircuit
import torch


class BasicHybridMLPRegressor(torch.nn.Module):
    '''
        This class implements a basic hybrid multilayer perceptron for regression purposes.
        BasicHybridMLPRegressor is composed of a quantum layers stacked between two fully connected layers.
        The size of fully connected layers is set by means of in_dim and ou_dim.
    '''

    def __init__(self, qcircuit : QuantumCircuit, in_dim : int, ou_dim : int) -> None:
        '''
            BasicHybridMLPRegressor constructor.

            Parameters:
            -----------
            - qcircuit : hqm.circuits.circuit.QuantumCircuit
                hqm quantum circuit to be stacked between two fully connected layers
            - in_dim : int
                integer representing the input size for the first fully connected layer
            - ou_dim : int
                integer representing the output size of the hybrid model
            
            Returns:
            --------
            Nothing, a BasicHybridMLPRegressor object will be created.  
        '''
        super().__init__()

        if in_dim < 1: raise Exception(f"The parameter in_dim must be greater than 1, found {in_dim}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")

        n_qubits  = qcircuit.n_qubits
        self.fc_1 = torch.nn.Linear(in_dim, n_qubits)
        self.qc_1 = qcircuit.qlayer
        self.fc_2 = torch.nn.Linear(n_qubits, ou_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x : torch.Tensor):
        '''
            Torch forward method

            Parameters:
            -----------
            - x : torch.Tensor
                input for the torch model

            Returns:
            --------
            - x : torch.Tensor
                output from the torch model
        '''
        x = self.fc_1(x)
        x = self.qc_1(x)
        x = self.fc_2(x)
        out = self.tanh(x)
        return out
    
class MultiHybridMLPRegressor(torch.nn.Module):
    '''
        This class implements a hybrid multilayer perceptron with multiple quantum circuits for regression purposes.
        MultiHybridMLPRegressor is composed of several quantum layers stacked between two fully connected layers.
        The size of fully connected layers is set by means of in_dim and ou_dim.
    '''

    def __init__(self, qcircuits : list, in_dim : int, ou_dim : int) -> None:
        '''
            MultiHybridMLPRegressor constructor.

            Parameters:
            -----------
            - qcircuits : list
                list of hqm quantum circuits to be stacked between two fully connected layers
            - in_dim : int
                integer representing the input size for the first fully connected layer
            - ou_dim : int
                integer representing the output size of the hybrid model
            
            Returns:
            --------
            Nothing, a MultiHybridMLPRegressor object will be created.  
        '''
        super().__init__()

        if in_dim < 1: raise Exception(f"The parameter in_dim must be greater than 1, found {in_dim}")
        if ou_dim < 1: raise Exception(f"The parameter ou_dim must be greater than 1, found {ou_dim}")
        if len(qcircuits) < 1: raise Exception(f"Size of qcircuis must be greater than 1, found {len(qcircuits)}")

        n_qubits_0  = qcircuits[0].n_qubits
        n_qubits_1  = qcircuits[-1].n_qubits
        self.fc_1 = torch.nn.Linear(in_dim, n_qubits_0)
        self.qcs = [circ.qlayer for circ in qcircuits]
        self.fc_2 = torch.nn.Linear(n_qubits_1, ou_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, x : torch.Tensor):
        '''
            Torch forward method

            Parameters:
            -----------
            - x : torch.Tensor
                input for the torch model

            Returns:
            --------
            - x : torch.Tensor
                output from the torch model
        '''
        x = self.fc_1(x)
        for qc in self.qcs: x = qc(x)
        x = self.fc_2(x)
        out = self.tanh(x)
        return out