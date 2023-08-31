import pennylane as qml
import torch

import sys
sys.path += ['.', './layers/']


class BasicMLP(torch.nn.Module):
    def __init__(self, circuit, in_dim, ou_dim):
        super().__init__()

        n_qubits = circuit.n_qubits
        self.fc_1 = torch.nn.Linear(in_dim, n_qubits)
        self.qc_1 = circuit.qlayer
        self.fc_2 = torch.nn.Linear(n_qubits, ou_dim)
        self.tanh  = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc_1(x)
        x = self.qc_1(x)
        x = self.fc_2(x)
        return self.tanh(x)
    
