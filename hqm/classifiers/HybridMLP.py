import pennylane as qml
import torch

import sys
sys.path += ['.', './layers/']


class HybridMLP(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()

        n_qubits = circuit.n_qubits
        self.fc_1 = torch.nn.Linear(n_qubits, n_qubits)
        self.qc_1 = circuit.qlayer
        self.fc_2 = torch.nn.Linear(n_qubits, n_qubits)
        self.softmax  = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.qc_1(x)
        x = self.fc_2(x)
        return self.softmax(x)
    
