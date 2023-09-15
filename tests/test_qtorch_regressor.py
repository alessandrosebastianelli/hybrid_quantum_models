import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import StronglyEntangledCircuit
from hqm.regression.hmlp import BasicHybridMLPRegressor
from hqm.layers.basiclayer import BasicLayer

import matplotlib.pyplot as plt
import pennylane as qml
import torch

def test_qtorch_regressor_output():
    N_QUBITS      = 12
    N_LAYERS      = 4
    IN_DIM        = 20
    OU_DIM        = 20
    BATCH_SIZE    = 16

    dev = qml.device("default.qubit", wires=N_QUBITS)
    qcircuit = StronglyEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    model = BasicHybridMLPRegressor(BasicLayer(qcircuit, aiframework='torch'), in_dim=IN_DIM, ou_dim=OU_DIM)
    
    x = torch.rand((BATCH_SIZE,   IN_DIM))
    o = model(x)

    assert o.shape == (BATCH_SIZE, OU_DIM)