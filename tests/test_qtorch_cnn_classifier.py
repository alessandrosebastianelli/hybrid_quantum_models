import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.classification.hcnn import HybridLeNet5
from hqm.layers.basiclayer import BasicLayer

import pennylane as qml
import torch


def test_qtorch_cnn_classifier_output():

    N_QUBITS      = 12
    N_LAYERS      = 1
    BATCH_SIZE    = 16

    IN_SHAPE      = (3, 64,64)
    OU_DIM        = 10
    
    
    dev = qml.device("default.qubit", wires=N_QUBITS)
    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    model = HybridLeNet5(qlayer=BasicLayer(qcircuit, aiframework='torch'), in_shape=IN_SHAPE, ou_dim=OU_DIM)
    
    x = torch.rand((BATCH_SIZE,) + IN_SHAPE)
    o = model(x)

    assert o.shape == (BATCH_SIZE, OU_DIM)