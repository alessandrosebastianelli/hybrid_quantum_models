import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.recurrent import QGRU
from hqm.utils.printer import Printer

import pennylane as qml
import numpy as np
import argparse
import torch


def test_qgru_output():
    N_QUBITS      = 9
    N_LAYERS      = 1
    INPUTSIZE     = 1
    HIDDENSIZE    = 10
    TIME_DIM      = 20
    BATCH_SIZE    = 16

    dev = qml.device("lightning.qubit", wires=N_QUBITS, shots = 1000)
    qcircuit1 = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    qcircuit2 = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    qcircuit3 = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    qcircuits = [qcircuit1, qcircuit2, qcircuit3]
    qgru      = QGRU(qcircuits=qcircuits, inputsize=INPUTSIZE, hiddensize=HIDDENSIZE)
    x = torch.rand(size=(BATCH_SIZE,TIME_DIM,INPUTSIZE))
    o =  qgru(x)

    assert o.shape == (BATCH_SIZE,TIME_DIM,HIDDENSIZE)