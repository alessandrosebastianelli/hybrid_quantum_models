import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.quanvolution import Quanvolution2D
from hqm.utils.printer import Printer

import pennylane as qml
import numpy as np
import argparse
import torch


def test_quanvolution_padding_same_output():
    N_QUBITS      = 9
    N_LAYERS      = 1
    FITLERS       = 9
    KERNELSIZE    = 3
    STRIDE        = 1
    IMG_SIZE      = 16
    BATCH_SIZE    = 2
    CHANNELS      = 3
    PADDING       = 'same'

    #=======================================================================
    # InizializeQuanvolution2D
    #=======================================================================
    print('Initializing hybrid model', '\n')
    dev = qml.device("lightning.qubit", wires=N_QUBITS)

    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    quanv    = Quanvolution2D(qcircuit=qcircuit, filters=FITLERS, kernelsize=KERNELSIZE, stride=STRIDE, padding=PADDING)

    #=======================================================================
    # Applying Quanvolution2D - padding same
    #=======================================================================

    x = torch.rand(size=(BATCH_SIZE,CHANNELS,IMG_SIZE,IMG_SIZE))
    o =  quanv(x)

    assert o.shape == (BATCH_SIZE, FITLERS, IMG_SIZE, IMG_SIZE)#int((IMG_SIZE - KERNELSIZE)/STRIDE + 1), int((IMG_SIZE - KERNELSIZE)/STRIDE + 1))

def test_quanvolution_padding_valid_output():
    N_QUBITS      = 9
    N_LAYERS      = 1
    FITLERS       = 9
    KERNELSIZE    = 3
    STRIDE        = 1
    IMG_SIZE      = 16
    BATCH_SIZE    = 2
    CHANNELS      = 3
    PADDING       = 'valid'

    #=======================================================================
    # InizializeQuanvolution2D
    #=======================================================================
    print('Initializing hybrid model', '\n')
    dev = qml.device("lightning.qubit", wires=N_QUBITS)

    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    quanv    = Quanvolution2D(qcircuit=qcircuit, filters=FITLERS, kernelsize=KERNELSIZE, stride=STRIDE, padding=PADDING)

    #=======================================================================
    # Applying Quanvolution2D - padding same
    #=======================================================================

    x = torch.rand(size=(BATCH_SIZE,CHANNELS,IMG_SIZE,IMG_SIZE))
    o =  quanv(x)

    assert o.shape == (BATCH_SIZE, FITLERS, int((IMG_SIZE - KERNELSIZE)/STRIDE + 1), int((IMG_SIZE - KERNELSIZE)/STRIDE + 1))
    