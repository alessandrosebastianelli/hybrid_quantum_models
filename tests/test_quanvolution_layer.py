import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit
from hqm.layers.quanvolution import Quanvolution2D
from hqm.utils.printer import Printer


import pennylane as qml
import numpy as np
import argparse
import torch


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Quanvolution layer')
    # Model's arguments
    parser.add_argument('--n_qubits',               type=int,   default=9)
    parser.add_argument('--n_layers',               type=int,   default=1)

    # Reading arguments
    args = parser.parse_args()
    
    
    N_QUBITS      = int(args.n_qubits)
    N_LAYERS      = int(args.n_layers)



    if N_QUBITS < 1:       raise Exception('n_qubits must be greater than 1, found {N_QUBITS}')
    if N_LAYERS < 1:       raise Exception('n_layers must be greater than 1, found {N_LAYERS}')


    #=======================================================================
    # Inizialize Hybrid Model
    #=======================================================================
    print('Initializing hybrid model', '\n')
    dev = qml.device("default.qubit", wires=N_QUBITS)

    qcircuit = BasicEntangledCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, dev=dev)
    quanv    = Quanvolution2D(qcircuit=qcircuit, filters=9, kernelsize=3, stride=1)
    Printer.draw_circuit(qcircuit)

    #=======================================================================
    # Applying Quanvolution2D
    #=======================================================================
    print('Applying Quanvolution2D', '\n')
    x = torch.rand(size=(2,3,16,16))
    o =  quanv.forward(x)
    print('Input Shape', x.shape, 'Output shape', o.shape)
