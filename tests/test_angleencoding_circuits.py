import sys
sys.path += ['.', './hqm/']

from hqm.circuits.angleencoding import BasicEntangledCircuit, StronglyEntangledCircuit, RandomCircuit

import numpy as np

def test_BasicEntangledCircuit_output():
    n_qubits = 2
    n_layers = 1
    circ = BasicEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers).circuit
    cinput   = np.zeros((n_qubits))
    cweights = np.zeros((n_layers,n_qubits))
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits


def test_StronglyEntangledCircuit_output():
    n_qubits = 2
    n_layers = 1
    circ = StronglyEntangledCircuit(n_qubits=n_qubits, n_layers=n_layers).circuit
    cinput   = np.zeros((n_qubits))
    cweights = np.zeros((n_layers,n_qubits, 3))
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits


def test_RandomCircuit_output():
    n_qubits = 2
    n_layers = 1
    circ = RandomCircuit(n_qubits=n_qubits, n_layers=n_layers).circuit
    cinput   = np.zeros((n_qubits))
    cweights = np.zeros((n_layers,n_qubits))
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits