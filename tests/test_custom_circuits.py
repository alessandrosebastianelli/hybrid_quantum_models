import sys
sys.path += ['.', './hqm/']

from hqm.circuits.customcircuits import BellmanCircuit, RealAmplitudesCircuit

import numpy as np

def test_BasicEntangledCircuit_angle_output():
    n_qubits = 2
    n_layers = 1
    circ     = BellmanCircuit(n_qubits=n_qubits, n_layers=n_layers, encoding='angle').circuit
    cinput   = np.random.randn(n_qubits)
    cweights = np.random.randn(n_layers, n_qubits, 3)
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits


def test_BasicEntangledCircuit_amplitude_output():
    n_qubits = 2
    n_layers = 1
    circ     = BellmanCircuit(n_qubits=n_qubits, n_layers=n_layers, encoding='amplitude').circuit
    cinput   = np.random.randn(2**n_qubits)
    cweights = np.random.randn(n_layers, n_qubits, 3)
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits

def test_RealAmplitudesCircuitt_angle_output():
    n_qubits = 2
    n_layers = 1
    circ     = RealAmplitudesCircuit(n_qubits=n_qubits, n_layers=n_layers, encoding='angle').circuit
    cinput   = np.random.randn(n_qubits)
    cweights = np.random.randn(n_layers, n_qubits, 3)
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits

def test_RealAmplitudesCircuit_amplitude_output():
    n_qubits = 2
    n_layers = 1
    circ     = RealAmplitudesCircuit(n_qubits=n_qubits, n_layers=n_layers, encoding='amplitude').circuit
    cinput   = np.random.randn(2**n_qubits)
    cweights = np.random.randn(n_layers, n_qubits, 3)
    out      = circ(cinput, cweights)
    assert len(out) == n_qubits