import sys
sys.path += ['.', './hqm/']

from hqm.noise.gaussianlike import GaussianLikeNoiseGenerator
from hqm.noise.randomcircuit import RandomCircuitNoiseGenerator
import pennylane as qml

def test_GaussianLikeNoiseGenerator_output():
    gen = GaussianLikeNoiseGenerator(location=0, scale=1, dev=qml.device("default.mixed", wires=1, shots=1000))
    n = gen.generate_noise_array((64,30))
    assert n.shape == (64,30)

def test_RandomCircuitNoiseGenerator_output():
    N_QUBTS  = 2
    N_LAYERS = 20
    gen = RandomCircuitNoiseGenerator(location=0, scale=1, n_qubits=N_QUBTS, n_layers=N_LAYERS, dev=qml.device("default.qubit", wires=N_QUBTS))
    n = gen.generate_noise_array((64,30))
    assert n.shape == (64,30)
