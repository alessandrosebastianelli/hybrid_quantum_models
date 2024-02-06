import sys
sys.path += ['.', './hqm/']

from hqm.noise.gaussianlike import GaussianLikeNoiseGenerator
import pennylane as qml

def test_GaussianLikeNoiseGenerator_output():
    gen = GaussianLikeNoiseGenerator(location=0, scale=1, dev=qml.device("default.mixed", wires=1, shots=1000))
    n = gen.generate_noise_array((64,30))
    assert n.shape == (64,30)
