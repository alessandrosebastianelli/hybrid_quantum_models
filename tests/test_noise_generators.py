import sys
sys.path += ['.', './hqm/']

from hqm.noise.gaussianlike import GaussianLikeNoiseGenerator
from hqm.noise.randomcircuit import RandomCircuitNoiseGenerator, RandomCZNoiseGenerator
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

def test_RandomCZtNoiseGenerator_output():
    N_QUBTS  = 5
    N_LAYERS = 2
    gen = RandomCZNoiseGenerator(location=0, scale=1, n_qubits=N_QUBTS, n_layers=N_LAYERS, dev=qml.device("default.qubit", wires=N_QUBTS))
    n = gen.generate_noise_array((64,30))
    assert n.shape == (64,30)


if __name__ == '__main__':
    N_QUBTS  = 5
    N_LAYERS = 2
    gen = RandomCZNoiseGenerator(location=0, scale=1, n_qubits=N_QUBTS, n_layers=N_LAYERS, dev=qml.device("default.qubit", wires=N_QUBTS))
    n = gen.generate_noise_array((32,32))

    gen = RandomCZNoiseGenerator(location=0, scale=3, n_qubits=N_QUBTS, n_layers=N_LAYERS, dev=qml.device("default.qubit", wires=N_QUBTS))
    n2 = gen.generate_noise_array((32,32))

    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(n2.flatten(), 200, label='n2')
    plt.hist(n.flatten(), 200, label='n1')
    plt.xlim([0,1])
    plt.legend()
    plt.show()